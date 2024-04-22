import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as tf
import numpy as np
from torch.autograd import Variable
from timeit import default_timer as timer

from utils import calc_metrics, prepare_mnist, weight_schedule
from model import convnet
from dataset import mnist
from config import Config
from loss import temporal_loss


class trainer:
    """
    A class that represents a trainer for a deep learning model.

    Attributes:
        model: The machine learning model to be trained.
        seed: The seed value for random number generation.
        kwargs: Additional keyword arguments for the trainer.

    Methods:
        train: Trains the machine learning model.
        test: Tests the trained model.
        save: Saves the trained model.
        load: Loads a pre-trained model.
        get_model: Returns the machine learning model.
        get_seed: Returns the seed value.
        get_kwargs: Returns the additional keyword arguments.
        set_model: Sets the machine learning model.
        set_seed: Sets the seed value.
        set_kwargs: Sets the additional keyword arguments.
    """

    def __init__(self, model, seed, **kwargs):
        self.model = model
        self.seed = seed
        # self.kwargs = kwargs
        self.dataset = kwargs["dataset"]
        self.config = kwargs["configs"]
        self.torch_dataset = datasets.MNIST  # load train data
        # setup param optimization
        self.train_dataset, self.test_dataset = self.get_data()
        self.train_loader,self.test_loader,_ = self.prepare_dataloader()
        # print(kwargs['dataloader']
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    def _iteration_train(
        self,
        model,
        epoch,
        optimizer,
        outputs,
        z
    ):
        # print("processing epoch {}".format(epoch))
        # print("the outputs vector shape is {}".format(outputs.shape))
        # print("the z vector shape is {}".format(z.shape))
        max_epochs= self.config.max_epochs
        max_val=self.config.max_val
        ramp_up_mult=self.config.ramp_up_mult
        k=self.config.k
        batch_size=self.config.batch_size
        num_epochs=self.config.num_epochs
        c=self.config.c
        alpha=self.config.alpha
        t = timer()
        n_samples = len(self.train_dataset)
        # evaluate unsupervised cost weight
        w = weight_schedule(epoch, max_epochs, max_val, ramp_up_mult, k, n_samples)

        if (epoch + 1) % 10 == 0:
            print("unsupervised loss weight : {}".format(w))

        # turn it into a usable pytorch object
        w = torch.autograd.Variable(torch.FloatTensor([w]), requires_grad=False)

        l = []
        supl = []
        unsupl = []
        for i, (images, labels) in enumerate(self.train_loader):
            images = Variable(images)
            labels = Variable(labels, requires_grad=False)

            # get output and calculate loss
            optimizer.zero_grad()
            out = model(images)
            zcomp = Variable(
                z[i * batch_size : (i + 1) * batch_size], requires_grad=False
            )
            loss, suploss, unsuploss, nbsup = temporal_loss(out, zcomp, w, labels)

            # save outputs and losses
            outputs[i * batch_size : (i + 1) * batch_size] = out.data.clone()
            # print(loss.data)
            l.append(loss.data.item())
            supl.append(nbsup * suploss.data.item())
            unsupl.append(unsuploss.data.item())

            # backprop
            loss.backward()
            optimizer.step()

            # print loss
            if (epoch + 1) % 10 == 0:
                if i + 1 == 2 * c:
                    print(
                        "Epoch [%d/%d], Step [%d/%d], Loss: %.6f, Time (this epoch): %.2f s"
                        % (
                            epoch + 1,
                            num_epochs,
                            i + 1,
                            len(self.train_dataset) // batch_size,
                            np.mean(l),
                            timer() - t,
                        )
                    )
                elif (i + 1) % c == 0:
                    print(
                        "Epoch [%d/%d], Step [%d/%d], Loss: %.6f"
                        % (
                            epoch + 1,
                            num_epochs,
                            i + 1,
                            len(self.train_dataset) // batch_size,
                            np.mean(l),
                        )
                    )

        # update temporal ensemble
        Z = alpha * Z + (1.0 - alpha) * outputs
        z = Z * (1.0 / (1.0 - alpha ** (epoch + 1)))

        return outputs, Z, z, l, supl, unsupl

    def fit(
        self,
        train_loader,
        test_loader,
        k=100,
        alpha=0.6,
        lr=0.002,
        beta2=0.99,
        num_epochs=150,
        batch_size=100,
        drop=0.5,
        std=0.15,
        fm1=16,
        fm2=32,
        divide_by_bs=False,
        w_norm=False,
        data_norm="pixelwise",
        early_stop=None,
        c=300,
        n_classes=10,
        max_epochs=80,
        max_val=30.0,
        ramp_up_mult=-5.0,
        n_samples=60000,
        print_res=True,
        **kwargs
    ):
        # train code goes here
        # train

        model = self.get_model()
        model.train()
        losses = []
        sup_losses = []
        unsup_losses = []
        best_loss = 20.0
        ntrain = len(train_loader.dataset)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, betas=(0.9, 0.99)
        )
        Z = torch.zeros(ntrain, n_classes).float()  # type: ignore # intermediate values
        z = torch.zeros(ntrain, n_classes).float()  # temporal outputs
        outputs = torch.zeros(ntrain, n_classes).float()  # current outputs

        for epoch in range(num_epochs):
            outputs, Z, z, l, supl, unsupl = self._iteration_train(
                model,
                epoch,
                optimizer,
                outputs,
                z
            )
            acc_best = self._iteration_test(model=model)
            # handle metrics, losses, etc.
            eloss = np.mean(l)
            losses.append(eloss)
            sup_losses.append(
                (1.0 / k) * np.sum(supl)
            )  # division by 1/k to obtain the mean supervised loss
            unsup_losses.append(np.mean(unsupl))
            # saving model
            if eloss < best_loss:
                best_loss = eloss
                torch.save({"state_dict": model.state_dict()}, "model_best.pth.tar")
        return losses, sup_losses, unsup_losses,acc_best

    def _iteration_test(self, model):
        with torch.no_grad():
            model.eval()

        acc = calc_metrics(model, self.test_loader)
        if Config.print_res:
            print("Accuracy of the network on the 10000 test images: %.2f %%" % (acc))

        # test best model
        checkpoint = torch.load("model_best.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        acc_best = calc_metrics(model, self.test_loader)
        if Config.print_res:
            print(
                "Accuracy of the network (best model) on the 10000 test images: %.2f %%"
                % (acc_best)
            )

        return acc_best

    def get_data(self):
        # normalize data
        m = (0.1307,)
        st = (0.3081,)
        normalize = tf.Normalize(m, st)
        self.train_dataset = self.torch_dataset(
            root="../data",
            train=True,
            transform=tf.Compose([tf.ToTensor(), normalize]),
            download=True,
        )

        # load test data
        self.test_dataset = self.torch_dataset(
            root="../data",
            train=False,
            transform=tf.Compose([tf.ToTensor(), normalize]),
        )

        # train_dataset = self.dataset(train_data.data, train_data.targets)
        # val_dataset  = self.dataset(test_data.data, train_data.targets)
        return self.train_dataset, self.test_dataset

    def prepare_dataloader(self)->torch.utils.data.DataLoader :
        n = len(self.train_dataset)
        rrng = np.random.RandomState(Config.seeds[0])
        k = Config.k
        cpt = 0
        indices = torch.zeros(k)
        other = torch.zeros(n - k)
        n_classes = len(self.train_dataset.classes)
        card = k // n_classes

        for i in range(n_classes):
            class_items = (self.train_dataset.train_labels == i).nonzero().squeeze()
            n_class = len(class_items)
            rd = np.random.permutation(np.arange(n_class))
            indices[i * card : (i + 1) * card] = class_items[rd[:card]]
            other[cpt : cpt + n_class - card] = class_items[rd[card:]]
            cpt += n_class - card

        other = other.long()
        indices = indices.long()
        self.train_dataset.train_labels[other] = -1

        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=self.config.shuffle_train,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=False,
        )
 
        if self.config.return_idxs:
            return train_loader, test_loader, indices
        else:
            return train_loader, test_loader
        # self.train_loader = torch.utils.data.DataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.config.batch_size,
        #     num_workers=4,
        #     shuffle=False,
        # )
        # self.test_loader = torch.utils.data.DataLoader(
        #     dataset=self.test_dataset,
        #     batch_size=self.config.batch_size,
        #     num_workers=4,
        #     shuffle=False,
        # )

    def get_model(self):
        return self.model

    def get_seed(self):
        return self.seed

    def get_kwargs(self):
        return self.kwargs

    def set_model(self, model):
        self.model = model()

    def set_seed(self, seed):
        self.seed = seed

    def set_kwargs(self, kwargs):
        self.kwargs = kwargs

if __name__ == "__main__":
    # print(Config.batch_size)
    cnn_model = trainer(model=convnet, seed=torch.seed(), dataset=mnist, configs=Config())
