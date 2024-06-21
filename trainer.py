import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import calc_metrics, prepare_dataset, weight_schedule
from config import Config
from eval import temporal_loss
from utils import GaussianNoise, savetime, save_exp
from model import Unet
from glob import glob
import os

class Trainer:
    def __init__(self, seed) -> None:
        self.seed = seed
        self.device = self._getdevice()
        self.config = Config()
        self.train_dataset, self.test_dataset = prepare_dataset(
            train_x=sorted(glob(os.path.join((self.config.train_x),"*"))),
            train_y=sorted(glob(os.path.join((self.config.train_y),"*"))),
            valid_x=sorted(glob(os.path.join((self.config.valid_x),"*"))),
            valid_y=sorted(glob(os.path.join((self.config.valid_y),"*"))),
            H=self.config.H,
            W=self.config.W,
        )
        self.train_loader, self.test_loader = self._get_dataloaders(
            self.train_dataset, self.test_dataset
        )
        self.model = self._get_model()
        self.writer = self._get_tensorboard()

    def _get_model(self):
        return Unet(img_ch=3, output_ch=1).to(self.device)

    def _get_tensorboard(self):
        return SummaryWriter(log_dir=self.config.experiment_name)

    def _getdevice(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device

    def _get_dataloaders(
        self,
        train_dataset,
        test_dataset,
    ):

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )
        return train_loader, test_loader

    def fit(self):
        ntrain = len(self.train_dataset)
        n_samples = self.config.n_samples
        writer = self.writer
        model = self.model
        # setup param optimization
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.lr, betas=(0.9, 0.99)
        )
        # train
        model.train()
        losses = []
        sup_losses = []
        unsup_losses = []
        best_loss = 20.0
        n_classes = int(self.config.n_classes)

        Z = (
            torch.zeros(ntrain, n_classes).float().to(self.device)
        )  # intermediate values
        z = torch.zeros(ntrain, n_classes).float().to(self.device)  # temporal outputs
        outputs = (
            torch.zeros(ntrain, n_classes).float().to(self.device)
        )  # current outputs

        for epoch in range(self.config.num_epochs):
            t = timer()

            # evaluate unsupervised cost weight
            w = weight_schedule(
                epoch,
                self.config.max_epochs,
                self.config.max_val,
                self.config.ramp_up_mult,
                self.config.k,
                self.config.n_samples,
            )

            if (epoch + 1) % 10 == 0:
                print("unsupervised loss weight : {}".format(w))

            # turn it into a usable pytorch object
            w = torch.autograd.Variable(torch.FloatTensor([w]), requires_grad=False).to(
                self.device
            )

            l = []
            supl = []
            unsupl = []

            for i, (images, labels) in enumerate(self.train_loader):
                images = Variable(images).to(self.device)
                labels = Variable(labels, requires_grad=False).to(self.device)

                # get output and calculate loss
                optimizer.zero_grad()
                out = model(images)
                zcomp = Variable(
                    z[i * self.config.batch_size : (i + 1) * self.config.batch_size],
                    requires_grad=False,
                ).to(self.device)
                loss, suploss, unsuploss, nbsup = temporal_loss(
                    out, zcomp, w, labels, self.device
                )

                # save outputs and losses
                outputs[
                    i * self.config.batch_size : (i + 1) * self.config.batch_size
                ] = out.data.clone()
                l.append(loss.data.item())
                supl.append(nbsup * suploss.data.item())
                unsupl.append(unsuploss.data.item())

                # backprop
                loss.backward()
                optimizer.step()

                i = int(i)
                epoch = int(epoch)
                try:
                    c = int(self.config.c)
                except Exception as e:
                    print(e, self.config.c)
                    c = 300

                # print loss
                if (epoch + 1) % 10 == 0:
                    if i + 1 == 2 * c:
                        print(
                            "Epoch [%d/%d], Step [%d/%d], Loss: %.6f, Time (this epoch): %.2f s"
                            % (
                                epoch + 1,
                                self.config.num_epochs,
                                i + 1,
                                len(self.train_dataset) // self.config.batch_size,
                                np.mean(l),
                                timer() - t,
                            )
                        )
                    elif (i + 1) % c == 0:
                        print(
                            "Epoch [%d/%d], Step [%d/%d], Loss: %.6f"
                            % (
                                epoch + 1,
                                self.config.num_epochs,
                                i + 1,
                                len(self.train_dataset) // self.config.batch_size,
                                np.mean(l),
                            )
                        )

            # update temporal ensemble
            Z = self.config.alpha * Z + (1.0 - self.config.alpha) * outputs
            z = Z * (1.0 / (1.0 - self.config.alpha ** (epoch + 1)))

            # handle metrics, losses, etc.
            eloss = np.mean(l)
            losses.append(eloss)
            sup_losses.append(
                (1.0 / self.config.k) * np.sum(supl)
            )  # division by 1/k to obtain the mean supervised loss
            unsup_losses.append(np.mean(unsupl))

            # saving model
            if eloss < best_loss:
                best_loss = eloss
                torch.save({"state_dict": model.state_dict()}, "model_best.pth.tar")
            # Log the losses in the writer
            # Log the loss and accuracy values at the end of each epoch
            writer.add_scalar("Loss/supervised_training_loss", sup_losses[-1], epoch)
            writer.add_scalar(
                "Loss/unsupervised_training_loss", unsup_losses[-1], epoch
            )
            writer.add_scalar("Loss/training_loss", losses[-1], epoch)

        # test
        model.eval()
        acc = calc_metrics(model, self.test_loader, device=self.device)
        if self.config.print_res:
            print("Accuracy of the networ on the 10000 test images: %.2f %%" % (acc))
        # test best model
        checkpoint = torch.load("model_best.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        acc_best = calc_metrics(model, self.test_loader, device=self.device)
        if self.config.print_res:
            print(
                "Accuracy of the network (best model) on the 10000 test images: %.2f %%"
                % (acc_best)
            )
        return acc, acc_best, losses, sup_losses, unsup_losses, self.indices
