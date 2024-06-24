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
from eval import compute_loss, DiceLoss, calculate_metrics
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
            train_x=sorted(glob(os.path.join((self.config.train_x), "*"))),
            train_y=sorted(glob(os.path.join((self.config.train_y), "*"))),
            valid_x=sorted(glob(os.path.join((self.config.valid_x), "*"))),
            valid_y=sorted(glob(os.path.join((self.config.valid_y), "*"))),
            H=self.config.H,
            W=self.config.W,
        )
        self.train_loader, self.test_loader = self._get_dataloaders(
            self.train_dataset, self.test_dataset
        )
        self.model = self._get_model()
        self.writer = self._get_tensorboard()
        self.dice_loss = DiceLoss()

    def _get_model(self):
        return Unet(img_ch=3, output_ch=1).to(self.device)

    def _get_tensorboard(self):
        return SummaryWriter(log_dir=self.config.experiment_name)

    def _getdevice(self):
        if torch.cuda.is_available():
            print("Using GPU")
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Apple MPS")
            device = torch.device("mps")
        else:
            print("Using CPU")
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
            num_workers=0,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return train_loader, test_loader

    def fit(self):
        writer = self.writer
        model = self.model
        # setup param optimization
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
        )
        # train
        model.train()
        losses = []
        best_loss = 20.0
        for epoch in range(self.config.num_epochs):
            t = timer()
            batch_losses = []
            for i, (images, labels) in enumerate(self.train_loader):
                images = Variable(images).to(self.device)
                labels = Variable(labels, requires_grad=False).to(self.device)

                # get output and calculate loss
                optimizer.zero_grad()
                out = model(images)
                # compute loss
                loss = compute_loss(output=out, target=labels)
                # backprop
                loss.backward()
                # update weights
                optimizer.step()
                # print loss
                i = int(i)
                epoch = int(epoch)
                batch_losses.append(loss.item())

            # print loss
            losses.append(np.mean(batch_losses))
            # saving model
            if np.mean(batch_losses) < best_loss:
                best_loss = np.mean(batch_losses)
                torch.save({"state_dict": model.state_dict()}, "model_best.pth.tar")

            # Log the losses in the writer
            # Log the loss and accuracy values at the end of each epoch
            writer.add_scalar("Loss/training_loss", losses[-1], epoch)

            if (epoch + 1) % 10 == 0:
                print(
                    "Epoch [%d/%d], Training Loss: %.6f, Time (this epoch): %.2f s"
                    % (
                        epoch + 1,
                        self.config.num_epochs,
                        np.mean(batch_losses),
                        timer() - t,
                    )
                )
            elif (epoch + 1) % 25 == 0:
                model.eval()
                batch_dice_score_valid = []
                for i, (images, labels) in enumerate(self.test_loader):
                    images = Variable(images).to(self.device)
                    labels = Variable(labels, requires_grad=False).to(self.device)
                    out = model(images)
                    out = F.sigmoid(out)
                    dice_score = 1 - self.dice_loss(inputs=out, targets=labels).item()
                    batch_dice_score_valid.append(dice_score)
                if self.config.print_res:
                    print(
                        "Dice Coefficient of the networ on the 10000 test images: %.2f %%"
                        % (np.mean(batch_dice_score_valid))
                    )
                writer.add_scalar(
                    "Loss/validation_dice_score", np.mean(batch_dice_score_valid), epoch
                )

        return model, losses

    def predict(self, model):
        model.eval()
        results = {"id": [], "metrics": []}
        for i, (images, labels) in enumerate(self.test_loader):
            images = Variable(images).to(self.device)
            labels = Variable(labels, requires_grad=False).to(self.device)
            out = model(images)
            out = F.sigmoid(out)
            results["id"].append(i)
            results["metrics"].append(
                calculate_metrics(y_true=labels, y_pred=out, threshold=0.5)
            )
        return results


if __name__ == "__main__":
    trainer = Trainer(seed=7)
    input = next(iter(trainer.train_loader))[0].to(trainer.device)
    output = trainer.model(input)
    target = next(iter(trainer.train_loader))[1].to(trainer.device)
    loss = compute_loss(output, target)
    print("output", output.shape)
    print("target", target.shape)
    print("loss", loss)
