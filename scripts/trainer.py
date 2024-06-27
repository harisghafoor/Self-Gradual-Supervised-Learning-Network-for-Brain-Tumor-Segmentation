import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

from scripts.utils import prepare_dataset, calculate_metrics
from scripts.test import evaluate_test_data
from scripts.loss import compute_loss, DiceLoss
from scripts.model import Unet
from config import Config

from glob import glob


class Trainer:
    """  Pytorch Abstraction Trainer class for training the model
    """
    def __init__(self, seed, device, model, config_file) -> None:
        self.seed = seed
        self.device = device
        self.config = config_file
        self.model = model
        self.writer = self._get_tensorboard()
        self.model_save_path = self.config.model_save_path
        os.makedirs(f"models/{self.config.experiment_name}", exist_ok=True)

    def _get_tensorboard(self):
        return SummaryWriter(log_dir=self.config.experiment_name)

    def fit(self, train_loader, test_dataset) -> pd.DataFrame:
        """ Fits the Training Data to the model to perform training

        Args:
            train_loader (torch.utils.data.Dataloader)
            test_dataset (torch.utils.data.Dataset)

        Returns:
            _type_: _description_
        """
        writer = self.writer
        model = self.model
        # setup param optimization
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.LR,
        )
        # train
        model.train()
        losses = []
        best_loss = 20.0
        for epoch in range(self.config.NUM_EPOCHS):
            t = timer()
            batch_losses = []
            for i, (images, labels) in enumerate(train_loader):
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
                torch.save({"state_dict": model.state_dict()}, self.model_save_path)

            # Log the losses in the writer
            # Log the loss and accuracy values at the end of each epoch
            writer.add_scalar("Loss/training_loss", losses[-1], epoch)

            if (epoch + 1) % 10 == 0:
                print(
                    "Epoch [%d/%d], Training Loss: %.6f, Time (this epoch): %.2f s"
                    % (
                        epoch + 1,
                        self.config.NUM_EPOCHS,
                        np.mean(batch_losses),
                        timer() - t,
                    )
                )
            elif (epoch + 1) % self.config.SHOW_PROGRESS_AFTER_EPOCH == 0:
                model.eval()
                df = self.predict(model,test_dataset=test_dataset)
                iou = df["jaccard"].mean().item()
                f1 = df["f1"].mean().item()
                recall = df["recall"].mean().item()
                precision = df["precision"].mean().item()
                accuracy = df["accuracy"].mean().item()
                print(
                    "Epoch [%d/%d], IOU: %.6f, F1: %.6f, Recall: %.6f, Precision: %.6f, Accuracy: %.6f",
                    epoch + 1,
                    self.config.NUM_EPOCHS,
                    iou,
                    f1,
                    recall,
                    precision,
                    accuracy,
                )

                writer.add_scalar("Loss/validation_dice_score", f1, epoch)
                writer.add_scalar("Loss/validation_iou_score", iou, epoch)
                writer.add_scalar("Loss/validation_accuracy_score", accuracy, epoch)
        writer.close()
        return model, losses

    def predict(self, model,test_dataset):
        df = evaluate_test_data(
            model=model,
            torch_dataset=test_dataset,
            torch_device=self.device,
            RESULT_DIR=self.config.RESULTS_DIR,
            THRESHOLD=self.config.THRESHOLD,
        )
        return df


if __name__ == "__main__":
    trainer = Trainer(seed=7)
    input = next(iter(trainer.train_loader))[0].to(trainer.device)
    output = trainer.model(input)
    target = next(iter(trainer.train_loader))[1].to(trainer.device)
    loss = compute_loss(output, target)
    print("output", output.shape)
    print("target", target.shape)
    print("loss", loss)
