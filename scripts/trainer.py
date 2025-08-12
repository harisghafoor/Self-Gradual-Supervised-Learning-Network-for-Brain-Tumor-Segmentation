import os
import numpy as np
import pandas as pd
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from torch.autograd import Variable  # type: ignore
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from scripts.utils import weight_schedule
from scripts.test import evaluate_test_data
from scripts.loss import compute_loss, pi_model_loss


class Trainer:
    """Pytorch Abstraction Trainer class for training the model"""

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
        """Fits the Training Data to the model to perform training

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
            model.parameters(), lr=self.config.LR, betas=(0.9, 0.99)
        )
        ntrain = len(train_loader.dataset)
        # WARNING: UPDATE THESE PARAMETERS
        n_samples = self.config.n_samples
        # setup param optimization
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.LR,
        )
        # train
        model.train()
        losses = []
        sup_losses = []
        unsup_losses = []
        best_loss = 20.0
        Z = (
            torch.zeros(ntrain, 1, self.config.H, self.config.W).float().to(self.device)
        )  # intermediate values
        z = (
            torch.zeros(ntrain, 1, self.config.H, self.config.W).float().to(self.device)
        )  # intermediate values
        outputs = (
            torch.zeros(ntrain, 1, self.config.H, self.config.W).float().to(self.device)
        )  # intermediate values

        for epoch in range(self.config.NUM_EPOCHS):
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
            # iterate over all batches
            # turn it into a usable pytorch object
            w = torch.autograd.Variable(torch.FloatTensor([w]), requires_grad=False).to(
                self.device
            )
            l = []
            supl = []
            unsupl = []
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images).to(self.device)
                labels = Variable(labels, requires_grad=False).to(self.device)
                # get output and calculate loss
                optimizer.zero_grad()
                out = model(images)
                out = F.sigmoid(out)
                # Store the outputs in the tensor
                # (batch_size,1,H,W)
                z_comp = Variable(
                    z[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE],
                    requires_grad=False,
                ).to(self.device)
                # compute temporal loss
                loss, sup_loss, unsup_loss, nbsup = pi_model_loss(
                    actual_mask=labels,
                    pred_mask=out,
                    ensemble_mask=z_comp,
                    weight=w,
                    device=self.device,
                )
                # save the output and losses
                outputs[
                    i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE
                ] = out.data.clone()
                # backprop
                loss.backward()
                # update weights
                optimizer.step()
                # print loss
                i = int(i)
                epoch = int(epoch)
                l.append(loss.item())
                supl.append(sup_loss.item())
                unsupl.append(unsup_loss.item())
            # All batches have been trained
            # Update the mean ensemble of outputs
            Z = (self.config.alpha * Z) + (
                1.0 - self.config.alpha
            ) * outputs  # outputs + Smoothed version of Outputs
            z = Z * (1.0 / (1.0 - self.config.alpha ** (epoch + 1)))
            # print loss
            # losses.append(np.mean(batch_losses))
            # saving model
            # handle metrics, losses, etc.
            eloss = np.mean(l)
            losses.append(eloss)
            sup_losses.append(
                (1.0 / self.config.k) * np.sum(supl)
            )  # division by 1/k to obtain the mean supervised loss
            unsup_losses.append(np.mean(unsupl))

            # Log the losses in the writer
            # Log the loss and accuracy values at the end of each epoch
            writer.add_scalar("Loss/supervised_training_loss", sup_losses[-1], epoch)
            writer.add_scalar(
                "Loss/unsupervised_training_loss", unsup_losses[-1], epoch
            )
            writer.add_scalar("Loss/training_loss", losses[-1], epoch)
            # saving model
            if eloss < best_loss:
                best_loss = np.mean(l)
                torch.save({"state_dict": model.state_dict()}, self.model_save_path)

            if (epoch + 1) % 10 == 0:
                # if True:
                print(
                    "Epoch [%d/%d], Training Loss: %.6f, Time (this epoch): %.2f s"
                    % (
                        epoch + 1,
                        self.config.NUM_EPOCHS,
                        eloss,
                        timer() - t,
                    )
                )
            elif (epoch + 1) % self.config.SHOW_PROGRESS_AFTER_EPOCH == 0:
                model.eval()
                df = self.predict(model, test_dataset=test_dataset)
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

    def predict(self, model, test_dataset):
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
