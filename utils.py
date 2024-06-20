from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gsp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as tf
from config import Config

# Global Variables
config = Config()


class GaussianNoise(nn.Module):

    def __init__(self, batch_size, input_shape=(1, 28, 28), std=0.05, device=None):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape)).to(device)
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise


def prepare_mnist(path):
    # normalize data
    m = (0.1307,)
    st = (0.3081,)
    normalize = tf.Normalize(m, st)

    # load train data
    train_dataset = datasets.MNIST(
        root=path,
        train=True,
        transform=tf.Compose([tf.ToTensor(), normalize]),
        download=True,
    )

    # load test data
    test_dataset = datasets.MNIST(
        root=path, train=False, transform=tf.Compose([tf.ToTensor(), normalize])
    )

    return train_dataset, test_dataset


def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.0
    elif epoch >= max_epochs:
        return max_val
    # print("mult value",mult)
    # printO("mult type",type)
    return max_val * np.exp(-5.0 * (1.0 - int(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epochs, max_val, mult)


def calc_metrics(model, loader, device=torch.device("cpu")):
    correct = 0
    total = 0
    for i, (samples, labels) in enumerate(loader):
        samples = Variable(samples, volatile=True).to(device)
        labels = Variable(labels).to(device)
        outputs = model(samples)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data.view_as(predicted)).sum()

    acc = 100 * float(correct) / total
    return acc


def savetime():
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")


def save_losses(losses, sup_losses, unsup_losses, fname, labels=None):
    plt.style.use("ggplot")

    # color palette from Randy Olson
    colors = [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
    ]

    colors = [(float(c[0]) / 255, float(c[1]) / 255, float(c[2]) / 255) for c in colors]

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    for i in range(3):
        axs[i].tick_params(
            axis="both",
            which="both",
            bottom="off",
            top="off",
            labelbottom="on",
            left="off",
            right="off",
            labelleft="on",
        )
    for i in range(len(losses)):
        axs[0].plot(losses[i], color=colors[i])
        axs[1].plot(sup_losses[i], color=colors[i])
        axs[2].plot(unsup_losses[i], color=colors[i])
    axs[0].set_title("Overall loss", fontsize=14)
    axs[1].set_title("Supervised loss", fontsize=14)
    axs[2].set_title("Unsupervised loss", fontsize=14)
    if labels is not None:
        axs[0].legend(labels)
        axs[1].legend(labels)
        axs[2].legend(labels)
    plt.savefig(fname)


def save_exp(time, losses, sup_losses, unsup_losses, accs, accs_best, idxs, config):

    def save_txt(fname, accs, config):
        with open(fname, "w") as fp:
            fp.write("GLOB VARS\n")
            fp.write("n_exp        = {}\n".format(config.n_exp))
            fp.write("k            = {}\n".format(config.k))
            fp.write("MODEL VARS\n")
            fp.write("drop         = {}\n".format(config.drop))
            fp.write("std          = {}\n".format(config.std))
            fp.write("w_norm       = {}\n".format(config.w_norm))
            fp.write("OPTIM VARS\n")
            fp.write("lr           = {}\n".format(config.lr))
            fp.write("batch_size   = {}\n".format(config.batch_size))
            fp.write("TEMP ENSEMBLING VARS\n")
            fp.write("alpha        = {}\n".format(config.alpha))
            fp.write("data_norm    = {}\n".format(config.data_norm))
            fp.write("divide_by_bs = {}\n".format(config.divide_by_bs))
            fp.write("\nRESULTS\n")
            fp.write("best accuracy : {}\n".format(np.max(accs)))
            fp.write("accuracy : {} (+/- {})\n".format(np.mean(accs), np.std(accs)))
            fp.write("accs : {}\n".format(accs))

    labels = ["seed_" + str(sd) for sd in config.seeds]
    if not os.path.isdir("exps"):
        os.mkdir("exps")
    time_dir = os.path.join("exps", time)
    if not os.path.isdir(time_dir):
        os.mkdir(time_dir)
    fname_bst = os.path.join("exps", time, "training_best.png")
    fname_fig = os.path.join("exps", time, "training_all.png")
    fname_smr = os.path.join("exps", time, "summary.txt")
    fname_sd = os.path.join("exps", time, "seed_samples")
    best = np.argmax(accs_best)
    save_losses([losses[best]], [sup_losses[best]], [unsup_losses[best]], fname_bst)
    save_losses(losses, sup_losses, unsup_losses, fname_fig, labels=labels)
    for seed, indices in zip(config.seeds, idxs):
        save_seed_samples(fname_sd + "_seed" + str(seed) + ".png", indices)
    save_txt(fname_smr, accs_best, config=config)


def save_seed_samples(fname, indices):
    train_dataset, test_dataset = prepare_mnist(path=config.dataset_path)
    imgs = train_dataset.train_data[indices.numpy().astype(int)]

    plt.style.use("classic")
    fig = plt.figure(figsize=(15, 60))
    gs = gsp.GridSpec(20, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.0, hspace=0.0)
    for ll in range(100):
        i = ll // 5
        j = ll % 5
        img = imgs[ll].numpy()
        ax = plt.subplot(gs[i, j])
        ax.tick_params(
            axis="both",
            which="both",
            bottom="off",
            top="off",
            labelbottom="off",
            left="off",
            right="off",
            labelleft="off",
        )
        ax.imshow(img)

    plt.savefig(fname)
