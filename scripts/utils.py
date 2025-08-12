import numpy as np
import os
import torch  # type: ignore
import cv2  # type: ignore
import torch.nn as nn  # type: ignore
import torchvision.transforms as tf  # type: ignore
import matplotlib.gridspec as gsp  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib  # type: ignore

matplotlib.use("Agg")


from torch.autograd import Variable  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from datetime import datetime
from matplotlib.lines import Line2D  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

from config import Config
from scripts.dataset import ThyroidNodules

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


def prepare_dataset(
    train_x: list,
    train_y: list,
    valid_x: list,
    valid_y: list,
    H: int,
    W: int,
    mean: list[float],
    std: list[float],
):
    """Create datasets and dataloaders for this fold

    Args:
        train_x (list): _description_
        train_y (list): _description_
        valid_x (list): _description_
        valid_y (list): _description_
        H (int): _description_
        W (int): _description_

    Returns:
        torch.utils.data.Dataset
    """
    # Create datasets and dataloaders for this fold
    train_dataset = ThyroidNodules(
        train_x,
        train_y,
        image_size=(H, W),
        transform=tf.Compose(
            [
                tf.Normalize(mean=mean, std=std),
            ]
        ),
    )

    test_dataset = ThyroidNodules(
        valid_x,
        valid_y,
        image_size=(H, W),
        transform=tf.Compose(
            [
                # tf.ToTensor(),
                tf.RandomRotation(90),
                tf.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=10
                ),
                tf.Normalize(mean=mean, std=std),
            ]
        ),
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


# def save_exp(time, losses, sup_losses, unsup_losses, accs, accs_best, idxs, config):

#     def save_txt(fname, accs, config):
#         with open(fname, "w") as fp:
#             fp.write("GLOB VARS\n")
#             fp.write("n_exp        = {}\n".format(config.n_exp))
#             fp.write("k            = {}\n".format(config.k))
#             fp.write("MODEL VARS\n")
#             fp.write("drop         = {}\n".format(config.drop))
#             fp.write("std          = {}\n".format(config.std))
#             fp.write("w_norm       = {}\n".format(config.w_norm))
#             fp.write("OPTIM VARS\n")
#             fp.write("lr           = {}\n".format(config.lr))
#             fp.write("batch_size   = {}\n".format(config.batch_size))
#             fp.write("TEMP ENSEMBLING VARS\n")
#             fp.write("alpha        = {}\n".format(config.alpha))
#             fp.write("data_norm    = {}\n".format(config.data_norm))
#             fp.write("divide_by_bs = {}\n".format(config.divide_by_bs))
#             fp.write("\nRESULTS\n")
#             fp.write("best accuracy : {}\n".format(np.max(accs)))
#             fp.write("accuracy : {} (+/- {})\n".format(np.mean(accs), np.std(accs)))
#             fp.write("accs : {}\n".format(accs))

#     labels = ["seed_" + str(sd) for sd in config.seeds]
#     if not os.path.isdir("exps"):
#         os.mkdir("exps")
#     time_dir = os.path.join("exps", time)
#     if not os.path.isdir(time_dir):
#         os.mkdir(time_dir)
#     fname_bst = os.path.join("exps", time, "training_best.png")
#     fname_fig = os.path.join("exps", time, "training_all.png")
#     fname_smr = os.path.join("exps", time, "summary.txt")
#     fname_sd = os.path.join("exps", time, "seed_samples")
#     best = np.argmax(accs_best)
#     save_losses([losses[best]], [sup_losses[best]], [unsup_losses[best]], fname_bst)
#     save_losses(losses, sup_losses, unsup_losses, fname_fig, labels=labels)
#     for seed, indices in zip(config.seeds, idxs):
#         save_seed_samples(fname_sd + "_seed" + str(seed) + ".png", indices)
#     save_txt(fname_smr, accs_best, config=config)


# def save_seed_samples(fname, indices):
#     train_dataset, test_dataset = prepare_mnist(path=config.dataset_path)
#     imgs = train_dataset.train_data[indices.numpy().astype(int)]

#     plt.style.use("classic")
#     fig = plt.figure(figsize=(15, 60))
#     gs = gsp.GridSpec(20, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.0, hspace=0.0)
#     for ll in range(100):
#         i = ll // 5
#         j = ll % 5
#         img = imgs[ll].numpy()
#         ax = plt.subplot(gs[i, j])
#         ax.tick_params(
#             axis="both",
#             which="both",
#             bottom="off",
#             top="off",
#             labelbottom="off",
#             left="off",
#             right="off",
#             labelleft="off",
#         )
#         ax.imshow(img)

#     plt.savefig(fname)


def plot_sample(x, y, out, cnt, RESULT_DIR, THRESHOLD):
    x = x.detach().cpu().numpy().transpose(1, 2, 0)
    y = y.detach().cpu().numpy().transpose(1, 2, 0)
    out = out.detach().cpu().numpy().transpose(1, 2, 0)

    # Convert to grayscale
    # x_gray = np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])

    y = y > THRESHOLD
    out = out > THRESHOLD
    y = y.astype(np.uint8)
    out = out.astype(np.uint8)

    # Find contours
    y_contours, _ = cv2.findContours(
        y[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    out_contours, _ = cv2.findContours(
        out[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the grayscale image
    ax.imshow(x, cmap="gray")

    # Plot ground truth contours in blue
    for contour in y_contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="blue", linewidth=2)

    # Plot predicted contours in red
    for contour in out_contours:
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="red", linewidth=2)

    # Create custom legend handles
    custom_lines = [
        Line2D([0], [0], color="blue", lw=2, label="Ground Truth"),
        Line2D([0], [0], color="red", lw=2, label="Prediction"),
    ]

    # Add legend to the plot
    ax.legend(handles=custom_lines, loc="upper right")
    ax.axis("off")
    ax.set_title("Visualization of Segmented Thyroid Nodules")

    # Save the figure
    fig.savefig(os.path.join(RESULT_DIR, f"results_sample_{cnt}.png"))
    plt.show()

    return x, y, out


def calculate_metrics(y_true, y_pred, threshold):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > threshold
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    # y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred > threshold
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return score_jaccard, score_f1, score_recall, score_precision, score_acc


def _get_device():
    """Returns the device to be used for training and inference.

    Returns:
        torch.device : Device to be used for training and inference
    """
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
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    BATCH_SIZE: int,
    NUM_WORKERS: int,
    SHUFFLE_TRAIN: bool = True,
    SHUFFLE_TEST: bool = False,
):
    """Returns the dataloaders for training and testing datasets.

    Args:
        train_dataset (torch.Datsets): Training dataset
        test_dataset (torch.Datset): Testing dataset
        BATCH_SIZE (int): Batch size
        NUM_WORKERS (int): Number of workers to be used for data loading
        SHUFFLE_TRAIN (bool, optional): Shuffle the samples while loading data . Defaults to True.
        SHUFFLE_TEST (bool, optional): Shuffle the samples while loading data. Defaults to False.

    Returns:
        _type_: _description_
    """

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TEST,
        num_workers=NUM_WORKERS,
    )
    return train_loader, test_loader


def get_normalized_mean(paths):
    list_m1 = []
    list_m2 = []
    list_m3 = []
    for path_i in paths:
        img = cv2.imread(path_i, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        # print("img shape", img.shape)
        # break
        m1, m2, m3 = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
        list_m1.append(m1)
        list_m2.append(m2)
        list_m3.append(m3)
    return [np.mean(list_m1), np.mean(list_m2), np.mean(list_m3)], [
        np.std(list_m1),
        np.std(list_m2),
        np.std(list_m3),
    ]


def get_labelled_indices(train_x, RATIO_LABELLED_SAMPLES, debug=False, seed=None):
    np.random.seed(seed)
    num_of_labelled_samples = int(len(train_x) / RATIO_LABELLED_SAMPLES)
    if debug:
        print("Number of labelled samples : %d" % num_of_labelled_samples)
    indices = np.random.choice(len(train_x), num_of_labelled_samples, replace=False)
    return indices


def sabotage_samples(labelled_idxs, train_dataset):
    # array = np.full((256, 256), -1)
    # unlabelled_idxs = set(labelled_indices) - set(train_dataset
    # # template_array = template_array.fill(0.5)
    # for i in range(len(train_dataset)):
    #     train_dataset.masks[i] = array
    array = np.full((256, 256), -1)
    unlabelled_idxs = list(set(np.arange(len(train_dataset))) - set(labelled_idxs))
    # template_array = template_array.fill(0.5)
    for i in unlabelled_idxs:
        train_dataset.masks[i] = array
    return unlabelled_idxs, train_dataset
