import numpy as np

rng = np.random.RandomState(26)


class Config:
    train_x = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/trainval-image"
    train_y = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/trainval-mask"
    valid_x = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/test-image"
    valid_y = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/test-mask"
    H = 256
    W = 256
    batch_size = 32
    num_epochs = 100
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    n_samples = 100
    n_classes = 2
    shuffle_train = True
    shuffle_valid = True
    print_res = True
    experiment_name = "fully_supervised_thyroid_nodules"
    n_exps = 3
    seeds = rng.randint(10000, size=n_exps)
    # c = int(batch_size / 2)
    RESULTS_DIR = "Fully_Supervised_Results"
    THRESHOLD = 0.5
    mode = "test"
