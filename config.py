import numpy as np

rng = np.random.RandomState(26)


class Config:
    train_x = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/trainval-image"
    train_y = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/trainval-mask"
    valid_x = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/test-image"
    valid_y = "/Users/eloise-em/Documents/Haris Ghafoor Archive/Research and Development/RnD/Thyroid Dataset/tn3k/test-mask"
    H = 256
    W = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    n_samples = 100
    n_classes = 2
    SHUFFLE_TRAIN = True
    SHUFFLE_TEST = False
    print_res = True
    experiment_name = "fully_supervised_thyroid_nodules"
    n_exps = 3
    seeds = rng.randint(10000, size=n_exps)
    # c = int(batch_size / 2)
    RESULTS_DIR = "Results"
    THRESHOLD = 0.5
    mode = "test"
    model_save_path = f"models/{experiment_name}/model_best.pth.tar"
    NUM_WORKERS = 0
    SHOW_PROGRESS_AFTER_EPOCH = 50
    RATIO_LABELLED_SAMPLES = 10 # 10% labelled samples
    # Need to replace with actual parameters for the dataset
    max_epochs = 10
    max_val = 10 
    ramp_up_mult = -5.0
    k = 1000
    # n_samples 