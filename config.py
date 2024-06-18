import numpy as np

rng = np.random.RandomState(42)


class Config:
    # global vars
    n_exp = 3
    k = 250
    # model vars
    drop = 0.5
    std = 0.15
    fm1 = 32
    fm2 = 64
    w_norm = True
    # optim vars
    lr = 0.002 # sqrt of the scacle with which I increased the batch size i.e 6
    beta2 = 0.99
    num_epochs = 15
    batch_size = 500
    # temporal ensembling vars
    alpha = 0.6
    data_norm = "channelwise"
    divide_by_bs = False
    seeds = [rng.randint(100) for _ in range(n_exp)]
    print_res = True
    max_epochs = 50
    max_val = 20
    early_stop = None
    c = 300
    n_classes = 10
    ramp_up_mult = -5.0
    print_res = True
    shuffle_train = False
    return_idxs = True
    # Directory for tensorboard logs
    folder_name = "semi supervised learning from scrath on mnist"
    experiment_name = "new_ssl"
    dataset_path = "../data"
    n_samples = 60000
