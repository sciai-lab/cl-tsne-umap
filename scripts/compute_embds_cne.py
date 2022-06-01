import numpy as np
import os
from vis_utils.loaders import load_dataset
from utils import get_noise_in_estimator, get_path
import pickle
import cne

dataset = "mnist"

# parameters
a = 1.0
b = 1.0
seed = 2
parametric = False
log_embds = True
log_norms = True
log_kl = True
optimizer = "sgd" #"sgd" or "adam"
n_epochs = 500
loss_mode = "neg_sample" #["umap", "nce", "neg_sample", "infonce", "infonce_alt"]
n_noise = 5
batch_size = 1024
rescale = 1.0 # how to rescale the initialization
anneal_lr = True
momentum=0.0
lr_min_factor=0.0
clamp_low = 1e-10
on_gpu = True
noise_in_estimator = 1. # reparametrization of Z_bar, default of 1. corresponds to normal negative sampling
init_type = "EE"  # "pca", "random" or "EE" for early exaggeration


root_path = get_path("data")

# get data
k = 15
x, y, sknn_graph, pca2 = load_dataset(root_path, dataset, k=k)


# get init
# default init (also for EE) is PCA
init = pca2
if rescale:
    init = pca2 / np.std(pca2[:, 0]) * rescale

if init_type == "random":
    np.random.seed(seed)
    init = np.random.randn(len(x), 2)
    if rescale:
        init = init / np.std(init) * rescale
    init_str = f"random_rescale_{rescale}"

elif init_type == "pca":
    # only collect correct string for filename
    init_str = f"pca_rescale_{rescale}"

elif init_type == "EE":
    # compute or load embedder of the early exaggeration phase
    file_name_init = os.path.join(root_path,
                                 dataset,
                                 f"cne_{loss_mode}_n_noise_{5}_n_epochs_{250}_init_pca_rescale_{rescale}_bs_{batch_size}"
                                 f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                 )

    try:
        with open(file_name_init, "rb") as file:
            embedder_init_cne = pickle.load(file)
    except FileNotFoundError:
        print(f"Starting with {file_name_init}")
        logger = cne.callbacks.Logger(log_embds=log_embds,
                                      log_norms=log_norms,
                                      log_kl=log_kl,
                                      graph=sknn_graph,
                                      n=len(x) if parametric else None)
        embedder_init = cne.CNE(loss_mode=loss_mode,
                                parametric=parametric,
                                negative_samples=5,
                                n_epochs=250,
                                batch_size=batch_size,
                                on_gpu=on_gpu,
                                print_freq_epoch=100,
                                print_freq_in_epoch=None,
                                callback=logger,
                                optimizer=optimizer,
                                momentum=momentum,
                                save_freq=1,
                                anneal_lr=anneal_lr,
                                lr_min_factor=lr_min_factor,
                                clamp_low=clamp_low,
                                seed=seed,
                                loss_aggregation="sum",
                                force_resample=True
                                )
        embedder_init.fit(x, init=init, graph=sknn_graph)
        embedder_init_cne = embedder_init.cne

        with open(file_name_init, "wb") as file:
            pickle.dump(embedder_init_cne, file, pickle.HIGHEST_PROTOCOL)

    init = embedder_init_cne.callback.embds[-1]
    init_str = "EE"

# main optimization phase
nbs_noise_in_estimator = get_noise_in_estimator(len(x), 5, dataset)

for noise_in_estimator in nbs_noise_in_estimator:
    if loss_mode != "neg_sample":
        file_name = os.path.join(root_path,
                                 dataset,
                                 f"cne_{loss_mode}_n_noise_{n_noise}_n_epochs_{n_epochs}_init_{init_str}_bs_{batch_size}"
                                 f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                 )
    else:
        file_name = os.path.join(root_path,
                                 dataset,
                                 f"cne_{loss_mode}_n_noise_{n_noise}_noise_in_estimator_{noise_in_estimator}_n_epochs_{n_epochs}_init_{init_str}_bs_{batch_size}"
                                 f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                 )
    print(f"Starting with {file_name}")
    try:
        with open(file_name, "rb") as file:
            embedder_cne = pickle.load(file)
    except FileNotFoundError:
        logger = cne.callbacks.Logger(log_embds=log_embds,
                                      log_norms=log_norms,
                                      log_kl=log_kl,
                                      graph=sknn_graph,
                                      n=len(x) if parametric else None)
        embedder = cne.CNE(loss_mode=loss_mode,
                           parametric=parametric,
                           negative_samples=n_noise,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           on_gpu=on_gpu,
                           print_freq_epoch=100,
                           print_freq_in_epoch=None,
                           callback=logger,
                           optimizer=optimizer,
                           momentum=momentum,
                           save_freq=1,
                           anneal_lr=anneal_lr,
                           noise_in_estimator=noise_in_estimator,
                           lr_min_factor=lr_min_factor,
                           clamp_low=clamp_low,
                           seed=seed,
                           loss_aggregation="sum",
                           force_resample=True
                           )
        embedder.fit(x, init=init, graph=sknn_graph)
        embedder_cne = embedder.cne

        with open(file_name, "wb") as file:
            pickle.dump(embedder_cne, file, pickle.HIGHEST_PROTOCOL)

    print(f"done with noise_in_estimator={noise_in_estimator}")
    print(f"Time in min: {embedder_cne.time / 60}")

print("done with mnist negtsne spectrum")


