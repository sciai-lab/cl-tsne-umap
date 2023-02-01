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
on_gpu = True
log_embds = True
log_norms = False
log_kl = False
optimizer = "sgd"  # "sgd" or "adam"
n_epochs = 750
loss_mode = "neg_sample"  # ["umap", "nce", "neg_sample", "infonce", "infonce_alt"]
n_noise = 5
rescale = 1.0  # how to rescale the initialization
anneal_lr = True
momentum = 0.0
lr_min_factor = 0.0
clamp_low = 1e-10
init_type = "pca"


batch_sizes = [2 ** power for power in range(7, 21)]
batch_sizes.append(1500006)
batch_sizes = np.array(batch_sizes)[::-1]

devices = ["cuda", "cpu"]
parametric_modes = [True, False]


root_path = get_path("data")

# get data
k = 15
x, y, sknn_graph, pca2 = load_dataset(root_path, dataset, k=k)

init = pca2
if rescale:
    init = pca2 / np.std(pca2[:, 0]) * rescale

if  init_type == "pca":
    init_str = f"pca_rescale_{rescale}"


# main optimization phase
nbs_noise_in_estimator = get_noise_in_estimator(len(x), 5, dataset)

seeds = [0, 1, 2]
noise_in_estimator = nbs_noise_in_estimator[0]  # Just pick the value of t-SNE
for device in devices:
    for parametric in parametric_modes:
        for seed in seeds:
            for batch_size in batch_sizes:

                device_str = ""
                on_gpu = True
                if device == "cpu":
                    device_str = "_device_cpu"
                    on_gpu = False  # Dataset should not be on GPU, when computation is on CPU

                file_name = os.path.join(root_path,
                                         dataset,
                                         f"cne_{loss_mode}_n_noise_{n_noise}_noise_in_estimator_{noise_in_estimator}_n_epochs_{n_epochs}_init_{init_str}_bs_{batch_size}"
                                         f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}{device_str}.pkl"
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
                                       device=device,
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

                print(f"done with seed={seed}")
                try:
                    print(f"Time in min: {embedder_cne.time / 60}")
                except AttributeError:
                    pass
                print(f"Done with batch size {batch_size}, seed {seed}, parametric {parametric} and device {device} ")


