import matplotlib
import numpy as np
import os
from vis_utils.loaders import load_dataset
from vis_utils.utils import  load_dict
from vis_utils.plot import save_animation
import pickle

# parameters for all methods
a = 1.0
b = 1.0
seed = 0
n_noise = 5
alpha_Q = 0.001
alpha = 1.0
n_epochs = 750
k = 15
cmap = matplotlib.cm.get_cmap("tab10")


# paths
root_path = "/export/ial-nfs/user/sdamrich/nce_data"
fig_path = "/export/ial-nfs/user/sdamrich/nce_data/figures"

dataset = "mnist"
# load data
x, y, sknn_graph, pca2 = load_dataset(root_path, dataset, k)


# method specifics

parametric = True
optimizer = "adam"
n_epochs = 750
loss_mode = "umap"
n_noise = 5
batch_size = 1024
rescale = 1.0
anneal_lr = False
momentum=0.9
learning_rate=0.001
lr_min_factor=0.0
clamp_low = 1e-10


file_name = os.path.join(root_path,
                         dataset,
                         f"cne_{loss_mode}_n_noise_{n_noise}_n_epochs_{n_epochs}_init_pca_rescale_{rescale}_bs_{batch_size}"
                         f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}.pkl"
                         )
with open(file_name, "rb") as file:
    embedder = pickle.load(file)

video_file_name = os.path.join(fig_path, file_name[:-4]+".mp4")

save_animation(embedder.callback.embds,
               labels=y,
               filename=video_file_name,
               cmap="tab10")

#print(f"Done with {lr}")

