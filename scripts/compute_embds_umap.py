import numpy as np
import os
from vis_utils.loaders import load_dataset
import pickle
import umap

dataset = "mnist"

# parameters
a = 1.0
b = 1.0
seed = 0
k = 15
n_epochs = 750
anneal_lr = True
rescale = 1.0
n_noise = 5

root_path = "/export/ial-nfs/user/sdamrich/nce_data"
fig_path = "/export/ial-nfs/user/sdamrich/nce_data/figures"



# get data
x, y, sknn_graph, pca2 = load_dataset(root_path, dataset, k)



## vary the epsilon to ablate the optimization trick
epsilons_umap = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 1e-10, 0.0]

for eps in epsilons_umap:
    # rescale input
    pca_rescaled = pca2 / np.std(pca2[:, 0]) * rescale if rescale else pca2

    # load or compute the embedding
    filename = os.path.join(root_path, dataset,
                            f"umap_bin_k_{k}_n_epochs_{n_epochs}_anneal_lr_{anneal_lr}_eps_{eps}_seed_{seed}_a_{a}_b_{b}_init_pca_rescaled_{rescale}.pkl")
    try:
        with open(filename, "rb") as file:
            umapper = pickle.load(file)
    except FileNotFoundError:
        umapper = umap.UMAP(n_neighbors=k,
                            a=a,
                            b=b,
                            n_epochs=n_epochs,
                            negative_sample_rate=n_noise,
                            log_embeddings=True,
                            random_state=seed,
                            init=pca_rescaled,
                            graph=sknn_graph,
                            verbose=True,
                            anneal_lr=anneal_lr,
                            eps=eps)
        umapper.fit_transform(x)
        with open(filename, "wb") as file:
            pickle.dump(umapper, file, pickle.HIGHEST_PROTOCOL)

    print(f"done with eps {eps}")

print("UMAP on MNIST with varying eps with annealing")


