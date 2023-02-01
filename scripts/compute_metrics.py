import pickle
import numpy as np
import os

from utils import get_path, get_noise_in_estimator
from vis_utils.loaders import load_dataset
from vis_utils.utils import acc_kNN, corr_pdist_subsample, load_dict

root_path = get_path("data")
fig_path = get_path("figures")



k=15
seeds = [0, 1, 2]

a = 1.0
b = 1.0

n_noise = 5
parametric = False
loss_mode = "neg_sample"
batch_size = 1024
rescale = 1.0
anneal_lr = True
momentum = 0.0
lr_min_factor = 0.0
clamp_low = 1e-10
on_gpu = True
optimizer = "sgd"
n_epochs = 500
n_early_epochs = 250
init = "EE"


datasets = ["mnist"]  # "mnist", "imba_mnist_lin_seed_0", "zebrafish", "human-409b2", "c_elegans", "k49"]
suffixes = ["", "_umap", "_tsne"]

for dataset in datasets:
    print(f"Starting with dataset {dataset}")
    # get data
    x, y, sknn_graph, pca2 = load_dataset(root_path, dataset)
    if dataset.startswith("mnist") or dataset == "k49" or dataset == "cifar10":
        pca50 = np.load(os.path.join(root_path, dataset, "pca50.npy"))
        x = pca50

    # values for the spectrum
    nbs_noise_in_estimator = get_noise_in_estimator(len(x), n_noise, dataset)
    nbs_noise_in_estimator.sort()

    embedders = []
    for seed in seeds:
        # load embeddings
        embedders_by_seed = []
        for noise_in_estimator in nbs_noise_in_estimator:
            file_name = os.path.join(root_path,
                                     dataset,
                                     f"cne_{loss_mode}_n_noise_{n_noise}_noise_in_estimator_{noise_in_estimator}_n_epochs_{n_epochs}_init_{init}_bs_{batch_size}"
                                     f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                     )

            with open(file_name, "rb") as file:
                embedder = pickle.load(file)

            embedders_by_seed.append(embedder)
        embedders.append(embedders_by_seed)
        print(f"Loaded seed {seed}")

    # load tsne, only one random seed as deterministic
    perplexity = 2 * k  # has no effect as sknn_graph is used
    rescale_tsne = True
    log_kl = False
    log_embds = False
    log_Z = True


    try:
        file_name_tsne = os.path.join(root_path,
                                      dataset,
                                      f"tsne_bin_k_{k}_n_epochs_{n_epochs}_n_early_epochs_{n_early_epochs}_perplexity_{perplexity}_seed_{0}_log_kl_{log_kl}_log_embds_{log_embds}_init_pca_rescale_{rescale_tsne}.pkl")
        tsne_data = load_dict(file_name_tsne)
    except FileNotFoundError:
        file_name_tsne = os.path.join(root_path,
                                 dataset,
                                 f"tsne_bin_k_{k}_n_epochs_{n_epochs}_n_early_epochs_{n_early_epochs}_perplexity_{perplexity}_seed_{0}_init_pca_rescale_{rescale_tsne}.pkl")
        tsne_data = load_dict(file_name_tsne)

    embd_tsne = tsne_data["embd"]

    # load umap, only one seed computed
    lr = 1.0
    n_epochs_umap = 750
    rescale_umap = 1.0

    embds_umap = []
    for seed in seeds:
        file_name_umap = os.path.join(root_path, dataset,
                                f"umap_bin_k_{k}_n_epochs_{n_epochs_umap}_lr_{lr}_seed_{seed}_a_{a}_b_{b}_init_pca_rescaled_{rescale_umap}.pkl")

        with open(file_name_umap, "rb") as file:
            umapper = pickle.load(file)
            embds_umap.append(umapper.embedding_)

    # evaluate tsne and umap
    print(f"evaluating tSNE")

    # kNN recall
    print("Starting kNN recall")
    file_name_recall = os.path.join(root_path,
                                    dataset,
                                    f"tsne_recall.pkl"
                                    )
    try:
        with open(file_name_recall, "rb") as file:
            recall_dict = pickle.load(file)
    except FileNotFoundError:
        recall = acc_kNN(embd_tsne, x, k)
        recall_dict = {"recalls": recall}
        with open(file_name_recall, "wb") as file:
            pickle.dump(recall_dict, file, pickle.HIGHEST_PROTOCOL)

    # spearman correlation
    print("Starting spearman correlation")
    file_name_s_corr = os.path.join(root_path,
                                    dataset,
                                    f"tsne_s_corr.pkl"
                                    )
    try:
        with open(file_name_s_corr, "rb") as file:
            s_corr_dict = pickle.load(file)
    except FileNotFoundError:
        s_corrs = np.empty( len(seeds))
        for j, seed in enumerate(seeds):
            _, s_corr = corr_pdist_subsample(embd_tsne, x, sample_size=5000,
                seed=seed)
            s_corrs[j] = s_corr
        s_corr_dict = {"s_corrs": s_corrs}
        with open(file_name_s_corr, "wb") as file:
            pickle.dump(s_corr_dict, file, pickle.HIGHEST_PROTOCOL)

    print(f"evaluating UMAP")
    # kNN recall
    print("Starting kNN recall")
    file_name_recall = os.path.join(root_path,
                                    dataset,
                                    f"umap_recall.pkl"
                                    )
    try:
        with open(file_name_recall, "rb") as file:
            recall_dict = pickle.load(file)
    except FileNotFoundError:
        recalls = np.empty(len(seeds))
        for j, seed in enumerate(seeds):
            recalls[j] = acc_kNN(embds_umap[j], x, k)
        recall_dict = {"recalls": recalls}
        with open(file_name_recall, "wb") as file:
            pickle.dump(recall_dict, file, pickle.HIGHEST_PROTOCOL)

    # spearman correlation
    print("Starting spearman correlation")
    file_name_s_corr = os.path.join(root_path,
                                    dataset,
                                    f"umap_s_corr.pkl"
                                    )
    try:
        with open(file_name_s_corr, "rb") as file:
            s_corr_dict = pickle.load(file)
    except FileNotFoundError:
        s_corrs = np.empty(len(seeds))
        for j, seed in enumerate(seeds):
            _, s_corr = corr_pdist_subsample(embds_umap[j],
                                             x,
                                             sample_size=5000,
                                             seed=seed)
            s_corrs[j] = s_corr
        s_corr_dict = {"s_corrs": s_corrs}
        with open(file_name_s_corr, "wb") as file:
            pickle.dump(s_corr_dict, file, pickle.HIGHEST_PROTOCOL)

    # compute metrics over spectra
    gt_xs = [x, embds_umap[0], embd_tsne]
    print(f"evaluating spectra")
    for suffix, gt_x in zip(suffixes, gt_xs):

        # kNN recall
        print("Starting kNN recall")
        file_name_recall = os.path.join(root_path,
                                        dataset,
                                        f"cne_{loss_mode}_n_noise_{n_noise}_n_epochs_{n_epochs}_init_{init}_bs_{batch_size}"
                                        f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_recall{suffix}.pkl"
                                        )
        try:
            with open(file_name_recall, "rb") as file:
                recall_dict = pickle.load(file)
        except FileNotFoundError:
            recalls = np.empty((len(nbs_noise_in_estimator), len(seeds)))
            for i, noise_in_estimator in enumerate(nbs_noise_in_estimator):
                for j, seed in enumerate(seeds):
                    recalls[i, j] = acc_kNN(embedders[j][i].callback.embds[-1], gt_x,
                                            k)
                print(f"Done with {noise_in_estimator}")
            recall_dict = {"nbs_noise_in_estimator": nbs_noise_in_estimator,
                           "recalls": recalls}
            with open(file_name_recall, "wb") as file:
                pickle.dump(recall_dict, file, pickle.HIGHEST_PROTOCOL)

        # spearman correlation
        print("Starting spearman correlation")
        file_name_s_corr = os.path.join(root_path,
                                        dataset,
                                        f"cne_{loss_mode}_n_noise_{n_noise}_n_epochs_{n_epochs}_init_{init}_bs_{batch_size}"
                                        f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_s_corr{suffix}.pkl"
                                        )
        try:
            with open(file_name_s_corr, "rb") as file:
                s_corr_dict = pickle.load(file)
        except FileNotFoundError:
            s_corrs = np.empty((len(nbs_noise_in_estimator), len(seeds)))
            for i, noise_in_estimator in enumerate(nbs_noise_in_estimator):
                for j, seed in enumerate(seeds):
                    _, s_corr = corr_pdist_subsample(
                        embedders[j][i].callback.embds[-1], gt_x, sample_size=5000,
                        seed=seed)
                    s_corrs[i, j] = s_corr
                print(f"Done with {noise_in_estimator}")
            s_corr_dict = {"nbs_noise_in_estimator": nbs_noise_in_estimator,
                           "s_corrs": s_corrs}
            with open(file_name_s_corr, "wb") as file:
                pickle.dump(s_corr_dict, file, pickle.HIGHEST_PROTOCOL)

        print(f"Done with mode {suffix}")

    # for mnist, also evaluate nce and infonce:
    inits = ["random", "pca", "EE"]
    init_strs = ["random_rescale_1.0", "pca_rescale_1.0", "EE"]
    modes = ["nce", "infonce"]
    n_noises = [5, 50, 500]
    n_epochs_list = [750, 750, 500]
    if dataset == "mnist":
        print("Starting with m ablation")
        for initialization, init_str, epochs in zip(inits, init_strs, n_epochs_list):
            for mode in modes:
                for m in n_noises:
                    embedders_nc = []
                    for seed in seeds:
                        file_name_nc = os.path.join(root_path,
                                                    dataset,
                                                    f"cne_{mode}_n_noise_{m}_n_epochs_{epochs}_init_{init_str}_bs_{batch_size}"
                                                    f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                                    )

                        with open(file_name_nc, "rb") as file:
                            embedder = pickle.load(file)
                        embedders_nc.append(embedder)

                    for suffix, gt_x in zip(suffixes, gt_xs):
                        if suffix == "_umap": continue

                        # kNN recall
                        print("Starting kNN recall")
                        file_name_recall = os.path.join(root_path,
                                                        dataset,
                                                        f"cne_{mode}_n_noise_{m}_n_epochs_{epochs}_init_{init_str}_bs_{batch_size}"
                                                        f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_recall{suffix}.pkl"
                                                        )
                        try:
                            with open(file_name_recall, "rb") as file:
                                recall_dict = pickle.load(file)
                        except FileNotFoundError:
                            recalls = np.empty(len(seeds))
                            for j, seed in enumerate(seeds):
                                recalls[j] = acc_kNN(
                                    embedders_nc[j].callback.embds[-1],
                                    gt_x,
                                    k)
                            recall_dict = {
                                "recalls": recalls}
                            with open(file_name_recall, "wb") as file:
                                pickle.dump(recall_dict, file,
                                            pickle.HIGHEST_PROTOCOL)

                        # spearman correlation
                        print("Starting spearman correlation")
                        file_name_s_corr = os.path.join(root_path,
                                                        dataset,
                                                        f"cne_{mode}_n_noise_{m}_n_epochs_{epochs}_init_{init_str}_bs_{batch_size}"
                                                        f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{parametric}_clamp_low_{clamp_low}_s_corr{suffix}.pkl"
                                                        )
                        try:
                            with open(file_name_s_corr, "rb") as file:
                                s_corr_dict = pickle.load(file)
                        except FileNotFoundError:
                            s_corrs = np.empty(len(seeds))
                            for j, seed in enumerate(seeds):
                                _, s_corr = corr_pdist_subsample(
                                    embedders_nc[j].callback.embds[-1],
                                    gt_x, sample_size=5000,
                                    seed=seed)
                                s_corrs[j] = s_corr
                            s_corr_dict = {
                                "s_corrs": s_corrs}
                            with open(file_name_s_corr, "wb") as file:
                                pickle.dump(s_corr_dict, file,
                                            pickle.HIGHEST_PROTOCOL)
                        print(
                            f"Done with mode {mode}, init {initialization}, m {m}, suffix {suffix}")

    print(f"Done with dataset {dataset}")



