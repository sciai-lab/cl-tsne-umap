#!/bin/env python3

from cne import ContrastiveEmbedding
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml
import os
from utils import get_path, mean, std, dim, DistortionTransformData, ResNetFC, check_knn_acc, check_lin_acc
import time
import pickle


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m")
    parser.add_argument("-r")
    args = parser.parse_args()

    run = int(args.r)  # serves as seed
    negative_samples = int(args.m)  # or 2 * batch_size
    loss_mode = "infonce"  # or infonce, nce, neg_sample, ...

    epochs = 1000
    epochs_warmup = 5
    batch_size = 1024
    learning_rate = 0.03 * batch_size / 256
    n_dim = 128
    hidden_dim = 1024
    metric = "cosine"
    temperature = 0.5
    optimizer = "sgd"
    weight_decay = 5e-4
    lr_anneal = "cosine"
    clamp_low = float("-inf")
    clamp_high = float("inf")
    num_workers = 8
    device = "cuda:0"
    model_name = "resnet18"
    root_dir = get_path("data")

    # def get_dataset():
    cifar = fetch_openml("CIFAR_10",
                         data_home=os.path.join(root_dir, "cifar10")
                         )
    # Separate the color channels and move them to the back.
    data = np.moveaxis(cifar.data.reshape(60000, 3, 32, 32), 1, -1)
    labels = np.vectorize(np.int8)(cifar.target)

    # full dataset for training SimCLR
    dataset = DistortionTransformData(data, mean, std, dim)

    gen = torch.Generator().manual_seed(run)
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=True,
    )

    model = ResNetFC(
        model_name=model_name,
        in_channel=3,
        feat_dim=n_dim,
        hidden_dim=hidden_dim
    )

    cne = ContrastiveEmbedding(
        model,
        n_epochs=epochs,
        batch_size=batch_size,
        negative_samples=negative_samples,
        device=device,
        learning_rate=learning_rate,
        loss_mode=loss_mode,
        metric=metric,
        optimizer=optimizer,
        anneal_lr=lr_anneal,
        weight_decay=weight_decay,
        warmup_epochs=epochs_warmup,
        warmup_lr=0,
        clamp_high=clamp_high,
        clamp_low=clamp_low,
        temperature=temperature,
        print_freq_epoch=100,
    )

    file_name = os.path.join(root_dir, "cifar10", "results",
                             f"{model_name}_epochs_{epochs}_m_{negative_samples}_run_{run}.pkl")

    print(f"0: starting with {file_name}")
    # contrastive training
    print("1: training CLR model")

    try:
        with open(file_name, "rb") as file:
            cne = pickle.load(file)
    except:
        start = time.time()
        cne.fit(loader)
        cne.time = time.time() - start

        print(f"Time: {cne.time / 60} mins")
        with open(file_name, "wb") as file:
            pickle.dump(cne, file, pickle.HIGHEST_PROTOCOL)

    # linear evaluation
    print("2: training linear classifier", end="")
    lin_acc = check_lin_acc(data=data,
                            labels=labels,
                            embedder=cne,
                            file_name=file_name,
                            run=run,
                            test_size=10000,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device
                            )
    print(f", done with accuracy {lin_acc}")

    print("3: training knn classifier", end="")
    # knn evaluation
    knn_acc = check_knn_acc(data=data,
                            labels=labels,
                            embedder=cne,
                            file_name=file_name,
                            run=run,
                            test_size=10000,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            device=device
                            )

    print(f",    done with accuracy {knn_acc}")

    print(f"4: Done with {file_name}")


if __name__ == "__main__":
    main()
