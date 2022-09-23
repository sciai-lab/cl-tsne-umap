#!/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
from cne import ContrastiveEmbedding
from utils import get_path


def get_transforms(mean, std, size, setting="train"):
    normalize = transforms.Normalize(mean=mean, std=std)

    if setting == "full":
        transform = transforms.Compose(
            [
                transforms.functional.to_pil_image,
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.functional.to_tensor,
                normalize,
            ]
        )

    elif setting == "train":
        transform = transforms.Compose(
            [
                transforms.functional.to_pil_image,
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "test":
        transform = transforms.Compose(
            [
                transforms.functional.to_pil_image,
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown transform {setting = }")

    return transform


class DistortionTransformData(Dataset):
    """Returns a pair of neighboring points in the dataset."""

    def __init__(self, dataset, mean, std, size, reshape=None):
        self.dataset = dataset
        if self.dataset.max() <= 1.0:
            self.dataset *= 255
        self.dataset = self.dataset.astype(np.uint8)

        if reshape is not None:
            # lambda d: torch.reshape(d, (-1, 28, 28))
            self.dataset = reshape(self.dataset)
        else:
            self.dataset = np.reshape(self.dataset, (-1, *size))

        self.transform = get_transforms(mean, std, size[:2], setting="full")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i, :, :]

        return self.transform(item), self.transform(item)


class TransformLabelData(torch.utils.data.Dataset):
    """Returns a pair of neighboring points in the dataset."""

    def __init__(
        self,
        dataset,
        labels,
        mean,
        std,
        size,
        reshape=None,
        setting="train",
    ):
        self.dataset = dataset
        self.labels = labels
        if self.dataset.max() <= 1.0:
            self.dataset *= 255
        self.dataset = self.dataset.astype(np.uint8)

        if reshape is not None:
            # lambda d: torch.reshape(d, (-1, 28, 28))
            self.dataset = reshape(self.dataset)
        else:
            self.dataset = np.reshape(self.dataset, (-1, *size))

        self.transform = get_transforms(mean, std, size[:2], setting=setting)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i, :, :]

        return self.transform(item), self.labels[i]


class LR_Scheduler(object):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        warmup_lr,
        num_epochs,
        base_lr,
        final_lr,
        iter_per_epoch,
        constant_predictor_lr=True,
        set_warmup_iter=-1,
    ):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        if set_warmup_iter > 0:
            warmup_iter = set_warmup_iter
            decay_iter = num_epochs * iter_per_epoch - warmup_iter
        else:
            warmup_iter = iter_per_epoch * warmup_epochs
            decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)

        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

    def set_iter(self, set_iter):
        self.iter = set_iter


class FCNetwork(nn.Module):
    "Fully-connected network"

    def __init__(self, in_dim=784, feat_dim=128, hidden_dim=100):
        super(FCNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ResNetFC(nn.Module):
    def __init__(
        self,
        model_name="resnet18",
        in_channel=3,
        feat_dim=2,
        hidden_dim=100,
        normalize=False,
    ):
        super(ResNetFC, self).__init__()
        if model_name == "resnet18":
            self.resnet = resnet18()
            in_dim = 512
        elif model_name == "resnet50":
            self.resnet = resnet50()
            in_dim = 2048
        self.backbone_dim = in_dim
        self.feat_dim = feat_dim
        self.resnet.output_dim = in_dim # self.resnet.fc.in_features
        #self.resnet.fc = torch.nn.Identity()

        self.fc = FCNetwork(in_dim=in_dim, feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.normalize = normalize

    def forward(self, x):
        h = self.resnet(x)
        z = self.fc(h)
        if self.normalize:
            z = F.normalize(z, dim=1)
        return z


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out  = self.conv3(out)
        print(out.shape)
        out = self.bn3(out)
        #out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def main():
    epochs = 1 # 1000
    epochs_warmup = 10
    epochs_linear_eval = 100
    batch_size = 512
    negative_samples = 16  # or 2 * batch_size
    learning_rate = 0.03 * batch_size / 256
    loss_mode = "neg_sample"  # or infonce, nce, ...
    n_dim = 128
    metric = "cosine"
    temperature = 0.5
    optimizer = "sgd"
    weight_decay = 5e-4
    lr_anneal = "cosine"
    num_workers = 8
    device = "cuda:0"
    model_name = "resnet50"
    root_dir = get_path("data")
    run = 0 # just used for numbering

    rng = np.random.default_rng(4123)

    ## def get_dataset():
    cifar = fetch_openml("CIFAR_10",
                         data_home=os.path.join(root_dir, "cifar10")
                         )
    # Separate the color channels and move them to the back.
    data = np.moveaxis(cifar.data.reshape(60000, 3, 32, 32), 1, -1)
    labels = np.vectorize(np.int8)(cifar.target)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    dim = (32, 32, 3)

    # full dataset for training SimCLR
    dataset = DistortionTransformData(data, mean, std, dim)

    seed = int(rng.integers(2**63))
    gen = torch.Generator().manual_seed(seed)
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
        clamp_low=0.0,
        temperature=temperature,
        print_freq_epoch=100,
    )

    # contrastive training
    print("1: training CLR model")
    cne.fit(loader)
    file_name = os.path.join(root_dir, "cifar10", "results", f"{model_name}_m_{negative_samples}_run_{run}")
    import pickle
    with open(file_name, "wb") as file:
        pickle.dump(cne, file, pickle.HIGHEST_PROTOCOL)

    ## linear evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, random_state=rng.integers(2**32 - 1), test_size=10000
    )

    train_dataset = TransformLabelData(
        X_train, y_train, mean, std, dim, setting="train"
    )
    seed = int(rng.integers(2**63))
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=False,
    )
    test_dataset = TransformLabelData(X_test, y_test, mean, std, dim, setting="test")
    seed = int(rng.integers(2**63))
    gen = torch.Generator().manual_seed(seed)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=False,
    )
    full_dataset = TransformLabelData(data, labels, mean, std, dim, setting="train")
    seed = int(rng.integers(2**63))
    gen = torch.Generator().manual_seed(seed)
    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=gen,
        drop_last=False,
    )

    classifier = torch.nn.Linear(
        in_features=512, out_features=10, bias=True, device=device
    )
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=30,
        momentum=0.9,
        weight_decay=0,
    )
    lr_scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs=0,
        warmup_lr=0,
        num_epochs=epochs_linear_eval,
        base_lr=30,
        final_lr=0,
        iter_per_epoch=len(train_loader),
    )

    model_transform = model.resnet

    # warm up batchnorm
    if True:
        model.train()
        with torch.no_grad():
            for _ in range(5):
                for (images, labels) in full_loader:
                    model_transform(images.to(device))

    model.eval()
    classifier.train()
    print("2: training linear classifier")
    for epoch in range(epochs_linear_eval):
        for (images, labels) in train_loader:
            labels = labels.type(dtype=torch.long).to(device)
            with torch.no_grad():
                features = model_transform(images.to(device))
            preds = classifier(features)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    classifier.eval()
    n = 0
    c = 0
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            features = model_transform(images.to(device))
            preds = classifier(features).argmax(dim=1)
            correct = (preds == labels.to(device)).sum().item()
            new_n = preds.shape[0]
            c += correct
            n += new_n

    print(f"3: done, accuracy is {c / n}")


if __name__ == "__main__":
    main()
