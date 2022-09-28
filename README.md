This repository contains scripts and notebooks to reproduce the experiments in 
**Contrastive learning unifies t-SNE and UMAP**. 

It depends on several other repositories, in particular [contrastive-ne](https://github.com/berenslab/contrastive-ne), which implement the actual logic and contain utilities.

# Installation
Create and activate the conda environment
```
conda env create -f environment.yml
conda activate cl_tsne_umap
```
Install `openTSNE`, `vis_utis`, `umap`, `ncvis` and `cne`
```
git clone https://github.com/sdamrich/openTSNE
cd openTSNE
python setup.py install
cd ..

git clone https://github.com/sdamrich/vis_utils
cd vis_utils
python setup.py install
cd ..

git clone https://github.com/sdamrich/UMAPs-true-loss
cd UMAPs-true-loss
python setup.py install
cd ..

git clone https://github.com/sdamrich/ncvis
cd ncvis
make libs
make wrapper

git clone -b arxiv-v1 https://github.com/berenslab/contrastive-ne
pip install --no-deps . 
cd ..
```
# Usage
To reproduce the Neg-t-SNE embeddings from Fig. 1 a)-e), run
```
python scripts/compute_embds_cne.py
```
and check out the results in `notebooks/negtsne.ipynb`.

<img width="400" alt="Neg-t-SNE on MNIST" src="/figures/Fig_1_a-e.png">


To reproduce the UMAP embeddings from Fig. S1 a)-c), run
```
python scripts/compute_embds_umap.py
```
and check out the results in `notebooks/umap_vs_negtsne.ipynb`.

<img width="400" alt="UMAP no annealing" src="/figures/Fig_S1_a-c.png">


To reproduce the SimCLR experiments with `m=16`, run
```
python cne_scripts_notebooks/scripts/cifar10-acc.py
```

For other experiments adapt the parameters at the top of `compute_embds_cne.py`
and `compute_embds_umap.py` or at the top of the `main` function in `cifar10-acc.py`
accordingly. Downloaded datasets and neighbor embedding results will be saved in `cne_scripts_notebooks/data` and figures 
will be saved in `cne_scripts_notebooks/figures`.

All neighbor embedding results alongside their parameters can be 
inspected in the jupyter notebooks in `cne_scripts_notebooks/notebooks`.
This list details which figures can be inspected using which notebooks:

- Fig 1:  `negtsne.ipynb`, `tsne.ipynb`, `ncvis.ipynb`
- Fig 2:  `umap_vs_negtsne.ipynb`
- Fig 3:  `parametric.ipynb`
- Fig S1: `umap_vs_negtsne.ipynb`
- Fig S2: `umap_vs_negtsne.ipynb`
- Fig S3: `attr_rep_plot_UMAP_neg.ipynb`
- Fig S4: `umap_vs_negtsne_vary_n_noise.ipynb`
- Fig S5: `tsne_vs_ncvis.ipynb`
- Fig S6: `tsne_vs_ncvis.ipynb`
- Fig S7: `negtsne.ipynb`, `tsne_ipynb`, `ncvis.ipynb`
- Fig S8: `imba_mnist_negtsne.ipynb`, `imba_mnist_tsne_ncvis_umap.ipynb`
- Fig S9: `human_negtsne.ipynb`, `human_tsne_ncvis_umap.ipynb`
- Fig S10: `zebrafish_negtsne.ipynb`, `zebrafish_tsne_ncvis_umap.ipynb`
- Fig S11: `c_elegans_negtsne.ipynb`, `c_elegans_tsne_ncvis_umap.ipynb`
- Fig S12: `k49_negtsne.ipynb`, `k49_tsne_ncvis_umap.ipynb`
- Fig S13: `k49_negtsne.ipynb`
- Fig S14: `ncvis.ipynb`, `tsne.ipynb`
- Fig S15: `infonctsne.ipynb`, `tsne.ipynb`