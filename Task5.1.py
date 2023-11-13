import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

adata = sc.read('C:/Users/Shoaib/Desktop/snRNA-seq-submission.h5ad')
random_indices = np.random.randint(0, adata.shape[0], size=10000)
adata = adata[random_indices, :]

#Task 5.1. What are some basic filters commonly used for single-cell data processing
#(Hint: number of genes, UMI count, fraction of reads mapped to mitochondrial genes). Plot those three features on the MI human dataset.

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_cells(adata, max_counts=5000)
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
adata = adata[adata.obs['percent_mito'] < 0.05]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(adata.obs['n_genes'], bins=50, kde=False)
plt.title('Number of Genes per Cell')

plt.subplot(1, 3, 2)
sns.histplot(adata.obs['n_counts'], bins=50, kde=False)
plt.title('UMI Count per Cell')

plt.subplot(1, 3, 3)
sns.histplot(adata.obs['percent_mito'], bins=50, kde=False)
plt.title('Fraction of Reads Mapped to Mitochondrial Genes')

plt.tight_layout()
plt.show()