import argparse
import pickle
import scanpy as sc
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from devCellPy.layer import Layer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-dataset_path', '--dataset_path', type=str, help='Specify the dataset path')
args = parser.parse_args()
dataset_path = args.dataset_path
adata = sc.read(dataset_path)
random_indices = np.random.randint(0, adata.shape[0], size=10000)
adata = adata[random_indices, :]

model_path = 'Cardiac Atlas Trained Models/Root/Root_object.pkl'
with open(model_path, 'rb') as file:
    layer = pickle.load(file)
model = layer.xgbmodel

data = adata.raw.X.toarray().tolist()
cell_types = adata.obs.cell_type_original.to_list()
cell_types = ["Endothelial Cells" if cell == "Endothelial" else cell for cell in cell_types]
cell_types = ["Cardiomyocytes" if cell == "Cardiomyocyte" else cell for cell in cell_types]
cell_types = ["Smooth Muscle Cells" if cell == "vSMCs" else cell for cell in cell_types]

observed_cell_types = []
mi_data = []
for data, cell_type in zip(data, cell_types):
    if cell_type in ["Endothelial Cells", "Cardiomyocytes", "Smooth Muscle Cells"]:
        mi_data.append(data)
        observed_cell_types.append(cell_type)

devcell_labels = list(layer.labeldict.values())
mi_features = adata.raw.var_names.tolist()
devcell_features = model.feature_names
matrix = []

'''
indices = []
for devcell_index, devcell_feature in enumerate(devcell_features):
    if devcell_feature in mi_features:
        mi_index = mi_features.index(devcell_feature)
        indices.append((devcell_index, mi_index))

for obs in tqdm(mi_data, desc='Creating matrix...'):
    m = [-1] * len(devcell_features)
    for devcell_index, mi_index in indices:
            mi_feature_value = obs[mi_index]
            m[devcell_index] = mi_feature_value
    matrix.append(m)
'''

for obs in mi_data:
    obs = obs + [0]*(len(devcell_features) - len(mi_features))
    matrix.append(obs)
matrix = xgb.DMatrix(matrix, feature_names=devcell_features)
predictions = model.predict(matrix)
predicted_indices = np.argmax(predictions, axis=1)
predicted_cell_types = [devcell_labels[i] for i in predicted_indices]
conf_matrix = confusion_matrix(observed_cell_types, predicted_cell_types, labels=devcell_labels)
cmap = sns.light_palette("navy", as_cmap=True)
sns.set(font_scale=0.75)
heatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, cbar=True, square=True, xticklabels=devcell_labels, yticklabels=devcell_labels, linewidths=.5, annot_kws={"size": 12})
plt.xticks(rotation=25, ha="right")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DevCellPy Heatmap")
cbar = heatmap.collections[0].colorbar
cbar.set_label('Count', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()