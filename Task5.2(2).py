import argparse

import scanpy as sc
import numpy as np
from keras import Model
from keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, Flatten
import tensorflow as tf
from keras.src.layers import Reshape
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-dataset_path', '--dataset_path', type=str, help='Specify the dataset path')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    adata = sc.read(dataset_path)
    random_indices = np.random.randint(0, adata.shape[0], size=10000)
    adata = adata[random_indices, :]
    data = adata.obsm['X_harmony']
    cell_types = adata.obs.cell_type_original
    input_shape = len(data[0])
    embed_dim = input_shape
    num_heads = 2
    ff_dim = 256
    input = Input(shape=input_shape, name='input')
    reshaped_input = Reshape((input_shape, 1))(input)  # Add a reshape layer
    conv1 = Conv1D(filters=32, kernel_size=8, strides=2, activation='relu', padding='same')(reshaped_input)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(bn1)
    bn2 = BatchNormalization()(conv2)
    do = Dropout(0.1)(bn2)
    flat = Flatten()(do)
    output = Dense(len(np.unique(cell_types)), activation="softmax", name='output')(flat)
    supervised_model = Model(inputs=input, outputs=output)
    label_encoder = LabelEncoder()
    encoded_cell_types = label_encoder.fit_transform(cell_types)
    supervised_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy')
    train_data, test_data, train_labels, test_labels = train_test_split(data, encoded_cell_types, test_size=0.3, random_state=42)
    over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    train_data, train_labels = over_sampler.fit_resample(train_data, train_labels)
    supervised_model.fit(train_data, train_labels, verbose=1, epochs=10, shuffle=False)
    predictions = supervised_model.predict(test_data)
    predicted_indices = np.argmax(predictions, axis=1)
    labels = np.unique(cell_types).tolist()
    conf_matrix = confusion_matrix(test_labels, predicted_indices)
    cmap = sns.light_palette("navy", as_cmap=True)
    sns.set(font_scale=0.75)
    heatmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, cbar=True, square=True, xticklabels=labels, yticklabels=labels, linewidths=.5, annot_kws={"size": 12})
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Transformer Model Heatmap")
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Count', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()
