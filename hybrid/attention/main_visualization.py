import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler





with open("dataset/human_modified.txt", "r") as f:
#with open("dataset/yanzheng1.txt", "r") as f:
    data_list = f.read().strip().split('\n')
data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
N = len(data_list)
_, sequence, _ = data_list[0].strip().split()
sequence_f =sequence[:-2]
print(len(sequence_f))


df = pd.read_csv('trgattn.csv')
df = df.values

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(df)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
X_pca = scaler.fit_transform(X_pca)
print("X_pca.shape:", X_pca.shape)
print(pca.explained_variance_ratio_)

rows_per_chunk = 40
num_chunks = (len(X_pca) + rows_per_chunk - 1) // rows_per_chunk  

fig, axes = plt.subplots(num_chunks, 1, figsize=(15, 0.4 * num_chunks), sharex=False, sharey=True)

if num_chunks == 1:
    axes = [axes]

sequence_labels = list(sequence_f)  

for i in range(num_chunks):
    start_idx = i * rows_per_chunk
    end_idx = min(start_idx + rows_per_chunk, len(X_pca))
    chunk_data = X_pca[start_idx:end_idx, :].reshape(-1, 1).T  

    if chunk_data.shape[1] < rows_per_chunk:
        chunk_data = np.pad(chunk_data, ((0, 0), (0, rows_per_chunk - chunk_data.shape[1])), mode='constant',
                            constant_values=0)
        sequence_labels_chunk = sequence_labels[start_idx:end_idx] + [''] * (rows_per_chunk - (end_idx - start_idx))
    else:
        sequence_labels_chunk = sequence_labels[start_idx:end_idx]

    sns.heatmap(chunk_data, annot=False, fmt=".2f", cmap="coolwarm", ax=axes[i], cbar=False, linewidths=1.2)
    axes[i].set_xticks(np.arange(chunk_data.shape[1]) + 0.5)  
    axes[i].set_xticklabels(sequence_labels_chunk, rotation=0)  
    axes[i].set_yticks([])

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
fig.colorbar(axes[-1].collections[0], cax=cbar_ax)
cbar_ax.set_ylabel('Value', fontsize=14)

plt.subplots_adjust(hspace=4.5)  
plt.tight_layout(rect=[0, 0, 0.9, 1])  
plt.savefig('RDV-TP.png')
plt.show()