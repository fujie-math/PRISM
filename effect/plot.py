import os
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def get_data(root_path, type_):
    hd = json.load(open(os.path.join(root_path, f"{type_}_last_token.json"), encoding='utf-8'))
    halu_hd = []
    for hs in hd:
        halu_hd.append(hs["hd"])

    enddata = []
    for i in range(len(halu_hd)):
        enddata.append({
            "hd": halu_hd[i],
            "label": hd[i]["label"],
            "type": type_
        })
    return enddata


data_type = ["animals", "cities", "companies", "elements", "facts", "inventions"]
Data_Type = ["Animals", "Cities", "Companies", "Elements", "Facts", "Inventions"]


data_path_1 = "../hd_data/true/llama2chat7b"
data_path_2 = "./prompt_1_hd/"


def prepare_data(data_path, data_type):
    pca_data = []
    for n, dt in enumerate(data_type):
        data = get_data(data_path, dt)
        pca_data += data
    return pca_data


fig, axes = plt.subplots(3, 4, figsize=(14, 9))


for i, dt in enumerate(data_type):

    pca_data_1 = get_data(data_path_1, dt)
    pca_data_2 = get_data(data_path_2, dt)


    features_1 = np.array([item["hd"] for item in pca_data_1])
    labels_1 = np.array([item["label"] for item in pca_data_1])

    features_2 = np.array([item["hd"] for item in pca_data_2])
    labels_2 = np.array([item["label"] for item in pca_data_2])


    pca = PCA(n_components=2)
    principal_components_1 = pca.fit_transform(features_1)
    principal_components_2 = pca.fit_transform(features_2)


    clf_1 = LogisticRegression()
    clf_1.fit(principal_components_1, labels_1)

    clf_2 = LogisticRegression()
    clf_2.fit(principal_components_2, labels_2)
    
    name = Data_Type[i]


    ax1 = axes[i // 2, (i * 2) % 4]
    scatter_1 = ax1.scatter(principal_components_1[:, 0], principal_components_1[:, 1], c=labels_1, cmap='viridis', alpha=0.7, s=2)  
    ax1.set_title(f'{name}', fontsize=10)


    ax1.grid(True)
    ax1.set_xticks([])
    ax1.set_yticks([])

    coef_1 = clf_1.coef_[0]
    intercept_1 = clf_1.intercept_

    slope_1 = -coef_1[0] / coef_1[1]
    intercept_1 = -intercept_1 / coef_1[1]

    x_vals_1 = np.array(ax1.get_xlim())
    y_vals_1 = slope_1 * x_vals_1 + intercept_1
    ax1.plot(x_vals_1, y_vals_1, color='black', linewidth=0.5, linestyle='--')
    

    ax2 = axes[i // 2, (i * 2 + 1) % 4]
    scatter_2 = ax2.scatter(principal_components_2[:, 0], principal_components_2[:, 1], c=labels_2, cmap='plasma', alpha=0.7, s=2)
    ax2.set_title(f'{name} - Prompt 1', fontsize=10)


    ax2.grid(True)
    ax2.set_xticks([])
    ax2.set_yticks([])

    coef_2 = clf_2.coef_[0]
    intercept_2 = clf_2.intercept_

    slope_2 = -coef_2[0] / coef_2[1]
    intercept_2 = -intercept_2 / coef_2[1]

    x_vals_2 = np.array(ax2.get_xlim())
    y_vals_2 = slope_2 * x_vals_2 + intercept_2
    ax2.plot(x_vals_2, y_vals_2, color='black', linewidth=1, linestyle='--')


legend1 = scatter_1.legend_elements()[0]
legend2 = scatter_2.legend_elements()[0]


labels1 = ['0', '1']
labels2 = ['0 (Prompt)', '1 (Prompt)']


handles = legend1 + legend2
labels = labels1 + labels2


fig.legend(handles, labels, title="Labels", loc="upper right")

plt.subplots_adjust(wspace=0.1, hspace=0.3)
plt.savefig('pca1.pdf')
