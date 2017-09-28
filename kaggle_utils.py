from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot.metrics as skplt
import seaborn as sns


# Feature scaling with StandardScaler
def feature_scaler(df, fields):
    scaler = StandardScaler()
    for field in fields:
        feature = np.array(df[field]).reshape((len(df[field]), 1))
        scaler.fit(feature)
        df[field] = scaler.transform(feature)
    return df

# Feature preprocessing with LabelEncoder
def feature_encoder(df, fields):
    encoder = LabelEncoder()
    for field in fields:
        df[field] = encoder.fit_transform(df[field])
    return df

# Feature selector
def feature_selector(df, recols):
    fields = [each for each in df.columns if each not in recols]
    return fields

# Correlation visualization with Heatmap
def feature_correlation(df, fields):
    corr_all = df[fields].corr()
    mask = np.zeros_like(corr_all, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize = (11, 9))
    sns.heatmap(corr_all, mask = mask, square = True, linewidths = .5, ax = ax, cmap = "BuPu")      
    plt.show()

# Dimensionality reduction with PCA
def feature_reduction(df, fields, label):
    x = df[fields]
    y = df[label]

    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(x)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

# Confusion matrix with classification score
def evaluate_models(y, preds):
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
    print('ROC-AUC: {}'.format(roc_auc_score(y, preds)))
    print('Recall: {}'.format(recall_score(y, preds)))
    skplt.plot_confusion_matrix(y, preds, figsize=(8, 6))
