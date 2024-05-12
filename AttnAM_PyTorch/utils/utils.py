import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix


def get_top_index_and_attention(data_to_explain,
                                model):
    top_attention, top_feature_idx = model.get_top_feature_index(data_to_explain)

    top_attention = top_attention.detach().cpu().numpy()
    top_attention = np.transpose(top_attention)

    top_feature_idx = top_feature_idx.detach().cpu().numpy()
    feature_idx_df = pd.DataFrame(top_feature_idx)

    feature_idx = []
    for i in list(feature_idx_df.columns):
        # print(feature_idx_df[i].value_counts()[:2])
        idx = feature_idx_df[i].mode().values
        feature_idx.append(idx)

    feature_idx = np.sort(np.array(feature_idx).reshape(-1, ).astype(int))

    return top_attention, feature_idx


def get_new_unique_features_with_weights(single_features, weights):
    unique_features = []
    unique_feature_weights = []
    for feature, weight in zip(single_features, weights):
        weight = weight.reshape(-1, 1)
        unique_feature, ind = np.unique(feature, axis=0, return_index=True)
        unique_feature_weight = weight[ind]

        unique_features.append(unique_feature)
        unique_feature_weights.append(unique_feature_weight)

    return unique_features, unique_feature_weights


def get_new_feature_names(feature_names, feature_idx):
    new_feature_names = []
    for i in feature_idx:
        new = feature_names[i]
        new_feature_names.append(new)

    return new_feature_names


def get_new_features(dataset, feature_idx):
    new_features_list = []
    for i in range(dataset.features.shape[0]):
        new = [dataset.features[i, :][x].cpu().detach().numpy() for x in feature_idx]
        new_features_list.append(new)
    new_features_array = np.array(new_features_list)
    new_features = torch.from_numpy(new_features_array)

    return new_features


def get_new_single_features(dataset, feature_names):
    new_single_features_array = []
    for i in feature_names:
        new = dataset.single_features[i]
        new_single_features_array.append(new)

    new_single_features = dict(zip(feature_names,
                                   new_single_features_array))

    return new_single_features


def compute_mape(true, pred):

    eps = np.finfo(np.float64).eps

    return np.mean(np.abs(true - pred) / np.maximum(np.abs(true), eps))


def compute_wape(true, pred):

    nominator = np.sum(np.abs(true - pred))
    denominator = np.sum(np.abs(true))

    return nominator / denominator


def compute_mase(train, true, pred):
    pred_naive = []
    for i in range(1, len(train)):
        pred_naive.append(train[(i - 1)])

    mae_naive = np.mean(abs(train[1:] - pred_naive))

    return np.mean(abs(true - pred)) / mae_naive


def compute_sampe(true, pred):

    nominator = np.abs(true - pred)
    denominator = np.abs(true) + np.abs(pred)

    return np.mean(2.0 * nominator / denominator)


def print_error_estimator(true, pred, train, regression=True):

    if regression:
        print('-----------------------------')
        smape = compute_sampe(true, pred)
        print("sMAPE is:", smape)
        mase = compute_mase(train, true, pred)
        print("MASE is:",  mase)
        mape = compute_mape(true, pred)
        print('MAPE is:', mape)
        wape = compute_wape(true, pred)
        print('WAPE is', wape)
        print('-----------------------------')
    else:
        print("Accuracy is: ", accuracy_score(true, pred))
        print("Precision is: ", precision_score(true, pred))
        print("Recall is: ", recall_score(true, pred))
        print("F1 Score is: ", f1_score(true, pred))
        print("AUC Score is: ", roc_auc_score(true, pred))


def calc_error_estimator(true, pred, train, regression=True):
    if regression:
        smape = compute_sampe(true, pred)
        mase = compute_mase(train, true, pred)
        mape = compute_mape(true, pred)
        wape = compute_wape(true, pred)

        return smape, mape, mase, wape
    else:
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        auc = roc_auc_score(true, pred)

        return accuracy, precision, recall, f1, auc


def plot_cm(labels, predictions):
    p = 0.5
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def plot_prediction(labels, predictions):
    plt.figure(figsize=(30, 10))
    plt.plot(predictions, label='Prediction')
    plt.plot(labels, label='True')
    plt.legend()
    plt.show()



