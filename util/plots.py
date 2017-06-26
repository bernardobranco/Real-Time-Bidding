import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_auc(predictions, model_name):
    scores = {}
    auc = plot_roc_auc_combined_dataset(predictions, model_name)
    scores.update({"all": {"combined": auc, "advertiser": auc}})
    scores.update(plot_roc_auc_per_advertiser(predictions, model_name))

    return scores

def plot_roc_auc_combined_dataset(predictions, model_name):
    plt.figure()
    plt.plot([0, 1], [0, 1], ls="--", color='grey')

    fpr, tpr, _ = metrics.roc_curve(predictions["click"].values, predictions["p_ctr_all"].values)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC(all): {:0.4f}'.format(roc_auc))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("{}: ROC".format(model_name))
    plt.legend(loc="lower right")

    return roc_auc

def plot_roc_auc_per_advertiser(predictions, model_name):
    scores = {}
    grouped = predictions.groupby("advertiser")
    n_advertisers = len(grouped)

    for advertiser, group in grouped:
        plt.figure()
        plt.plot([0, 1], [0, 1], ls="--", color='grey')

        fpr_advertiser_ds_model, tpr_advertiser_ds_model, _ = metrics.roc_curve(group["click"].values, group["p_ctr_adv"].values)
        roc_auc_advertiser_ds_model = metrics.auc(fpr_advertiser_ds_model, tpr_advertiser_ds_model)
        plt.plot(fpr_advertiser_ds_model, tpr_advertiser_ds_model, label='ROC advertiser model: {:0.4f}'.format(roc_auc_advertiser_ds_model))

        fpr_combined_ds_model, tpr_combined_ds_model, _ = metrics.roc_curve(group["click"].values, group["p_ctr_all"].values)
        roc_auc_combined_ds_model = metrics.auc(fpr_combined_ds_model, tpr_combined_ds_model)
        plt.plot(fpr_combined_ds_model, tpr_combined_ds_model, label='ROC combined model: {:0.4f}'.format(roc_auc_combined_ds_model))

        scores.update({advertiser: {"combined": roc_auc_combined_ds_model, "advertiser": roc_auc_advertiser_ds_model}})

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("{}: ROC({})".format(model_name, advertiser))
        plt.legend(loc="lower right")

    return scores

def plot_confusion_matrix(predictions, model_name, normalize=True):
    cm_adv = metrics.confusion_matrix(predictions["click"], predictions["p_click_adv"])
    cm_all = metrics.confusion_matrix(predictions["click"], predictions["p_click_all"])

    if normalize:
        cm_adv = cm_adv.astype('float') / cm_adv.sum(axis=1)[:, np.newaxis]
        cm_all = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]

    title_adv = "Confusion Matrix ADV - {}".format(model_name)
    title_all = "Confusion Matrix ALL - {}".format(model_name)
    for title, cm in zip([title_adv, title_all], [cm_adv, cm_all]):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("{}{}".format("Normalised " if normalize else "", title))
        classes = ["no click", "click"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')


def plot_bidding_strategy_results(strategy_results_df, max_clicks, title):
    plt.figure()
    for name, group in strategy_results_df.groupby("name"):
        if "per_advertiser" not in group or group["per_advertiser"].isnull().any():
            plt.scatter(group["won_clicks"], group["total_spent"], marker=".", label=name)
        else:
            for sub_name, sub_group in group.groupby("per_advertiser"):
                plt.scatter(group["won_clicks"], group["total_spent"], marker=".", label="({}, {})".format(name, sub_name))

    plt.scatter(max_clicks, 0, marker="x", color="grey", label="optimum")

    if not strategy_results_df["score"].isnull().any():
        best = strategy_results_df.ix[strategy_results_df['score'].idxmax()]
        plt.scatter(best["won_clicks"], best["total_spent"], marker="x", color="red", label="best")

    plt.title(title)
    plt.xlabel("clicks")
    plt.ylabel("total cost")
    plt.legend()

