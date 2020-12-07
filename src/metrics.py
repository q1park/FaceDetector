import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from sklearn.metrics import precision_score, balanced_accuracy_score, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

def compute_metrics(preds, labels):
    Pr, Re, F1 = [x[1] for x in precision_recall_fscore_support(labels, preds)[:3]]
    BA = balanced_accuracy_score(labels, preds)
    return Pr, Re, F1, BA

def make_ROC(output, labels):
    lr_probs = F.softmax(output, dim=1)[:,1].detach().numpy()
    ns_probs = [0 for _ in range(len(labels))]
    # calculate scores
    ns_roc_auc = roc_auc_score(labels, ns_probs)
    lr_roc_auc = roc_auc_score(labels, lr_probs)
    print('ROC AUC=%.3f'%(lr_roc_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(labels, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC=%.3f'%(lr_roc_auc))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def make_PR(output, labels):
    lr_probs = F.softmax(output, dim=1)[:,1].detach().numpy()
    lr_precision, lr_recall, _ = precision_recall_curve(labels, lr_probs)
    # calculate scores
    lr_auc = auc(lr_recall, lr_precision)
    print('PR AUC=%.3f' % (lr_auc))
    # plot the precision-recall curves
    no_skill = len(labels[labels==1]) / len(labels)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR AUC=%.3f' % (lr_auc))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    