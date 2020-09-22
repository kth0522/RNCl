import torch
from torch.autograd import Variable
import torch.nn.functional as F

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


class_num = 6

def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    df = pd.DataFrame({'x': x, "y": y})
    plot = df.plot(x='x', y='y')

    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')

    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot

def plot_roc(y_true, y_score, title="Receciver Operating Characteristic"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)

    return auc_score, plot

def openset_softmax_confidence(dataloader, netC):
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = Variable(images)
            images = images.cuda()
            net_y = netC(images)
            preds = F.softmax(net_y, dim=1)
            openset_scores.extend(preds.max(dim=1)[0].data.cpu().numpy())
    return -np.array(openset_scores)

def evalute_classifier(net, dataloader):
    net.eval()
    classification_closed_correct = 0
    classification_total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = Variable(images)
            images = images.cuda()
            net_y = net(images)
            class_predictions = F.softmax(net_y, dim=1)
            labels = torch.argmax(labels, dim=1)
            _, predicted = class_predictions.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)

    return float(classification_closed_correct) / classification_total

def evaluate_openset(net, dataloader_on, dataloader_off):
    net.eval()

    d_scores_on = openset_softmax_confidence(dataloader_on, net)
    d_scores_off = openset_softmax_confidence(dataloader_off, net)

    y_true = np.array([0] * len(d_scores_on) + [1] * len(d_scores_off))
    y_discriminator = np.concatenate([d_scores_on, d_scores_off])

    auc_d, plot_d = plot_roc(y_true, y_discriminator, 'Discriminator ROC vs {}'.format(dataloader_off.dsf.name))

    return auc_d