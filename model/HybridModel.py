import torch
import math
from torch import nn
import numpy as np
from model.resflow import ResidualFlow
import torch.nn.functional as F

input_size = (64, 128, 4, 4)
n_classes = 20


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def compute_loss(x, model, beta=1.0):
    bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
    logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

    nvals = 32

    logpu = torch.zeros(x.shape[0], 1).to(x)

    z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
    z, delta_logp = z_logp

    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

    logpx = logpz - beta * delta_logp - np.log(nvals) * (64 * 64 * 3) - logpu

    bits_per_dim = -torch.mean(logpx) / (64 * 64 * 3) / np.log(2)

    logpz = torch.mean(logpz).detach()
    delta_logp = torch.mean(-delta_logp).detach()

    return bits_per_dim, logits_tensor, logpz, delta_logp


class encoder32(nn.Module):
    def __init__(self, latent_size=100, num_classes=20, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        # Shortcut out of the network at 8x8
        self.conv_out_6 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        # Shortcut out of the network at 4x4
        self.conv_out_9 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.conv10 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.conv_out_10 = nn.Conv2d(128, latent_size, 3, 1, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, latent_size)

        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)
        self.dr4 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        batch_size = len(x)

        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr4(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = nn.LeakyReLU(0.2)(x)

        # x = x.view(batch_size, -1)
        # x = self.fc1(x)

        return x


class classifier32(nn.Module):
    def __init__(self, latent_size=100, num_classes=20, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()

        if num_classes == 10:
            self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        elif num_classes == 200 or num_classes == 20:
            self.fc1 = nn.Linear(128 * 4 * 4, num_classes)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        batch_size = len(x)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, encoder, classifier, flow):
        super().__init__()
        self.beta = 1

        self.encoder = encoder
        self.flow = flow
        self.classifier = classifier

        self.cuda()

    def forward(self, x):
        x = self.encoder(x)
        cls_logits = self.classifier(x)
        cls_preds = F.softmax(cls_logits, dim=1)
        bpd, flow_logits, _, _ = compute_loss(x, self.flow, beta=self.beta)
        s = 0.31
        taus = F.softmax(flow_logits, dim=1).min(1)
        zero_logit = torch.zeros([20]).cuda()

        for i in range(cls_preds.shape[0]):
            logit = cls_preds[i]
            tau = taus.values[i]
            predicted = torch.max(logit)
            if predicted < tau + s:
                cls_preds[i] = zero_logit

        return cls_preds
