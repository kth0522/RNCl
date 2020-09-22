import torch
from torch import nn
from itertools import combinations
from torchsummary import summary
from utils import TensorNode, PriorityTensorQueue


def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class classifier32(nn.Module):
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

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        if num_classes == 10:
            self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        elif num_classes == 200 or num_classes == 20:
            self.fc1 = nn.Linear(128 * 8 * 8, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x = x.view(batch_size, -1)
        if return_features:
            return x

        x = self.fc1(x)
        return x


class custom_network_1(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_row = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_col = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_other = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        row = [(0, 1), (2, 3)]
        col = [(0, 2), (1, 3)]
        other = [(0, 3), (1, 2)]

        x1 = x[:, :, 0:2, 0:2]
        x1 = x1.reshape(-1, 128 * 2 * 2)
        x_list.append(x1)

        x2 = x[:, :, 0:2, 2:4]
        x2 = x2.reshape(-1, 128 * 2 * 2)
        x_list.append(x2)

        x3 = x[:, :, 2:4, 0:2]
        x3 = x3.reshape(-1, 128 * 2 * 2)
        x_list.append(x3)

        x4 = x[:, :, 2:4, 2:4]
        x4 = x4.reshape(-1, 128 * 2 * 2)
        x_list.append(x4)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in row:
            cat_tensor += self.g_theta_row(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in col:
            cat_tensor += self.g_theta_col(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in other:
            cat_tensor += self.g_theta_other(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_2(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        idx = [0, 1, 2, 3]

        pairs = list(combinations(idx, 2))

        x1 = x[:, :, 0:2, 0:2]
        x1 = x1.reshape(-1, 128 * 2 * 2)
        x_list.append(x1)

        x2 = x[:, :, 0:2, 2:4]
        x2 = x2.reshape(-1, 128 * 2 * 2)
        x_list.append(x2)

        x3 = x[:, :, 2:4, 0:2]
        x3 = x3.reshape(-1, 128 * 2 * 2)
        x_list.append(x3)

        x4 = x[:, :, 2:4, 2:4]
        x4 = x4.reshape(-1, 128 * 2 * 2)
        x_list.append(x4)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in pairs:
            cat_tensor += self.g_theta(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_3(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        idx = [0, 1, 2, 3]

        pairs = list(combinations(idx, 2))

        x1 = x[:, :, 0:3, 0:3]
        x1 = x1.reshape(-1, 128 * 3 * 3)
        x_list.append(x1)

        x2 = x[:, :, 0:3, 1:4]
        x2 = x2.reshape(-1, 128 * 3 * 3)
        x_list.append(x2)

        x3 = x[:, :, 1:4, 0:3]
        x3 = x3.reshape(-1, 128 * 3 * 3)
        x_list.append(x3)

        x4 = x[:, :, 1:4, 1:4]
        x4 = x4.reshape(-1, 128 * 3 * 3)
        x_list.append(x4)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in pairs:
            cat_tensor += self.g_theta(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_4(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        idx = [0, 1, 2, 3]

        pairs = list(combinations(idx, 2))

        x1 = x[:, :, 0:4, 0:4]
        x1 = x1.reshape(-1, 128 * 4 * 4)
        x_list.append(x1)

        x2 = x[:, :, 0:4, 4:8]
        x2 = x2.reshape(-1, 128 * 4 * 4)
        x_list.append(x2)

        x3 = x[:, :, 4:8, 0:4]
        x3 = x3.reshape(-1, 128 * 4 * 4)
        x_list.append(x3)

        x4 = x[:, :, 4:8, 4:8]
        x4 = x4.reshape(-1, 128 * 4 * 4)
        x_list.append(x4)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in pairs:
            cat_tensor += self.g_theta(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_5(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_row = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_col = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_other = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        row = [(0, 1), (2, 3)]
        col = [(0, 2), (1, 3)]
        other = [(0, 3), (1, 2)]

        x1 = x[:, :, 0:6, 0:6]
        x1 = x1.reshape(-1, 128 * 6 * 6)
        x_list.append(x1)

        x2 = x[:, :, 0:6, 2:8]
        x2 = x2.reshape(-1, 128 * 6 * 6)
        x_list.append(x2)

        x3 = x[:, :, 2:8, 0:6]
        x3 = x3.reshape(-1, 128 * 6 * 6)
        x_list.append(x3)

        x4 = x[:, :, 2:8, 2:8]
        x4 = x4.reshape(-1, 128 * 6 * 6)
        x_list.append(x4)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in row:
            cat_tensor += self.g_theta_row(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in col:
            cat_tensor += self.g_theta_col(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in other:
            cat_tensor += self.g_theta_other(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_6(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x = self.pool1(x)

        x_list = []

        idx = [0, 1, 2, 3]

        pairs = list(combinations(idx, 2))

        x1 = x[:, :, 0:2, 0:2]
        x1 = x1.reshape(-1, 128 * 2 * 2)
        x_list.append(x1)

        x2 = x[:, :, 0:2, 2:4]
        x2 = x2.reshape(-1, 128 * 2 * 2)
        x_list.append(x2)

        x3 = x[:, :, 2:4, 0:2]
        x3 = x3.reshape(-1, 128 * 2 * 2)
        x_list.append(x3)

        x4 = x[:, :, 2:4, 2:4]
        x4 = x4.reshape(-1, 128 * 2 * 2)
        x_list.append(x4)

        cat_tensor = torch.zeros([2, 256]).cuda()

        for pair in pairs:
            cat_tensor += self.g_theta(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_7(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.pool1 = nn.AvgPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x = self.pool1(x)

        x_list = []

        idx = [0, 1, 2, 3]

        pairs = list(combinations(idx, 2))

        x1 = x[:, :, 0:2, 0:2]
        x1 = x1.reshape(-1, 128 * 2 * 2)
        x_list.append(x1)

        x2 = x[:, :, 0:2, 2:4]
        x2 = x2.reshape(-1, 128 * 2 * 2)
        x_list.append(x2)

        x3 = x[:, :, 2:4, 0:2]
        x3 = x3.reshape(-1, 128 * 2 * 2)
        x_list.append(x3)

        x4 = x[:, :, 2:4, 2:4]
        x4 = x4.reshape(-1, 128 * 2 * 2)
        x_list.append(x4)

        cat_tensor = torch.zeros([2, 256]).cuda()

        for pair in pairs:
            cat_tensor += self.g_theta(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_8(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.pool1 = nn.AvgPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_row = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_col = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_other = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x = self.pool1(x)

        x_list = []

        row = [(0, 1), (2, 3)]
        col = [(0, 2), (1, 3)]
        other = [(0, 3), (1, 2)]

        x1 = x[:, :, 0:3, 0:3]
        x1 = x1.reshape(-1, 128 * 3 * 3)
        x_list.append(x1)

        x2 = x[:, :, 0:3, 1:4]
        x2 = x2.reshape(-1, 128 * 3 * 3)
        x_list.append(x2)

        x3 = x[:, :, 1:4, 0:3]
        x3 = x3.reshape(-1, 128 * 3 * 3)
        x_list.append(x3)

        x4 = x[:, :, 1:4, 1:4]
        x4 = x4.reshape(-1, 128 * 3 * 3)
        x_list.append(x4)

        cat_tensor = torch.zeros([2, 256]).cuda()

        for pair in row:
            cat_tensor += self.g_theta_row(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in col:
            cat_tensor += self.g_theta_col(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in other:
            cat_tensor += self.g_theta_other(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_9(nn.Module):
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

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_row = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_col = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_other = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        row = [(0, 1), (2, 3)]
        col = [(0, 2), (1, 3)]
        other = [(0, 3), (1, 2)]

        x1 = x[:, :, 0:6, 0:6]
        x1 = x1.reshape(-1, 128 * 6 * 6)
        x_list.append(x1)

        x2 = x[:, :, 0:6, 2:8]
        x2 = x2.reshape(-1, 128 * 6 * 6)
        x_list.append(x2)

        x3 = x[:, :, 2:8, 0:6]
        x3 = x3.reshape(-1, 128 * 6 * 6)
        x_list.append(x3)

        x4 = x[:, :, 2:8, 2:8]
        x4 = x4.reshape(-1, 128 * 6 * 6)
        x_list.append(x4)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in row:
            cat_tensor += self.g_theta_row(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in col:
            cat_tensor += self.g_theta_col(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in other:
            cat_tensor += self.g_theta_other(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_10(nn.Module):
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

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_list = []

        idx = range(16)

        pairs = list(combinations(idx, 2))

        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                x_temp = x[:, :, i:i + 2, j:j + 2]
                x_temp = x_temp.reshape(-1, 128 * 2 * 2)
                x_list.append(x_temp)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in pairs:
            cat_tensor += self.g_theta(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_11(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_small = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_big = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_small_list = []
        x_big_list = []

        small_idx = range(16)
        big_idx = range(4)

        pairs_1 = list(combinations(small_idx, 2))
        pairs_2 = list(combinations(big_idx, 2))

        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                x_temp = x[:, :, i:i + 2, j:j + 2]
                x_temp = x_temp.reshape(-1, 128 * 2 * 2)
                x_small_list.append(x_temp)

        for i in range(0, 8, 4):
            for j in range(0, 8, 4):
                x_temp = x[:, :, i:i + 4, j:j + 4]
                x_temp = x_temp.reshape(-1, 128 * 4 * 4)
                x_big_list.append(x_temp)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in pairs_1:
            cat_tensor += self.g_theta_small(torch.cat([x_small_list[pair[0]], x_small_list[pair[1]]], dim=1))

        for pair in pairs_2:
            cat_tensor += self.g_theta_big(torch.cat([x_big_list[pair[0]], x_big_list[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_12(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_1 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x_list_1 = []
        x_list_2 = []

        idx_1 = range(16)
        idx_2 = range(4)

        pairs_1 = list(combinations(idx_1, 2))
        pairs_2 = list(combinations(idx_2, 2))

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

        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                x_temp = x[:, :, i:i + 4, j:j + 4]
                x_temp = x_temp.reshape(-1, 128 * 4 * 4)
                x_list_1.append(x_temp)

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

        for i in range(0, 8, 4):
            for j in range(0, 8, 4):
                x_temp = x[:, :, i:i + 4, j:j + 4]
                x_temp = x_temp.reshape(-1, 128 * 4 * 4)
                x_list_2.append(x_temp)

        cat_tensor = torch.zeros([64, 256]).cuda()

        for pair in pairs_1:
            cat_tensor += self.g_theta_1(torch.cat([x_list_1[pair[0]], x_list_1[pair[1]]], dim=1))

        for pair in pairs_2:
            cat_tensor += self.g_theta_2(torch.cat([x_list_2[pair[0]], x_list_2[pair[1]]], dim=1))

        x = self.f_phi(cat_tensor)

        return x


class custom_network_13(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_small = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_big = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_small_list = []

        small_idx = range(16)

        pairs_1 = list(combinations(small_idx, 2))


        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                x_temp = x[:, :, i:i + 2, j:j + 2]
                x_temp = x_temp.reshape(-1, 128 * 2 * 2)
                x_small_list.append(x_temp)


        cat_tensor = torch.zeros([64, 256]).cuda()
        q = PriorityTensorQueue()

        for pair in pairs_1:
            temp_tensor = self.g_theta_small(torch.cat([x_small_list[pair[0]], x_small_list[pair[1]]], dim=1))
            q.insert(TensorNode(temp_tensor))

        size = q.size()
        for i in range(size//10):
            cat_tensor += q.queue[size-1-i].tensor


        x = self.f_phi(cat_tensor)

        return x

class custom_network_14(nn.Module):
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

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_small = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_big = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_small_list = []

        small_idx = range(16)

        pairs_1 = list(combinations(small_idx, 2))


        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                x_temp = x[:, :, i:i + 2, j:j + 2]
                x_temp = x_temp.reshape(-1, 128 * 2 * 2)
                x_small_list.append(x_temp)


        cat_tensor = torch.zeros([64, 256]).cuda()
        q = PriorityTensorQueue()

        for pair in pairs_1:
            temp_tensor = self.g_theta_small(torch.cat([x_small_list[pair[0]], x_small_list[pair[1]]], dim=1))
            q.insert(TensorNode(temp_tensor))

        size = q.size()
        for i in range(size//2):
            cat_tensor += q.queue[size-1-i].tensor


        x = self.f_phi(cat_tensor)

        return x


class custom_network_15(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_1 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
        batch_size = len(x)

        x_list_1 = []
        x_list_2 = []

        idx_1 = range(16)
        idx_2 = range(4)

        pairs_1 = list(combinations(idx_1, 2))
        pairs_2 = list(combinations(idx_2, 2))

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

        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                x_temp = x[:, :, i:i + 4, j:j + 4]
                x_temp = x_temp.reshape(-1, 128 * 4 * 4)
                x_list_1.append(x_temp)

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

        for i in range(0, 8, 4):
            for j in range(0, 8, 4):
                x_temp = x[:, :, i:i + 4, j:j + 4]
                x_temp = x_temp.reshape(-1, 128 * 4 * 4)
                x_list_2.append(x_temp)

        cat_tensor = torch.zeros([64, 256]).cuda()

        q = PriorityTensorQueue()

        for pair in pairs_1:
            temp_tensor = self.g_theta_1(torch.cat([x_list_1[pair[0]], x_list_1[pair[1]]], dim=1))
            q.insert(TensorNode(temp_tensor))

        for pair in pairs_2:
            temp_tensor = self.g_theta_2(torch.cat([x_list_2[pair[0]], x_list_2[pair[1]]], dim=1))
            q.insert(TensorNode(temp_tensor))

        size = q.size()
        for i in range(size // 2):
            cat_tensor += q.queue[size - 1 - i].tensor

        x = self.f_phi(cat_tensor)

        return x

class custom_network_16(nn.Module):
    def __init__(self, latent_size=100, num_classes=10, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, num_classes)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.g_theta_small = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_big = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 200)
        )

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_features=False):
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

        x_small_list = []
        x_big_list = []

        small_idx = range(16)
        big_idx = range(4)

        pairs_1 = list(combinations(small_idx, 2))
        pairs_2 = list(combinations(big_idx, 2))

        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                x_temp = x[:, :, i:i + 2, j:j + 2]
                x_temp = x_temp.reshape(-1, 128 * 2 * 2)
                x_small_list.append(x_temp)

        for i in range(0, 8, 4):
            for j in range(0, 8, 4):
                x_temp = x[:, :, i:i + 4, j:j + 4]
                x_temp = x_temp.reshape(-1, 128 * 4 * 4)
                x_big_list.append(x_temp)

        cat_tensor = torch.zeros([64, 256]).cuda()

        q = PriorityTensorQueue()


        for pair in pairs_1:
            temp_tensor = self.g_theta_small(torch.cat([x_small_list[pair[0]], x_small_list[pair[1]]], dim=1))
            q.insert(TensorNode(temp_tensor))

        for pair in pairs_2:
            temp_tensor = self.g_theta_big(torch.cat([x_big_list[pair[0]], x_big_list[pair[1]]], dim=1))
            q.insert(TensorNode(temp_tensor))

        size = q.size()
        for i in range(size // 2):
            cat_tensor += q.queue[size - 1 - i].tensor

        x = self.f_phi(cat_tensor)

        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = classifier32()
    net.to(device)
    summary(net, (3, 32, 32))
