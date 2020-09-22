import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicRN(nn.Module):
    def __init__(self):
        super(BasicRN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.g_theta_row = nn.Sequential(
            nn.Linear(16 * 3 * 3 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_col = nn.Sequential(
            nn.Linear(16 * 3 * 3 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.g_theta_other = nn.Sequential(
            nn.Linear(16 * 3 * 3 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f_pi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x_list = []
        idx = [0, 1, 2, 3]
        pairs = combinations(idx, 2)

        row = [(0, 1), (2, 3)]
        col = [(0, 2), (1, 3)]
        other = [(0, 3), (1, 2)]

        x1 = x[:, :, 0:3, 0:3]
        x1 = x1.reshape(-1, 16 * 3 * 3)

        x_list.append(x1)

        x2 = x[:, :, 2:5, 0:3]
        x2 = x2.reshape(-1, 16 * 3 * 3)

        x_list.append(x2)

        x3 = x[:, :, 0:3, 2:5]
        x3 = x3.reshape(-1, 16 * 3 * 3)

        x_list.append(x3)

        x4 = x[:, :, 2:5, 2:5]
        x4 = x4.reshape(-1, 16 * 3 * 3)

        x_list.append(x4)

        cat_tensor = torch.zeros([32, 256]).cuda()

        for pair in row:
            cat_tensor += self.g_theta_row(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in col:
            cat_tensor += self.g_theta_col(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))
        for pair in other:
            cat_tensor += self.g_theta_other(torch.cat([x_list[pair[0]], x_list[pair[1]]], dim=1))


        x = self.f_pi(cat_tensor)

        return x
