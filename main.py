import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

import numpy as np
import datetime
import argparse
import os

from tensorboardX import SummaryWriter

from dataloader import CustomDataloader, FlexibleCustomDataloader
from model.ResNet import resnet18
from model.CustomResNet import custom_resnet18
from model.CustomResNet_v2 import custom_resnet18_v2
from model.CustomResNet_v3 import CustomResNet18_v3
from model.OSRCI_Network import classifier32, custom_network_9, custom_network_13, custom_network_15, custom_network_10, custom_network_14, custom_network_16
from model.cifar10_ResNet import cifar10_ResNet18, cifar10_ResNet34, cifar10_ResNet50
from model.BaseNet import BaseNet, BasicRN
from evalutation import evalute_classifier, evaluate_openset
from utils import select_n_random, plot_classes_preds
from torchsummary import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--fold', '-f', type=int, default=0, help='which fold you gonna train with')
    args = parser.parse_args()

    DATASET = 'tiny_imagenet-known-20-split'
    # MODEL = 'custom_classifier_9'
    MODEL = 'classifier32'
    fold_num = args.fold
    batch_size = args.batch_size
    is_train = True
    is_write = True

    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')
    runs = 'runs/{}-{}{}-{}'.format(MODEL, DATASET, fold_num, start_time)
    if is_write:
        writer = SummaryWriter(runs)


    closed_trainloader = FlexibleCustomDataloader(fold='train', batch_size=batch_size, dataset='./data/{}{}a.dataset'.format(DATASET, fold_num))
    closed_testloader = FlexibleCustomDataloader(fold='test', batch_size=batch_size, dataset='./data/{}{}a.dataset'.format(DATASET, fold_num))

    open_trainloader = FlexibleCustomDataloader(fold='train', batch_size=batch_size, dataset='./data/{}{}b.dataset'.format(DATASET, fold_num))
    open_testloader = FlexibleCustomDataloader(fold='test', batch_size=batch_size, dataset='./data/{}{}b.dataset'.format(DATASET, fold_num))

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    PATH = '{}/{}{}_custom_network_15'.format(runs, DATASET, fold_num)
    if is_train:
        net = classifier32()
        net.to(device)
        net.train()

        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300, 350, 400, 450], gamma=0.1)

        running_loss = 0.0
        for epoch in range(30):
            for i, (images, labels) in enumerate(closed_trainloader, 0):
                images = Variable(images)
                images = images.cuda()

                labels = Variable(labels)



                optimizer.zero_grad()

                # writer.add_graph(net, images)
                outputs = net(images)

                labels = torch.argmax(labels, dim=1)

                # writer.add_embedding(outputs, metadata=class_labels, label_img=images.unsqueeze(1))
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                #scheduler.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    if is_write:
                        writer.add_scalar('training loss',
                                          running_loss / 100,
                                          epoch * len(closed_trainloader) + i)
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')
                    print(current_time)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))

                    # writer.add_figure('predictions vs. actuals',
                    #                   plot_classes_preds(net, images, labels))
                    running_loss = 0.0
            if epoch % 50 == 49:
                torch.save(net.state_dict(), "{}_{}.pth".format(PATH, epoch+1))
            torch.save(net.state_dict(), "{}_latest.pth".format(PATH))

    test_net = classifier32()
    # PATH_1 = "/home/taehokim/PycharmProjects/RNCl/runs/custom_classifier_14-tiny_imagenet-known-20-split0-2020-08-26_08-25-01-AM"
    # PATH = '{}/{}{}_custom_classifier_13'.format(PATH_1, DATASET, fold_num)
    test_net.load_state_dict(torch.load("{}_latest.pth".format(PATH)))
    test_net.to(device)

    closed_acc = evalute_classifier(test_net, closed_testloader)
    print("closed-set accuracy: ", closed_acc)
    auc_d = evaluate_openset(test_net, closed_testloader, open_testloader)
    print("auc discriminator: ", auc_d)

    result_file = '{}/{}{}.txt'.format(runs, DATASET, fold_num)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')

    if os.path.exists(result_file):
        f = open(result_file, 'a')
        f.write(current_time+"\n")
        f.write("{}{} \n".format(DATASET, fold_num))
        f.write("{} epoch".format(i))
        f.write("close-set accuracy: {} \n".format(closed_acc))
        f.write("AUROC: {} \n".format(auc_d))
        f.close()
    else:
        f = open(result_file, 'w')
        f.write(current_time+"\n")
        f.write("{}{} \n".format(DATASET, fold_num))
        f.write("{} epoch".format(i))
        f.write("close-set accuracy: {} \n".format(closed_acc))
        f.write("AUROC: {} \n".format(auc_d))
        f.close()


    # for i in range(50, 550, 50):
    #     test_net = custom_network_15()
    #     # PATH_1 = "/home/taehokim/PycharmProjects/RNCl/runs/custom_classifier_14-tiny_imagenet-known-20-split0-2020-08-26_08-25-01-AM"
    #     # PATH = '{}/{}{}_custom_classifier_13'.format(PATH_1, DATASET, fold_num)
    #     test_net.load_state_dict(torch.load("{}_{}.pth".format(PATH, i)))
    #     test_net.to(device)
    #
    #     closed_acc = evalute_classifier(test_net, closed_testloader)
    #     print("closed-set accuracy: ", closed_acc)
    #     auc_d = evaluate_openset(test_net, closed_testloader, open_testloader)
    #     print("auc discriminator: ", auc_d)
    #
    #
    #     result_file = '{}/{}{}.txt'.format(runs, DATASET, fold_num)
    #
    #     current_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')
    #
    #     if os.path.exists(result_file):
    #         f = open(result_file, 'a')
    #         f.write(current_time+"\n")
    #         f.write("{}{} \n".format(DATASET, fold_num))
    #         f.write("{} epoch".format(i))
    #         f.write("close-set accuracy: {} \n".format(closed_acc))
    #         f.write("AUROC: {} \n".format(auc_d))
    #         f.close()
    #     else:
    #         f = open(result_file, 'w')
    #         f.write(current_time+"\n")
    #         f.write("{}{} \n".format(DATASET, fold_num))
    #         f.write("{} epoch".format(i))
    #         f.write("close-set accuracy: {} \n".format(closed_acc))
    #         f.write("AUROC: {} \n".format(auc_d))
    #         f.close()

    # # correct = 0
    # # total = 0
    # # with torch.no_grad():
    # #     for data in eval_dataloader:
    # #         images, labels = data
    # #         outputs = test_net(images)
    # #         _, predicted = torch.max(outputs.data, 1)
    # #         total += labels.size(0)
    # #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    #
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = test_net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    main()
