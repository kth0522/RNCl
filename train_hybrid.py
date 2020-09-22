import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import gc
import torchvision.transforms as transforms

import numpy as np
import datetime
import argparse
import os
import math
from collections import OrderedDict

from tensorboardX import SummaryWriter

from model.HybridModel import encoder32, classifier32, HybridModel
from model.resflow import ResidualFlow
import model.layers as layers
import model.layers.base as base_layers
import torch.nn.functional as F
from dataloader import CustomDataloader, FlexibleCustomDataloader
from evalutation import evalute_classifier, evaluate_openset
from utils import RunningAverageMeter, ExponentialMovingAverage
import utils

input_size = (64, 128, 4, 4)
n_classes = 20

def parallelize(model):
    return torch.nn.DataParallel(model)

def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.iResBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--fold', '-f', type=int, default=0, help='which fold you gonna train with')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--multi-eval', type=bool, default=False)
    parser.add_argument('--update-freq', type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.seed is None:
        args.seed = np.random.randint(100000)

    print("seed: {}".format(args.seed))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    DATASET = 'tiny_imagenet-known-20-split'
    # MODEL = 'custom_classifier_9'
    MODEL = 'hybrid'
    fold_num = args.fold
    batch_size = args.batch_size
    is_train = False
    is_write = False

    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')
    runs = 'runs/{}-{}{}-{}'.format(MODEL, DATASET, fold_num, start_time)
    if is_write:
        writer = SummaryWriter(runs)

    closed_trainloader = FlexibleCustomDataloader(fold='train', batch_size=batch_size,
                                                  dataset='./data/{}{}a.dataset'.format(DATASET, fold_num))
    closed_testloader = FlexibleCustomDataloader(fold='test', batch_size=batch_size,
                                                 dataset='./data/{}{}a.dataset'.format(DATASET, fold_num))

    open_trainloader = FlexibleCustomDataloader(fold='train', batch_size=batch_size,
                                                dataset='./data/{}{}b.dataset'.format(DATASET, fold_num))
    open_testloader = FlexibleCustomDataloader(fold='test', batch_size=batch_size,
                                               dataset='./data/{}{}b.dataset'.format(DATASET, fold_num))

    batch_time = RunningAverageMeter(0.97)
    bpd_meter = RunningAverageMeter(0.97)
    logpz_meter = RunningAverageMeter(0.97)
    deltalogp_meter = RunningAverageMeter(0.97)
    firmom_meter = RunningAverageMeter(0.97)
    secmom_meter = RunningAverageMeter(0.97)
    gnorm_meter = RunningAverageMeter(0.97)
    ce_meter = RunningAverageMeter(0.97)

    PATH = '{}/{}{}_hybrid'.format(runs, DATASET, fold_num)
    if is_train:
        encoder = encoder32()
        encoder.to(device)
        encoder.train()

        flow = ResidualFlow(n_classes=20,
                            input_size=(64, 128, 4, 4),
                            n_blocks=[32, 32, 32],
                            intermediate_dim=512,
                            factor_out=False,
                            quadratic=False,
                            init_layer=None,
                            actnorm=True,
                            fc_actnorm=False,
                            dropout=0,
                            fc=False,
                            coeff=0.98,
                            vnorms='2222',
                            n_lipschitz_iters=None,
                            sn_atol=1e-3, sn_rtol=1e-3,
                            n_power_series=None,
                            n_dist='poisson',
                            n_samples=1,
                            kernels='3-1-3',
                            activation_fn='swish',
                            fc_end=True,
                            n_exact_terms=2,
                            preact=True,
                            neumann_grad=True,
                            grad_in_forward=False,
                            first_resblock=True,
                            learn_p=False,
                            classification='hybrid',
                            classification_hdim=256,
                            block_type='resblock')
        flow.to(device)
        flow.train()

        classifier = classifier32()
        classifier.to(device)
        classifier.train()

        ema = ExponentialMovingAverage(flow)

        flow.train()

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
        optimizer_2 = optim.Adam(flow.parameters(), lr=0.0001)
        optimizer_3 = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
        # optimizer_3 = optim.Adam(classifier.parameters(), lr=0.0001)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        #                                                  milestones=[50, 100, 150, 200, 250, 300, 350, 400, 450],
        #                                                  gamma=0.1)
        beta = 1
        running_loss = 0.0
        running_bpd = 0.0
        running_cls = 0.0
        best_loss = 1000
        tau = 100000
        for epoch in range(600):
            for i, (images, labels) in enumerate(closed_trainloader, 0):
                global_itr = epoch * len(closed_trainloader) + i
                images = Variable(images)
                images = images.cuda()

                labels = Variable(labels)

                # writer.add_graph(net, images)
                outputs = encoder(images)

                bpd, logits, logpz, neg_delta_logp = compute_loss(outputs, flow, beta=beta)
                cls_outputs = classifier(outputs)


                labels = torch.argmax(labels, dim=1)
                cls_loss = criterion(cls_outputs, labels)

                firmom, secmom = estimator_moments(flow)

                bpd_meter.update(bpd.item())
                logpz_meter.update(logpz.item())
                deltalogp_meter.update(neg_delta_logp.item())
                firmom_meter.update(firmom)
                secmom_meter.update(secmom)

                loss = bpd + cls_loss
                #
                # loss.backward()
                #
                # labels = torch.argmax(labels, dim=1)
                #
                # # writer.add_embedding(outputs, metadata=class_labels, label_img=images.unsqueeze(1))
                # loss = criterion(outputs, labels)
                loss.backward()

                if global_itr % args.update_freq == args.update_freq - 1:
                    if args.update_freq > 1:
                        with torch.no_grad():
                            for p in flow.parameters():
                                if p.grad is not None:
                                    p.grad /= args.update_freq

                    grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(flow.parameters(), 1.)

                    optimizer.step()
                    optimizer_2.step()
                    optimizer_3.step()

                    optimizer.zero_grad()
                    optimizer_2.zero_grad()
                    optimizer_3.zero_grad()

                    update_lipschitz(flow)
                    ema.apply()
                    gnorm_meter.update(grad_norm)

                running_bpd += bpd.item()
                running_cls += cls_loss.item()
                running_loss += loss.item()




                if i % 100 == 99:
                    if is_write:
                        writer.add_scalar('bits per dimension',
                                         running_bpd / 100,
                                         global_itr)
                        writer.add_scalar('classification loss',
                                          running_cls / 100,
                                          global_itr)
                        writer.add_scalar('total loss',
                                          running_loss / 100,
                                          global_itr)
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')
                    print(current_time)
                    print('[%d, %5d] bpd: %.3f, cls_loss: %.3f, total_loss: %.3f' % (
                    epoch + 1, i + 1, running_bpd / 100, running_cls / 100, running_loss / 100))
                    if epoch > 1 and running_loss / 100 < best_loss:
                        best_loss = running_loss / 100
                        print("best loss updated! :", best_loss)
                        torch.save({
                            'state_dict': flow.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'args': args,
                            'ema': ema,
                        }, "{}_flow_best.pth".format(PATH))

                        torch.save(encoder.state_dict(), "{}_encoder_best.pth".format(PATH))
                        torch.save(classifier.state_dict(), "{}_classifier_best.pth".format(PATH))

                    # writer.add_figure('predictions vs. actuals',
                    #                   plot_classes_preds(net, images, labels))
                    running_loss = 0.0
                    running_bpd = 0.0
                    running_cls = 0.0

                del images
                torch.cuda.empty_cache()
                gc.collect()

            if epoch % 50 == 49:
                torch.save({
                    'state_dict': flow.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'ema': ema,
                }, "{}_flow_{}.pth".format(PATH, epoch+1))

                torch.save(encoder.state_dict(), "{}_encoder_{}.pth".format(PATH, epoch + 1))
                torch.save(classifier.state_dict(), "{}_classifier_{}.pth".format(PATH, epoch + 1))

    PATH_1 = "/home/taehokim/PycharmProjects/RNCl/runs/hybrid-tiny_imagenet-known-20-split0-2020-09-21_05-49-50-PM"
    PATH = "{}/{}{}_hybrid".format(PATH_1, DATASET, fold_num)

    if args.multi_eval:
        for i in range(50, 550, 50):
            test_encoder = encoder32()
            test_encoder.to(device)
            test_encoder.load_state_dict(torch.load("{}_encoder_{}.pth".format(PATH, i)))
            # state_dict = torch.load("{}_encoder_{}.pth".format(PATH, i))
            # # create new OrderedDict that does not contain `module.`
            #
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:]  # remove `module.`
            #     new_state_dict[name] = v
            # # load params
            # test_encoder.load_state_dict(new_state_dict)

            test_classifier = classifier32()
            test_classifier.to(device)
            # state_dict = torch.load("{}_classifier_{}.pth".format(PATH, i))
            # # create new OrderedDict that does not contain `module.`
            #
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:]  # remove `module.`
            #     new_state_dict[name] = v
            # # load params
            # test_classifier.load_state_dict(new_state_dict)
            test_classifier.load_state_dict(torch.load("{}_classifier_{}.pth".format(PATH, i)))


            test_flow = ResidualFlow(n_classes=20,
                                     input_size=(64, 128, 4, 4),
                                     n_blocks=[32, 32, 32],
                                     intermediate_dim=512,
                                     factor_out=False,
                                     quadratic=False,
                                     init_layer=None,
                                     actnorm=True,
                                     fc_actnorm=False,
                                     dropout=0,
                                     fc=False,
                                     coeff=0.98,
                                     vnorms='2222',
                                     n_lipschitz_iters=None,
                                     sn_atol=1e-3, sn_rtol=1e-3,
                                     n_power_series=None,
                                     n_dist='poisson',
                                     n_samples=1,
                                     kernels='3-1-3',
                                     activation_fn='swish',
                                     fc_end=True,
                                     n_exact_terms=2,
                                     preact=True,
                                     neumann_grad=True,
                                     grad_in_forward=False,
                                     first_resblock=True,
                                     learn_p=False,
                                     classification='hybrid',
                                     classification_hdim=256,
                                     block_type='resblock')

            test_flow.to(device)

            with torch.no_grad():
                x = torch.rand(1, *input_size[1:]).to(device)
                test_flow(x)
            checkpt = torch.load("{}_flow_{}.pth".format(PATH, i))
            sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
            state = test_flow.state_dict()
            state.update(sd)
            test_flow.load_state_dict(state, strict=True)
            # test_ema.set(checkpt['ema'])

            hybrid = HybridModel(test_encoder, test_classifier, test_flow)

            closed_acc = evalute_classifier(hybrid, closed_testloader)
            print("closed-set accuracy: ", closed_acc)
            auc_d = evaluate_openset(hybrid, closed_testloader, open_testloader)
            print("auc discriminator: ", auc_d)

            result_file = '{}/{}{}.txt'.format(runs, DATASET, fold_num)

            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%I-%M-%S-%p')

            if is_write:
                if os.path.exists(result_file):
                    f = open(result_file, 'a')
                    f.write(current_time + "\n")
                    f.write("seed: {}\n".format(args.seed))
                    f.write("{}{} \n".format(DATASET, fold_num))
                    f.write("{} epoch".format(i))
                    f.write("close-set accuracy: {} \n".format(closed_acc))
                    f.write("AUROC: {} \n".format(auc_d))
                    f.close()
                else:
                    f = open(result_file, 'w')
                    f.write(current_time + "\n")
                    f.write("seed: {}\n".format(args.seed))
                    f.write("{}{} \n".format(DATASET, fold_num))
                    f.write("{} epoch".format(i))
                    f.write("close-set accuracy: {} \n".format(closed_acc))
                    f.write("AUROC: {} \n".format(auc_d))
                    f.close()
    else:
        PATH_1 = "/home/taehokim/PycharmProjects/RNCl/runs/hybrid-tiny_imagenet-known-20-split0-2020-09-21_05-49-50-PM"
        PATH = "{}/{}{}_hybrid".format(PATH_1, DATASET, fold_num)

        test_encoder = encoder32()
        test_encoder.to(device)
        test_encoder.load_state_dict(torch.load("{}_encoder_latest.pth".format(PATH)))


        test_classifier = classifier32()
        test_classifier.to(device)
        test_classifier.load_state_dict(torch.load("{}_classifier_latest.pth".format(PATH)))

        test_flow = ResidualFlow(n_classes=20,
                                 input_size=(64, 128, 4, 4),
                                 n_blocks=[32, 32, 32],
                                 intermediate_dim=512,
                                 factor_out=False,
                                 quadratic=False,
                                 init_layer=None,
                                 actnorm=True,
                                 fc_actnorm=False,
                                 dropout=0,
                                 fc=False,
                                 coeff=0.98,
                                 vnorms='2222',
                                 n_lipschitz_iters=None,
                                 sn_atol=1e-3, sn_rtol=1e-3,
                                 n_power_series=None,
                                 n_dist='poisson',
                                 n_samples=1,
                                 kernels='3-1-3',
                                 activation_fn='swish',
                                 fc_end=True,
                                 n_exact_terms=2,
                                 preact=True,
                                 neumann_grad=True,
                                 grad_in_forward=False,
                                 first_resblock=True,
                                 learn_p=False,
                                 classification='hybrid',
                                 classification_hdim=256,
                                 block_type='resblock')

        test_flow.to(device)

        with torch.no_grad():
            x = torch.rand(1, *input_size[1:]).to(device)
            test_flow(x)
        checkpt = torch.load("{}_flow_latest.pth".format(PATH))
        sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
        state = test_flow.state_dict()
        state.update(sd)
        test_flow.load_state_dict(state, strict=True)

        hybrid = HybridModel(test_encoder, test_classifier, test_flow)

        closed_acc = evalute_classifier(hybrid, closed_testloader)
        print("closed-set accuracy: ", closed_acc)
        auc_d = evaluate_openset(hybrid, closed_testloader, open_testloader)
        print("auc discriminator: ", auc_d)

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
