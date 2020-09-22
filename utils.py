import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def images_to_probs(net, images):
    outputs = net(images)
    _, preds_tensor = torch.max(outputs, 1)
    preds_tensor_cpu = preds_tensor.cpu()
    preds = np.squeeze(preds_tensor_cpu.numpy())
    return preds, [F.softmax(el, dim=0)[i].item for i, el in zip(preds, outputs)]

def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2}".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

class TensorNode:
    def __init__(self, tensor):
        self.tensor = tensor
        self.norm = torch.norm(self.tensor)


class PriorityTensorQueue:

    def __init__(self):
        self.queue = list()

    def insert(self, node):
        if self.size() == 0:
            self.queue.append(node)
        else:
            for x in range(0, self.size()):
                if node.norm >= self.queue[x].norm:
                    if x == (self.size()-1):
                        self.queue.insert(x+1, node)
                    else:
                        continue
                else:
                    self.queue.insert(x, node)
                    return True

    def delete(self):
        return self.queue.pop(0)

    def show(self):
        for x in self.queue:
            print(str(x.tensor)+" - "+str(x.norm))

    def size(self):
        return len(self.queue)

class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.module = module
        self.decay = decay
        self.shadow_params = {}
        self.nparams = sum(p.numel() for p in module.parameters())

    def init(self):
        for name, param in self.module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def apply(self):
        if len(self.shadow_params) == 0:
            self.init()
        else:
            with torch.no_grad():
                for name, param in self.module.named_parameters():
                    self.shadow_params[name] -= (1 - self.decay) * (self.shadow_params[name] - param.data)

    def set(self, other_ema):
        self.init()
        with torch.no_grad():
            for name, param in other_ema.shadow_params.items():
                self.shadow_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module.named_parameters():
            param.data.copy_(self.shadow_params[name])

    def swap(self):
        for name, param in self.module.named_parameters():
            tmp = self.shadow_params[name].clone()
            self.shadow_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def save_checkpoint(state, save, epoch, last_checkpoints=None, num_checkpoints=None):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)

    if last_checkpoints is not None and num_checkpoints is not None:
        last_checkpoints.append(epoch)
        if len(last_checkpoints) > num_checkpoints:
            rm_epoch = last_checkpoints.pop(0)
            os.remove(os.path.join(save, 'checkpt-%ds04.pth' % rm_epoch))