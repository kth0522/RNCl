"""
A Converter converts between:
    examples (each one a dict with keys like "filename" and "label")
    arrays (numpy arrays input to or output from a network)

Dataset augmentation can be accomplished with a Converter that returns a
different array each time to_array is called with the same example
"""
import os
import numpy as np
import torch
import random
import imutil
from PIL import Image
import torchvision.transforms as transforms

DATA_DIR = '/home/taehokim/PycharmProjects/RNCl/data'

# Converters can be used like a function, on a single example or a batch
class Converter(object):
    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            return [self.from_array(e) for e in inputs]
        elif isinstance(inputs, list):
            return np.array([self.to_array(e) for e in inputs])
        else:
            return self.to_array(inputs)


# Crops, resizes, normalizes, performs any desired augmentations
# Outputs images as eg. 32x32x3 np.array or eg. 3x32x32 torch.FloatTensor
class ImageConverter(Converter):
    def __init__(self,
            dataset,
            image_size=32,
            transform=None,
            **kwargs):
        width, height = image_size, image_size
        self.transform = transform
        self.img_shape = (width, height)
        self.data_dir = dataset.data_dir

    def to_array(self, example):
        filename = os.path.expanduser(example['filename'])
        if not filename.startswith('/'):
            filename = os.path.join(DATA_DIR, filename)
        img = imutil.load(filename)
        # img = img.transpose((2, 0, 1))
        pil_img = Image.fromarray(img.astype('uint8'), 'RGB')

        # img *= 1.0 / 255
        tensor_img = self.transform(pil_img)
        img = tensor_img.numpy()
        img *= 1.0 / 255
        return img

    def from_array(self, array):
        return array


# LabelConverter extracts the class labels from DatasetFile examples
# Each example can have only one class
class LabelConverter(Converter):
    def __init__(self, dataset, label_key="label", **kwargs):
        self.label_key = label_key
        self.labels = get_labels(dataset, label_key)
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}
        print("LabelConverter: labels are {}".format(self.labels))

    def to_array(self, example):
        return self.idx[example[self.label_key]]

    def from_array(self, array):
        return self.labels[np.argmax(array)]


# Each example now has a label for each class:
#    1 (X belongs to class Y)
#   -1 (X does not belong to class Y)
#   0  (X might or might not belong to Y)
class FlexibleLabelConverter(Converter):
    def __init__(self, dataset, label_key="label", negative_key="label_n", **kwargs):
        self.label_key = label_key
        self.negative_key = negative_key
        self.labels = sorted(list(set(get_labels(dataset, label_key) + get_labels(dataset, negative_key))))
        #self.labels = get_labels(dataset, label_key)
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}
        print("FlexibleLabelConverter: labels are {}".format(self.labels))

    def to_array(self, example):
        array = np.zeros(self.num_classes)
        if self.label_key in example:
            array[:] = 0  # Negative labels
            idx = self.idx[example[self.label_key]]
            array[idx] = 1  # Positive label
        if self.negative_key in example:
            idx = self.idx[example[self.negative_key]]
            array[idx] = 0
        return array

    def from_array(self, array):
        return self.labels[np.argmax(array)]


def get_labels(dataset, label_key):
    unique_labels = set()
    for example in dataset.examples:
        if label_key in example:
            unique_labels.add(example[label_key])
    return sorted(list(unique_labels))


# AttributeConverter extracts boolean attributes from DatasetFile examples
# An example might have many attributes. Each attribute is True or False.
class AttributeConverter(Converter):
    def __init__(self, dataset, **kwargs):
        unique_attributes = set()
        for example in dataset.examples:
            for key in example:
                if key.startswith('is_') or key.startswith('has_'):
                    unique_attributes.add(key)
        self.attributes = sorted(list(unique_attributes))
        self.num_attributes = len(self.attributes)
        self.idx = {self.attributes[i]: i for i in range(self.num_attributes)}

    def to_array(self, example):
        attrs = np.zeros(self.num_attributes)
        for i, attr in enumerate(self.attributes):
            # Attributes not present on an example are set to False
            attrs[i] = float(example.get(attr, False))
        return attrs

    def from_array(self, array):
        return ",".join(self.attributes[i] for i in range(self.attributes) if array[i > .5])
