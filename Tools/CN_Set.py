import torch
from torch.utils.data.dataset import Subset
from torch.utils.data.dataset import ConcatDataset
import numpy as np

from torchvision.datasets import ImageFolder

class LoadedImageFolderDataset(ImageFolder):

    def __init__(self,root, transform=None, use_cache=False):


        self.use_cache = use_cache
        super(LoadedImageFolderDataset, self).__init__(root, transform)
        self.cached_data=[None for i in range(len(self.samples))]

    def __getitem__(self, index):

        path, target = self.samples[index]

        if not self.use_cache:

            sample = self.loader(path)
            self.cached_data[index]=sample
        else:
            sample = self.cached_data[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



    def set_use_cache(self, use_cache):
        import PIL
        counter=0
        print("the len of cash is",len(self.cached_data))
        for x in self.cached_data:
            if isinstance(x, PIL.Image.Image):
                counter+=1
        print("the len of cash is", counter)
        # if use_cache:
        #     self.cached_data = torch.stack(self.cached_data)
        # else:
        #     self.cached_data = []
        self.use_cache = use_cache

class CN_Set(Subset):

    def __init__(self, dataset,indices,pre_num_classes):

        super(CN_Set, self).__init__(dataset,indices)
        self.shift=pre_num_classes

    def remove_inds(self, out_inds):
        new_inds=list(set( self.indices)-set(out_inds))
        self.indices=new_inds

    def add_inds(self,in_inds):
        new_inds = list(set(self.indices) + set(in_inds))
        self.indices = new_inds

    def __getitem__(self, idx):
        item = super(CN_Set, self).__getitem__(idx)
        return (item[0], item[1] + (self.shift))



class Continual_ConcatDataset(ConcatDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, with shifting labels
    Args:
        datasets (iterable): List of datasets to be concatenated
        dataset_classes:number of classes in each dataset
    """

    def __init__(self, datasets,dataset_classes):
        super(Continual_ConcatDataset, self).__init__(datasets)

        self.datasets = list(datasets)
        self.dataset_classes=dataset_classes
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.cum_sizes = np.cumsum([len(x) for x in self.datasets])

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        item=super(Continual_ConcatDataset, self).__getitem__(idx)

        dataset_index = self.cum_sizes.searchsorted(idx, 'right')

        return (item[0],item[1]+sum(self.dataset_classes[:dataset_index]))



import random

from torchvision import transforms
#from data.imgfolder import *

import bisect
import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []
    # print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname
                    # print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    return images


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainVal(ImageFolder):
    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None, class_to_idx=None, imgs=None,use_cache=True):
        """
        :param root: root path of the dataset
        :param files_list: list of filenames to include in this dataset
        :param classes: classes to include, based on subdirs of root if None
        :param class_to_idx: overwrite class to idx mapping
        :param imgs: list of image paths (under root)
        """
        if classes is None:
            assert class_to_idx is None
            classes, class_to_idx = find_classes(root)
        elif class_to_idx is None:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("Creating Imgfolder with root: {}".format(root))
        imgs = make_dataset(root, class_to_idx, files_list) if imgs is None else imgs
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: {}\nSupported image extensions are: {}".
                                format(root, ",".join(IMG_EXTENSIONS))))
        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.use_cache = False
        self.cached_data = [None for i in range(len(self.samples))]

    def __getitem__(self, index):

        path, target = self.samples[index]

        if not self.use_cache:

            sample = self.loader(path)
            self.cached_data[index]=sample
        else:
            sample = self.cached_data[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



    def set_use_cache(self, use_cache):
        import PIL
        counter=0
        print("the len of cash is",len(self.cached_data))
        for x in self.cached_data:
            if isinstance(x, PIL.Image.Image):
                counter+=1
        print("the len of cash is", counter)
        # if use_cache:
        #     self.cached_data = torch.stack(self.cached_data)
        # else:
        #     self.cached_data = []
        self.use_cache = use_cache