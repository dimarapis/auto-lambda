import os
import cv2
import random
import torch
import fnmatch

import numpy as np
#import panoptic_parts as pp
import torch.utils.data as data
import matplotlib.pylab as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from PIL import Image


class DataTransform(object):
    def __init__(self, scales, crop_size, is_disparity=False):
        self.scales = scales
        self.crop_size = crop_size
        self.is_disparity = is_disparity

    def __call__(self, data_dict):
        if type(self.scales) == tuple:
            # Continuous range of scales
            sc = np.random.uniform(*self.scales)

        elif type(self.scales) == list:
            # Fixed range of scales
            sc = random.sample(self.scales, 1)[0]

        raw_h, raw_w = data_dict['im'].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  # initialise cropping coordinates
        flip_prop = random.random()

        for task in data_dict:
            if len(data_dict[task].shape) == 2:   # make sure single-channel labels are in the same size [H, W, 1]
                data_dict[task] = data_dict[task].unsqueeze(0)

            # Resize based on randomly sampled scale
            if task in ['im', 'noise']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.BILINEAR)
            elif task in ['normal', 'depth', 'seg', 'part_seg', 'disp']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.NEAREST)

            # Add padding if crop size is smaller than the resized size
            if self.crop_size[0] > resized_size[0] or self.crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(self.crop_size[1] - resized_size[1], 0), max(self.crop_size[0] - resized_size[0], 0)
                if task in ['im']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       padding_mode='reflect')
                elif task in ['seg', 'part_seg', 'disp']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=-1, padding_mode='constant')  # -1 will be ignored in loss
                elif task in ['normal', 'depth', 'noise']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=0, padding_mode='constant')  # 0 will be ignored in loss

            # Random Cropping
            if i + j + h + w == 0:  # only run once
                i, j, h, w = transforms.RandomCrop.get_params(data_dict[task], output_size=self.crop_size)
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            # Random Flip
            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
                if task == 'normal':
                    data_dict[task][0, :, :] = - data_dict[task][0, :, :]

            # Final Check:
            if task == 'depth':
                data_dict[task] = data_dict[task] / sc

            if task == 'disp':  # disparity is inverse depth
                data_dict[task] = data_dict[task] * sc

            if task in ['seg', 'part_seg']:
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict

class SimWarehouse(data.Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """
    
    def __init__(self, root, train=True, augmentation=False):
        
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)).astype(np.int32)).long()
        depth = torch.from_numpy(np.load(self.data_path + '/depth/{:d}.npy'.format(index))).float()  / 1000.0
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)).float()
        noise = self.noise[index].float()
        # Reshape the data, remove 4th channel
        image = image[:3, :, :]
        # Add depth channel
        depth = depth.unsqueeze(0)
        #semantic_resized = semantic.unsqueeze(0)
        #print(semantic_resized.shape)
        #print(semantic.unsqueeze(0).shape)
        semantic_resized = transforms.Resize((360,640))(semantic.unsqueeze(0)).squeeze(0)
        #semantic_resized = #torch.nn.functional.interpolate(semantic, size=(360,640), mode='interpolate', align_corners=True)
        #print(image.shape, semantic_resized.shape, depth.shape, normal.shape, noise.shape)
        #print(semantic_resized.max(), semantic_resized.min())
        
        
        data_dict = {'im': image, 'seg': semantic_resized, 'depth': depth, 'normal': normal, 'noise': noise}

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len
    
class NYUv2(data.Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """
    def __init__(self, root, train=True, augmentation=False):
        
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index))).long()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)).float()
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)).float()
        noise = self.noise[index].float()

        data_dict = {'im': image, 'seg': semantic, 'depth': depth, 'normal': normal, 'noise': noise}
        #print(image.shape, semantic.shape, depth.shape, normal.shape, noise.shape)

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len
    
    
    
class Taskonomy(data.Dataset):
    """
    NYUv2 dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Semantic Segmentation,
        2. Depth prediction,
        3. Surface Normal prediction,
        4. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """

    def __init__(self, root, train=True, augmentation=False):
        
        seed = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        # load the image

        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)).astype(np.int32)).long()
        depth = torch.from_numpy(np.load(self.data_path + '/depth/{:d}.npy'.format(index))).float()  / 512.0
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normal/{:d}.npy'.format(index)), -1, 0)).float()
        noise = self.noise[index].float()
        depth = depth.unsqueeze(0)
        semantic = transforms.Resize((256,256))(semantic.unsqueeze(0)).squeeze(0)
        depth = transforms.Resize((256,256))(depth)
        normal = transforms.Resize((256,256))(normal)
        noise = transforms.Resize((256,256))(noise)
        image = transforms.Resize((256,256))(image)
        data_dict = {'im': image, 'seg': semantic, 'depth': depth, 'normal': normal, 'noise': noise}

        #print(image.shape, semantic.shape, depth.shape, normal.shape, noise.shape)
        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
        return im, data_dict

    def __len__(self):
        return self.data_len