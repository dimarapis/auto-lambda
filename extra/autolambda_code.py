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

        raw_h, raw_w = data_dict['rgb'].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  # initialise cropping coordinates
        flip_prop = random.random()

        for task in data_dict:
            if task == 'file':
                continue
                
            if len(data_dict[task].shape) == 2:   # make sure single-channel labels are in the same size [H, W, 1]
                data_dict[task] = data_dict[task].unsqueeze(0)

            # Resize based on randomly sampled scale
            if task in ['rgb', 'noise']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.BILINEAR)
            elif task in ['normals', 'depth', 'semantic', 'part_seg', 'disp']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.NEAREST)

            # Add padding if crop size is smaller than the resized size
            if self.crop_size[0] > resized_size[0] or self.crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(self.crop_size[1] - resized_size[1], 0), max(self.crop_size[0] - resized_size[0], 0)
                if task in ['rgb']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       padding_mode='reflect')
                elif task in ['semantic', 'part_seg', 'disp']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=-1, padding_mode='constant')  # -1 will be ignored in loss
                elif task in ['normals', 'depth', 'noise']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=0, padding_mode='constant')  # 0 will be ignored in loss

            # Random Cropping
            if i + j + h + w == 0:  # only run once
                i, j, h, w = transforms.RandomCrop.get_params(data_dict[task], output_size=self.crop_size)
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            # Random Flip
            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
                if task == 'normals':
                    data_dict[task][0, :, :] = - data_dict[task][0, :, :]

            # Final Check:
            if task == 'depth':
                data_dict[task] = data_dict[task] / sc

            if task == 'disp':  # disparity is inverse depth
                data_dict[task] = data_dict[task] * sc

            if task in ['semantic', 'part_seg']:
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict


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
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/test'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/rgb_npy'), '*.npy'))
        self.noise = torch.rand(self.data_len, 1, 288, 384)

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        file = self.data_path + '/rgb_npy/{:d}.npy'.format(index)

        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/rgb_npy/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/semantic_npy/{:d}.npy'.format(index))).long()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth_npy/{:d}.npy'.format(index)), -1, 0)).float()
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normals_npy/{:d}.npy'.format(index)), -1, 0)).float()
        #noise = self.noise[index].float()

        data_dict = {'file': file, 'rgb': image, 'depth': depth, 'semantic': semantic, 'normals': normal}# 'noise': noise}

        
        
        # apply data augmentation if required
        #if self.augmentation:
        #    data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

        #normalized_image = 2. * data_dict.pop('rgb') - 1.  # normalised to [-1, 1]
        #data_dict['rgb'] = normalized_image
        #return im, data_dict
    
        return data_dict
    
    def __len__(self):
        return self.data_len

class SimWarehouse(data.Dataset):
    """
    SimWarehouse dataset, 3 tasks + 1 generated useless task
    Included tasks:
        1. Depth prediction,
        2. Semantic Segmentation,
        3. Surface Normal prediction,
        4. Not implemented. Noise prediction [to test auxiliary learning, purely conflict gradients]
    """
    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.files = []
        self.dataset_type = 'warehouse_sim'
        self.root = '.'
        self.data_file = ''

        # read the data file
        if train:
            self.data_path = root + '/train'
            self.data_file = 'dataset/sim_warehouse/train/datalist_train_warehouse_sim.list'
        else:
            self.data_path = root + '/test'
            self.data_file = 'dataset/sim_warehouse/test/datalist_test_warehouse_sim.list'
        
        
        
        with open(os.path.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data_row in data_list:
                if len(data_row) == 0:
                    continue
                
                data_columns = data_row.split(' ')
                if self.dataset_type == 'warehouse_sim':
                    self.files.append({
                        "rgb": data_columns[0].replace('rgb','rgb_npy')+'.npy',
                        "depth": data_columns[1].replace('depth','depth_npy')+'.npy',
                        "semantic": data_columns[2],#.replace('semantic','semantic_npy')+'.npy',
                        "normals" : data_columns[3].replace('normals','normals_npy')+'.npy'
                    })


        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/rgb_npy'), '*.npy'))
        #self.noise = torch.rand(self.data_len, 1, 360, 640)

    def __getitem__(self, index):
        #print(index)
        # load data from the pre-processed npy files
        if self.dataset_type == 'warehouse_sim':
            
            #file = self.data_path + '/rgb_npy/{:d}.npy'.format(index)
            #print(self.files[index])
            file = self.files[index]['rgb']
            image_np = np.load(self.files[index]['rgb'])
            if np.shape(image_np)[2] == 4:
                image_np = image_np[:,:,:3]
            image = torch.from_numpy(np.moveaxis(image_np, -1, 0)).float() / 255.0
            seg = torch.from_numpy(np.load(self.files[index]['semantic']).astype(np.int32)).long()
            depth = torch.from_numpy(np.load(self.files[index]['depth'])).float() / 1000.0#, -1, 0)).float()
            normal = torch.from_numpy(np.moveaxis(np.load(self.files[index]['normals']), -1, 0)).float()
            #noise = self.noise[index].float()
            #print(semantic.shape)
            #semantic_resized = torch.nn.functional.interpolate(semantic, size=(360,640), mode='interpolate', align_corners=True)
            #semantic_resized = transforms_f.resize(semantic, (360,640), Image.NEAREST)
   
            data_dict = {'file': file, 'im': image, 'depth': depth, 'seg': seg, 'normal': normal}#, 'noise': noise}# 'noise': noise}

            #data_dict = DataTransform(crop_size=[360, 640])(data_dict)
            
            #data_dict = {'file': file, 'rgb': image, 'depth': depth, 'semantic': semantic, 'normals': normal}# 'noise': noise}
            
            #for task in data_dict.keys():
                #print(task)
                #print(data_dict[task])
            #    if task != 'file' and task != 'im' and len(data_dict[task].shape) == 2:#torch.Size([:, :, :]):
                    #print(task)
                    #pass
                    #print(data_dict[task])
                    #data_dict[task] = data_dict[task].unsqueeze(0)
            #        data_dict[task] = data_dict[task].unsqueeze(0)    
            
                        
            #if self.augmentation:
            #    data_dict = DataTransform(crop_size=[160, 320], scales=[1.0, 1.2, 1.5])(data_dict)
        # rgb = np.array(Image.open(self.files[index]['rgb']))
        # if np.shape(rgb)[2] == 4:
        #     rgb = rgb[:,:,:3]
        # depth = np.array(Image.open(self.files[index]['depth']))
        # semantic = np.load(self.files[index]['semantic']).astype(np.float32)
        
        # #normals = np.array(Image.open(self.files[index]['normals']))
        # normals = np.load(self.files[index]['normals'])#.astype(np.float32)
        
        # file_id = self.files[index]['rgb']
        # #random_sample_no = self.files[index]['random_sampling'] #Only for NYUv2
        # transformed_data_sample = DecnetDataloader.decnet_transform(file_id, rgb, depth, semantic, normals)

        # return transformed_data_sample

        # apply data augmentation if required
            #data_dict['rgb'] = self.normalize(data_dict['rgb'])
            
            return image,data_dict

        else:
        
            file = self.data_path + '/rgb_npy/{:d}.npy'.format(index)

            
            image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/rgb_npy/{:d}.npy'.format(index)), -1, 0)).float()
            semantic = torch.from_numpy(np.load(self.data_path + '/semantic_npy/{:d}.npy'.format(index))).long()
            depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth_npy/{:d}.npy'.format(index)), -1, 0)).float()
            normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/normals_npy/{:d}.npy'.format(index)), -1, 0)).float()
            noise = self.noise[index].float()

            semantic_resized = semantic.squeeze(1)
            print(semantic_resized.shape)
            semantic_resized = torch.nn.functional.interpolate(semantic_resized, size=(360,640), mode='interpolate', align_corners=True)
            data_dict = {'file': file, 'rgb': image, 'depth': depth, 'semantic': semantic_resized, 'normals': normal}# 'noise': noise}
            #print(image)
            # apply data augmentation if required


            
            if self.augmentation:
                data_dict = DataTransform(crop_size=[288, 384], scales=[1.0, 1.2, 1.5])(data_dict)

            #im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
            #return im, data_dict
            return data_dict

    def __len__(self):
        #print('datalength',self.data_len)
        return self.data_len
    
    def normalize(self,data):
        normalization = transforms.ToTensor()
        return normalization(data)