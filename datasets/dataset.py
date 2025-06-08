import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from .utils import get_camera_rays, alphanum_key, as_intrinsics_matrix
import matplotlib.pyplot as plt
import re

def compute_edge(rgb_data, depth_data, instance=None, UseInstance=False):

    if not UseInstance :
        #edges0 = cv2.Canny(depth_data.astype(np.uint8), 8, 20)
        edges0 = cv2.adaptiveThreshold(depth_data.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        mask = (depth_data == 0).astype(np.uint8)
        kernel = np.ones((11, 11), np.uint8)
        kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        rgb_image = cv2.dilate(rgb_data, kernel_e, iterations=10)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        remaining_mask = cv2.bitwise_not(dilated_mask)
        remaining_mask[remaining_mask == 254] = 0
        edges_depth = edges0 #* remaining_mask
        edges_rgb = cv2.Canny(rgb_image, 60, 180)#*dilated_mask
        edges = edges_rgb + edges_depth
    else:
        edges = cv2.Canny(instance, 50, 150)
    edges = np.where(edges == 255, 0, 1).astype(np.uint8)
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 0, dstType=cv2.CV_32F)
    edge_data = np.exp(-dist_transform / 10)

    return edge_data

def compute_edge_semantic(semantic_data, depth_data, instance=None, UseInstance=False):
    if not UseInstance :
        edges0 = cv2.adaptiveThreshold(depth_data.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        mask = (depth_data == 0).astype(np.uint8)                      
        kernel = np.ones((11, 11), np.uint8)                           
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)          
        remaining_mask = cv2.bitwise_not(dilated_mask)                
        remaining_mask[remaining_mask == 254] = 0                     
        edges_depth = edges0 #* remaining_mask                           
        edges_semantic = cv2.Canny(semantic_data, 1, 1)#*dilated_mask    
        edges = edges_semantic                         
    else:
        edges = cv2.Canny(instance, 50, 150)
    edges = np.where(edges == 255, 0, 1).astype(np.uint8)
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 0, dstType=cv2.CV_32F)
    edge_data = np.exp(-dist_transform / 10)

    return edge_data

def create_border_data(depth_data, border_width=10):
    border_data = torch.zeros_like(depth_data, dtype=torch.float32)
    border_data[:border_width, :] = 1 
    border_data[-border_width:, :] = 1 
    border_data[:, :border_width] = 1 
    border_data[:, -border_width:] = 1 
    return border_data

def get_dataset(config):
    '''
    Get the dataset class from the config file.
    '''
    if config['dataset'] == 'stereomis':
        dataset = StereoMISDataset
        
    elif config['dataset'] == 'super':
        dataset = SuperDataset
    

    
    return dataset(config, 
                   config['data']['datadir'], 
                   trainskip=config['data']['trainskip'], 
                   downsample_factor=config['data']['downsample'], 
                   sc_factor=config['data']['sc_factor'])

class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H']//cfg['data']['downsample'],\
            cfg['cam']['W']//cfg['data']['downsample']

        self.fx, self.fy =  cfg['cam']['fx']//cfg['data']['downsample'],\
             cfg['cam']['fy']//cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx']//cfg['data']['downsample'],\
             cfg['cam']['cy']//cfg['data']['downsample']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size*2) * (self.W - self.crop_size*2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
        
    
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()

class StereoMISDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(StereoMISDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(glob.glob(f'{self.basedir}/video_frames/*l.png'))[-4000:]
        self.depth_paths = sorted(glob.glob(f'{self.basedir}/depth/*.png'))[-4000:]
        # pattern = re.compile(r'^\d+\.png$')
        # self.depth_paths = sorted(
        #     [f for f in glob.glob(os.path.join(basedir, 'depth', '*.png')) if pattern.match(os.path.basename(f))],
        #     key=lambda x: int(os.path.basename(x)[:-4])
        # )
        # print(self.depth_paths)
        # assert 0


        self.semantic_paths = sorted(
           glob.glob(os.path.join(
           self.basedir, 'masks', '*.png')))[-2000:]#, key=lambda x: int(os.path.basename(x)[:-4]))

        # all_pixel_values = set()  # 使用集合来保存所有不同的像素值

        # for pth in self.semantic_paths:
        #     img = cv2.imread(pth)  # 以灰度图形式读取图像
        #     all_pixel_values.update(np.unique(img))

        # # 将集合转换为列表并打印
        # unique_pixel_values = list(all_pixel_values)
        # print(unique_pixel_values)

        # assert 0

        # print(self.instance_path)
        # assert 0

        self.load_poses(os.path.join(self.basedir, 'pose'))

        # self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F, 5)
        

        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge']*2
            self.W -= self.config['cam']['crop_edge']*2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']
   
    def __len__(self):
        return self.num_frames
  
    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]
        semantic_path = self.semantic_paths[index//2]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_data0 = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()
        # instance_data = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        # instance_data = cv2.resize(instance_data, (W, H))

        semantic_data = cv2.imread(semantic_path)
        semantic_data = cv2.resize(semantic_data, (W, H))
        
        #edge_data = compute_edge(color_data, depth_data0)
        edge_data_semantic = compute_edge_semantic(semantic_data, depth_data0)


        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor


        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)
            #edge_data = cv2.resize(edge_data, (W, H), interpolation=cv2.INTER_AREA)
            edge_data_semantic = cv2.resize(edge_data_semantic, (W, H), interpolation=cv2.INTER_AREA)
        
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            #edge_data = edge_data[edge:-edge, edge:-edge]
            edge_data_semantic = edge_data_semantic[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        #edge_data = torch.from_numpy(edge_data.astype(np.float32))
        edge_data_semantic = torch.from_numpy(edge_data_semantic.astype(np.float32))
        border_data = create_border_data(depth_data)

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            #"edge": edge_data,
            "edge_semantic": edge_data_semantic,
            "border": border_data,
            "direction": self.rays_d
        }

        return ret

    def load_poses(self, path):
        self.poses = []
        #with open(path, "r") as f:
        #    lines = f.readlines()
        for i in range(len(self.img_files)):
            #line = lines[i]
            #c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w=np.eye(4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class SuperDataset(BaseDataset):
    def __init__(self, cfg, basedir, trainskip=1, 
                 downsample_factor=1, translation=0.0, 
                 sc_factor=1., crop=0):
        super(SuperDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop

        self.img_files = sorted(glob.glob(f'{self.basedir}/rgb/*left.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.basedir}/rgb/*left_depth.npy'))
        self.semantic_paths=sorted(glob.glob(f'{self.basedir}/seg/png_masks/*left.png'))

        # all_pixel_values = set()  # 使用集合来保存所有不同的像素值

        # for pth in self.semantic_paths:
        #     img = cv2.imread(pth)  # 以灰度图形式读取图像
        #     all_pixel_values.update(np.unique(img))

        # # 将集合转换为列表并打印
        # unique_pixel_values = list(all_pixel_values)
        # print(unique_pixel_values)

        # assert 0

        # print(self.instance_path)
        # assert 0

        self.load_poses(os.path.join(self.basedir, 'pose'))

        # self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F, 5)
        

        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge']*2
            self.W -= self.config['cam']['crop_edge']*2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']
   
    def __len__(self):
        return self.num_frames
  
    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]
        semantic_path = self.semantic_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_data0 = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        elif '.npy' in depth_path:
            depth_data = np.load(depth_path)
            depth_data = depth_data.reshape(depth_data.shape[-2], depth_data.shape[-1])

            depth_data0 = depth_data
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()
        # instance_data = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        # instance_data = cv2.resize(instance_data, (W, H))
        if '.png' in semantic_path:
            semantic_data = cv2.imread(semantic_path)
            semantic_data = cv2.resize(semantic_data, (W, H))
        elif '.npy' in semantic_path:
            semantic_data = np.load(semantic_path)#.reshape(3,semantic_data.shape[-2], semantic_data.shape[-1])
            semantic_data = semantic_data.reshape(semantic_data.shape[-2], semantic_data.shape[-1],3)
            print(semantic_data)
            semantic_data = cv2.cvtColor(semantic_data, cv2.COLOR_BGR2GRAY)
        #edge_data = compute_edge(color_data, depth_data0)
        #print(semantic_data.shape)

        edge_data_semantic = compute_edge_semantic(semantic_data, depth_data0)


        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor


        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)
            #edge_data = cv2.resize(edge_data, (W, H), interpolation=cv2.INTER_AREA)
            edge_data_semantic = cv2.resize(edge_data_semantic, (W, H), interpolation=cv2.INTER_AREA)
        
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            #edge_data = edge_data[edge:-edge, edge:-edge]
            edge_data_semantic = edge_data_semantic[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))
        #edge_data = torch.from_numpy(edge_data.astype(np.float32))
        edge_data_semantic = torch.from_numpy(edge_data_semantic.astype(np.float32))
        border_data = create_border_data(depth_data)

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            #"edge": edge_data,
            "edge_semantic": edge_data_semantic,
            "border": border_data,
            "direction": self.rays_d
        }

        return ret

    def load_poses(self, path):
        self.poses = []
        #with open(path, "r") as f:
        #    lines = f.readlines()
        for i in range(len(self.img_files)):
            #line = lines[i]
            #c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w=np.eye(4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w[:3, 3] *= self.sc_factor
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
