import numpy as np
import os
from nuscenes import NuScenes,NuScenesExplorer
from config import cfg
import torch
import matplotlib.pyplot as plt

class NuScenesDataset:      
    def __init__(self, cfg):  
        self.dataset_version = cfg.DATASET.VERSION 
        self.dataroot = cfg.PATH.DATAROOT
        self.nusc = NuScenes(version=self.dataset_version, dataroot=self.dataroot, verbose=False)
        self.nusc_explorer = NuScenesExplorer(self.nusc)
        # 参数
        self.original_width = cfg.DATASET.ORIGINAL_SIZE.WIDTH
        self.original_height = cfg.DATASET.ORIGINAL_SIZE.HEIGHT
    
    
    def get_depth(self, index, sensor):
        lidar_data = self.nusc.get('sample_data', self.nusc.sample[index]['data']['LIDAR_TOP'])
        cam_data = self.nusc.get('sample_data', self.nusc.sample[index]['data'][sensor])
        points, depth, _ = self.nusc_explorer.map_pointcloud_to_image(pointsensor_token=lidar_data['token'],
                                                                    camera_token=cam_data['token'])
        lidar_proj = self.generate_image_from_points(points[:2], depth, (self.original_height, self.original_width))

        return np.expand_dims(lidar_proj, axis=0)
    
    
    @staticmethod
    def generate_image_from_points(points, features, imsize):
        h, w = imsize
        points = points.astype(np.int32)
        projection = np.zeros((h, w), dtype=np.float32)
        projection[points[1], points[0]] = features

        return projection


nuScenesDataset=NuScenesDataset(cfg)
