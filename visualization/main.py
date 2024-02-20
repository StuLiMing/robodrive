import torch
import os
from PIL import Image

from visualize import *
from utils import *
from load_gt import nuScenesDataset
from config import cfg

if __name__ == '__main__':
    orientation="back"
    idx=100
    sensor=orientation2camera(orientation)
    depth_gt = torch.Tensor(nuScenesDataset.get_depth(idx, sensor))
    img_path=nuScenesDataset.nusc.get('sample_data', nuScenesDataset.nusc.sample[idx]['data'][sensor])["filename"]
    path=os.path.join(cfg.PATH.DATAROOT,img_path)
    img =Image.open(path)
    # visualize_gt_depth(depth_gt,img,"333",0.1,80)
    visualize_rgb(img,"222")