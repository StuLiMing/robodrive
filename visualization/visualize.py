import cv2
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from matplotlib.cm import get_cmap
import PIL

from config import cfg


def visualize_rgb(img_rgb, filename):
    """
    保存一张 rgb 图像
    ---------------
    参数: img_rgb: PIL加载的图像或者tensor或np加载的图像
    """
    if isinstance(img_rgb,PIL.JpegImagePlugin.JpegImageFile):
        img_rgb.save(os.path.join(cfg.PATH.RGBIMG,f"{filename}.jpg"))
    elif isinstance(img_rgb,torch.Tensor) or isinstance(img_rgb,np.array):
        save_image(img_rgb, os.path.join(cfg.PATH.RGBIMG, f"{filename}.jpg"))
     

def visualize_depth(inv_depth, filename):
    """
    可视化一张预测的逆深度图
    -----------------------
    参数:
    inv_depth: torch.tensor or np.array，预测的逆深度图
    """
    
    if type(inv_depth) == torch.Tensor:
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()

    cm = get_cmap("plasma")
    # normalizer = np.percentile(inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    normalizer = np.percentile(inv_depth, 95)
    inv_depth /= (normalizer + 1e-6)
    # 去除了透明度通道
    inv_depth=cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]*255
    # 从 RGB 转成 BGR
    inv_depth = cv2.cvtColor(inv_depth, cv2.COLOR_RGB2BGR)
    # 保存图像,因为后面的 imwrite() 要求这种图像
    cv2.imwrite(os.path.join(cfg.PATH.EVALIMG, f"{filename}_inv_depth.jpg"), inv_depth)

def visualize_gt_depth(gt_depth,img, filename,min_depth=cfg.GT.MINDEPTH,max_depth=cfg.GT.MAXDEPTH):
    """
    可视化gt深度图
    --------------
    参数：
    gt_depth: GT 图
    img: 摄像机拍摄的原图
    min_depth: 需要被可视化的最小深度(必须大于0)
    max_depth: 需要被可视化的最大深度
    """
    assert(min_depth>0)
    z = gt_depth.cpu().numpy().flatten()
    valid_mask = (min_depth <= z) & (z <= max_depth)

    _, height, width = gt_depth.shape
    indices = np.indices((height, width))
    indices_x = indices[1, :, :].flatten()
    indices_y = indices[0, :, :].flatten()
    indices_x = indices_x[valid_mask]
    indices_y = indices_y[valid_mask]
    z = z[valid_mask]
    # Create the figure and the axes
    fig, ax = plt.subplots()
    # Show image
    if isinstance(img,PIL.JpegImagePlugin.JpegImageFile):
        plt.imshow(img)
    elif isinstance(img,torch.Tensor):
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    # Scatter LiDAR points
    plt.scatter(indices_x, indices_y, s=2, c=z, edgecolors='none', cmap="plasma_r", vmin=0, vmax=800)
    # Axis limiting
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.xticks([])
    plt.yticks([])

    # Remove boundaries
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    # Save the figure
    plt.savefig(os.path.join(cfg.PATH.GTIMG, f"{filename}.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()
    