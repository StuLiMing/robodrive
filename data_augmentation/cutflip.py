import random
import copy
import numpy as np

def cutflip(image,depth,cut_graft=None,dim=0):
    """在 0.2h 和 0.8h 之间随机找一个点切一刀，将切分后的图片的上下两部分位置调换。

    该操作有 0.5 的概率执行，否则什么也不做

    可选参数: 

    cut_graft: 自定义 cut 的位置。

    比如，cut_graft=[0.3h,0.4h]，那新图的0到0.3h会是原图的0.7h到h，新图的0.3h到0.4h会是原图的0.6h到0.7h，新图的0.4h到h会是原图的0到h

    dim: cut 的方向

    默认 dim=0，横着切。当 dim=1 时竖着切。 

    """
    p = random.random()
    if p<0.5:
        return image,depth
    image_copy = copy.deepcopy(image)
    depth_copy = copy.deepcopy(depth)
    h,w,c = image.shape
    if dim==0:
        L=h
    elif dim==1:
        L=w
    if cut_graft!=None:
        N=len(cut_graft)+1
    else:
        N=2
    cut_list=[]      
    cut_interval_list = []       
    if cut_graft!=None:
        cut_list.extend(cut_graft)
    else:
        cut_list.append(random.randint(int(0.2*L),int(0.8*L)))
    cut_list.append(L)
    cut_list.append(0)  
    cut_list.sort()
    cut_list_inv = np.array([L]*(N+1))-np.array(cut_list)
    for i in range(len(cut_list)-1):
        cut_interval_list.append(cut_list[i+1]-cut_list[i])
    if dim==0:
        for i in range(N):
            image[cut_list[i]:cut_list[i+1],:,:] = image_copy[cut_list_inv[i]-cut_interval_list[i]:cut_list_inv[i],:,:]
            depth[cut_list[i]:cut_list[i+1],:,:] = depth_copy[cut_list_inv[i]-cut_interval_list[i]:cut_list_inv[i],:,:]
    elif dim==1:
        for i in range(N):
            image[:,cut_list[i]:cut_list[i+1],:] = image_copy[:,cut_list_inv[i]-cut_interval_list[i]:cut_list_inv[i],:]
            depth[:,cut_list[i]:cut_list[i+1],:] = depth_copy[:,cut_list_inv[i]-cut_interval_list[i]:cut_list_inv[i],:]
    return image,depth