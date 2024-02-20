import numpy as np
from PIL import Image
import os

from cutflip import cutflip
from augmix import augment_and_mix

def testcutflip():
    # 使用Pillow读取图像
    img = Image.open('./images/test.jpg')

    # 将图像转换为NumPy数组
    img_array = np.array(img)
    depth=np.arange(img_array.shape[0]*img_array.shape[1]*img_array.shape[2]).reshape(img_array.shape)

    img_array,depth=cutflip(img_array,depth,cut_graft=[int(img_array.shape[1]*0.3),int(img_array.shape[1]*0.4)],dim=1)

    # 将NumPy数组转换为PIL图像
    img = Image.fromarray(img_array)

    # 保存图像到文件，假设我们保存为JPEG格式
    img.save('./images/cutflip.jpg')
    
def testaugmix():
    image_path = './images/test.jpg'
    image = Image.open(image_path)
    image_array = np.array(image,dtype=np.float32)/255
    augmented_image = augment_and_mix(image_array, severity=3, width=3, depth=-1, alpha=1.)
    augmented_image = np.clip(augmented_image * 255, 0, 255).astype(np.uint8)
    augmented_image = Image.fromarray(augmented_image)
    augmented_image.save("./images/augmix.jpg")
 
if __name__=="__main__":
    path="./data_augmentation"
    os.chdir(path)
    testcutflip()
    testaugmix()
    
    
