import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as tf
import cv2
import random
import albumentations as A
from matplotlib import pyplot as plt
import cv2 as cv
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage import io, morphology
from skimage import io, color
from utils.mytool import *
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
#from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator

IMG_MEAN = np.array((83.00327463, 91.67919424, 79.05708592), dtype=np.float32)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


def random_crop(image,mask,gt,crop_size=(512,512)):
    not_valid = True
    n = 0
    while not_valid:
        i, j, h, w = transforms.RandomCrop.get_params(image,output_size=crop_size)
        image_crop = tf.crop(image,i,j,h,w)
        mask_crop = tf.crop(mask,i,j,h,w)
        gt_crop = tf.crop(gt,i,j,h,w)
            
        label = np.asarray(mask_crop, np.float32)       
        n=n+1

        if np.sum(label==255)>-1:
            not_valid = False

    return image_crop,mask_crop,gt_crop

def random_crops_with_masks(image, crop_size):

    if image.ndim == 3 and image.shape[0] <= 4:  
        # (C, H, W)
        C, H, W = image.shape
        channel_first = True
    elif image.ndim == 3:
        # (H, W, C)
        H, W, C = image.shape
        channel_first = False
    else:
        raise ValueError("输入图像必须是 3 维的 (H, W, C) 或 (C, H, W)")

    crop_h, crop_w = crop_size
    if crop_h > H or crop_w > W:
        raise ValueError("裁剪大小不能超过原图大小")

    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    if channel_first:
        crop = image[:, top:top+crop_h, left:left+crop_w]
    else:
        crop = image[top:top+crop_h, left:left+crop_w, :]

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top:top+crop_h, left:left+crop_w] = 1

    #crops = np.stack(crops, axis=0)   # (N, crop_h, crop_w, C) or (N, C, crop_h, crop_w)
    #masks = np.stack(masks, axis=0)   # (N, H, W)

    return crop, mask,(top,left)

def detect_edge(image):
    dy = cv.Sobel(image, cv.CV_64F,0,1)
    dx = cv.Sobel(image, cv.CV_64F,1,0)
    edge_intensity = np.sqrt(dy**2+dx**2)
    return edge_intensity 

def image_linear_transform(image, scale=[2,98]):
    image = np.float32(image)
    for i in range(image.shape[0]):
        min_,max_ = np.percentile(image[i],scale)
        image[i] = (image[i]-min_)/(max_-min_+1e-7)
    return np.clip(image,0,1)

def assign_superpixel_labels(segments, point_label_map, ignore_label=5):
    """根据点标签给整个超像素赋值"""
    label_map = np.zeros_like(segments, dtype=np.int32) + 255
    ys, xs = np.where(point_label_map != ignore_label)
    for y, x in zip(ys, xs):
        sp_id = segments[y, x]
        label_value = point_label_map[y, x]
        label_map[segments == sp_id] = label_value
    return label_map

def filter_skeleton_by_color_intensity(img, segments, skeleton_label, point_label_map, intensity_thresh=30):
    """
    根据超像素内点标签亮度均值过滤骨架线
    img: 彩色图 (H,W,3)
    segments: 超像素分割结果 (H,W)
    skeleton_label: 骨架线标签 (H,W)，255表示未标记区域
    point_label_map: 点标签 (H,W)，255表示无标签
    intensity_thresh: RGB亮度差阈值
    """
    filtered_skeleton = skeleton_label.copy()

    # 找到所有骨架像素
    ys, xs = np.where(skeleton_label != 255)
    if len(ys) == 0:
        return filtered_skeleton

    skel_indices = np.ravel_multi_index((ys, xs), skeleton_label.shape)
    skel_segments = segments[ys, xs]

    # 只处理骨架线所在的超像素
    unique_sp = np.unique(skel_segments)

    for sp_id in unique_sp:
        sp_mask = (segments == sp_id)

        # 该超像素中的点标签
        mask_point = sp_mask & (point_label_map != 255)
        if np.sum(mask_point) == 0:
            continue

        mean_rgb = np.mean(img[mask_point], axis=0)

        # 该超像素的骨架线像素
        mask_skel = sp_mask & (skeleton_label != 255)
        ys_skel, xs_skel = np.where(mask_skel)
        if len(ys_skel) == 0:
            continue

        skel_rgb = img[ys_skel, xs_skel, :]
        diff = np.linalg.norm(skel_rgb - mean_rgb, axis=1)

        # 过滤掉不合格的骨架点
        mask_invalid = diff > intensity_thresh
        filtered_skeleton[ys_skel[mask_invalid], xs_skel[mask_invalid]] = 255

    return filtered_skeleton

#training dataset
class ISPRSDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, ignore_label=255,set='P',id=11,mode=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.type = set
        self.id = id
        
        n_repeat = 50
        self.img_ids = self.img_ids * n_repeat

        self.files = []

         
        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s" % name)
            label_file = osp.join(self.root, str(self.type)+"/"+name)       
            gt_file = osp.join(self.root, "gt/"+name) #ground truth
            self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "gt": gt_file,
                    "name": name
                })
           

    def _rotation(self,img,gt,label):
        
        index = random.randint(1, 6)
        if index == 1:
            new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            new_gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            new_label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
            
        elif index ==2:
            new_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif index == 3:
            new_img = cv2.rotate(img, cv2.ROTATE_180)
            new_gt = cv2.rotate(gt, cv2.ROTATE_180)
            new_label = cv2.rotate(label, cv2.ROTATE_180)

        elif index == 4:
            new_img = cv2.flip(img,1)
            new_gt = cv2.flip(gt,1)
            new_label = cv2.flip(label,1)

        elif index==5:
            new_img = cv2.flip(img,0)
            new_gt = cv2.flip(gt,0)
            new_label = cv2.flip(label,0)   
        else:
            new_img = img
            new_gt = gt
            new_label = label

        return new_img,new_gt,new_label,index


    def _augment_strong(self, img,mask):
        transformer = A.Compose([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,val_shift_limit=10, p=0.3),#色调饱和度值
            A.RandomBrightnessContrast(brightness_limit=0.1,contrast_limit=0.1, p=0.2), 
            A.ChannelShuffle(blur_limit=3, p=1)
            ])#平移缩放旋转

        augmented = transformer(image=img, mask=mask)

        return augmented['image'], augmented.get('mask', None)

    def make_clslabel(self, label,num_classes=5,ingore_index=255):
        label_set = np.unique(label)
        label_set = label_set.astype(int)
        cls_label = np.zeros(num_classes)
        cls_label = cls_label.astype(int)
        for i in label_set:
            if i < ingore_index:
                cls_label[i] += 1
        return cls_label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('P')
        gt = Image.open(datafiles["gt"]).convert('P')
            
        image1,label1,gt1 = random_crop(image,label,gt,self.crop_size)

        if (image,label) == (0,0):
            return (0,0)

        label1 = np.asarray(label1, np.float32)
        image1 = np.asarray(image1)
        gt1 = np.asarray(gt1,np.float32)

        label1[label1>3]=255
        gt1[gt1>3]=255
                
        image_aug,gt_aug,label_aug,index  = self._rotation(image1,gt1,label1,)  

        #image_aug = cv2.resize(image_aug,(256,256),interpolation=cv2.INTER_CUBIC)      
        #label_aug = cv2.resize(label_aug,(256,256),interpolation=cv2.INTER_NEAREST)    
        #gt_aug = cv2.resize(gt_aug,(256,256),interpolation=cv2.INTER_NEAREST) 

        image_aug = tf.to_tensor(image_aug)
        image_aug = image_aug.numpy()

        return image_aug.copy(), label_aug.copy(),gt_aug.copy()
    
    
    
class ISPRSFullDataSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, ignore_label=255,set='P',id=11,mode=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.type = set
        self.id = id
        
        n_repeat = 1#50
        self.img_ids = self.img_ids * n_repeat

        self.files = []


        for name in self.img_ids:
            
            #img_file = osp.join(self.root, "img/%s" % name)
            img_file = osp.join(self.root, "img/%s" % name.replace('mask.png','sat.jpg'))
            label_file = osp.join(self.root, "fs/"+name)   #.replace('.tif','.png')
            gt_file = osp.join(self.root, "gt/"+name)#ground truth #.replace('.tif','.png')
                        
            #img_file = osp.join(self.root, "img/%s" % name.replace('mask.png','sat.jpg'))
            #label_file = osp.join(self.root, "pz1/"+name)   
            #gt_file = osp.join(self.root, "gt/"+name)#ground truth
            self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "gt": gt_file,
                    "name": name
                })
           

    def _rotation(self,img,gt,label):
        
        index = random.randint(1, 6)
        if index == 1:
            new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            new_gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            new_label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
            
        elif index ==2:
            new_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif index == 3:
            new_img = cv2.rotate(img, cv2.ROTATE_180)
            new_gt = cv2.rotate(gt, cv2.ROTATE_180)
            new_label = cv2.rotate(label, cv2.ROTATE_180)

        elif index == 4:
            new_img = cv2.flip(img,1)
            new_gt = cv2.flip(gt,1)
            new_label = cv2.flip(label,1)

        elif index==5:
            new_img = cv2.flip(img,0)
            new_gt = cv2.flip(gt,0)
            new_label = cv2.flip(label,0)   
        else:
            new_img = img
            new_gt = gt
            new_label = label

        return new_img,new_gt,new_label


    def _augment_strong(self, img,mask):
        transformer = A.Compose([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10,val_shift_limit=10, p=0.3),#色调饱和度值
            A.RandomBrightnessContrast(brightness_limit=0.1,contrast_limit=0.1, p=0.2), 
            A.ChannelShuffle(blur_limit=3, p=1)
            ])#平移缩放旋转

        augmented = transformer(image=img, mask=mask)

        return augmented['image'], augmented.get('mask', None)

    def make_clslabel(self, label,num_classes=5,ingore_index=255):
        label_set = np.unique(label)
        label_set = label_set.astype(int)
        cls_label = np.zeros(num_classes)
        cls_label = cls_label.astype(int)
        for i in label_set:
            if i < ingore_index:
                cls_label[i] += 1
        return cls_label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('P')
        gt = Image.open(datafiles["gt"]).convert('P')
            
        image1,label1,gt1 = random_crop(image,label,gt,self.crop_size)

        if (image,label) == (0,0):
            return (0,0)

        label1 = np.asarray(label1, np.float32)
        image1 = np.asarray(image1)
        gt1 = np.asarray(gt1,np.float32)

        label1[label1==255]=1
        gt1[gt1==255]=1
                
        image_aug,gt_aug,label_aug  = self._rotation(image1,gt1,label1)             

        #image_aug = cv2.resize(image_aug,(256,256),interpolation=cv2.INTER_CUBIC)      
        #label_aug = cv2.resize(label_aug,(256,256),interpolation=cv2.INTER_NEAREST)    
        #gt_aug = cv2.resize(gt_aug,(256,256),interpolation=cv2.INTER_NEAREST) 

        image_aug1,_ = self._augment_strong(image_aug,label_aug)
        
        image_aug = tf.to_tensor(image_aug)
        image_aug = image_aug.numpy()

        image_aug1 = tf.to_tensor(image_aug1)
        image_aug1 = image_aug1.numpy()
        
        #gt_aug = torch.from_numpy(np.expand_dims(gt_aug, 0)).float()
        return image_aug.copy(),image_aug1.copy(),label_aug.copy(),gt_aug.copy()
#test  dataset
class ISPRSTestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None,crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False, ignore_label=255,set='train',mode=0):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.mode = mode
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []


        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s" % name.replace('mask.png','sat.jpg'))
            #img_file = osp.join(self.root, "img/%s" % name)
            gt_file = osp.join(self.root, "gt/"+name) #.replace('.tif','.png')
            self.files.append({
                    "img": img_file,
                    "gt": gt_file,
                    "name": name
                })
           
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        gt = Image.open(datafiles["gt"]).convert('P')
            
        name = datafiles["name"]
        image = np.asarray(image)
        gt = np.asarray(gt,np.float32)

        gt[gt==255]=1
        
        image = tf.to_tensor(image)
        image = image.numpy()
  
        return image.copy(), gt.copy(),name
