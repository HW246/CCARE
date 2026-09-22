import argparse
import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm 
from skimage.io import imread, imsave
import torchvision.transforms.functional as TF
from osgeo import gdal
from osgeo.gdalconst import GDT_Byte, GDT_Float32, GDT_UInt16
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from collections import OrderedDict
from network.CLCFormer_model import CLCFormer
from utils.PAR import PAR
from model_weak.models_DBF import DBFNet
from model_weak.models_agmm.semseg.deeplabv3plus import DeepLabV3Plus
#from scribeFormer.scribformer import ScribFormer
from model_weak.Networks_crgnet import BaseNet
import torch.nn as nn
#from wetr.model_attn_aff import WeTr_point
from core.networks import  MTNet
from network.SparseFormer_model import SparseFormer
from models.CMTFNet import CMTFNet
from models.ResNet import *
from models.UNetformer import UNetFormer
from models.fcn import FCN8s
def main():

    data_root = 'J:/aaaaaaaa/exp/code1/data/ottawa/img/'
    label_root = './data/deep/fs/'
    ckpt = r'J:\aaaaaaaa\exp\code1\Result_ottawa\time0828_1110\ottawa_SAM_CMTFNet_batch2mF1_8547.pth'
    txt = r'J:\aaaaaaaa\exp\code1\data\ottawa\test.txt'
    file = open(txt, 'r')

    num_classes=2
    save_root = 'J:/aaaaaaaa/exp/code1/ottawa_pred/ATSG/'

    img_paths = list(file)
    #img_paths = os.listdir(data_root)
    
    if os.path.exists(save_root)==False:
        os.makedirs(save_root)
    
    cudnn.enabled = True
    cudnn.benchmark = True
           
    model = CMTFNet(encode_channels=[256, 512, 1024, 2048],decode_channels=512,dropout=0.1,num_classes=2,backbone=ResNet50).cuda()    
    #model = FCN8s(nclass=2).cuda() 
    '''
    model = UNetFormer(
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=2).cuda()
    '''
    #model = SparseFormer(num_classes=5, drop_rate=0.4, normal_init=True, pretrained=True,attn1='CA',attn2='CABM').cuda()
    state_dict1 =torch.load(ckpt)
    model.load_state_dict(state_dict1,strict=False)
    model.eval()

    block_size =512,512
    min_overlap = 100

    #img_paths = os.listdir(data_root)
    #img_paths = img_paths[10:]

    for img_path in tqdm(img_paths):
        torch.cuda.empty_cache()
        img_path = img_path.replace('\n','')
        #img_path = img_path.replace('tif','jpg')
        
        RGBimg=gdal.Open(str(data_root+img_path).replace('mask.png','sat.jpg'))
        band=RGBimg.GetRasterBand(1)  
                                              
        image = imread(str(data_root+img_path).replace('mask.png','sat.jpg'))
        #label = imread(str(label_root+img_path))
        #image = np.asarray(image, np.float32)
        image_np = np.array(image,dtype='uint8') 
        image_size = image.shape[0:2]
        #label = torch.from_numpy(label).cuda().unsqueeze(0)
        h,w = image.shape[0],image.shape[1]
        if h<512 or w<512:
            continue
        else:
            y_end,x_end = np.subtract(image_size, block_size)
            x = np.linspace(0, x_end, int(np.ceil(x_end/np.float64(block_size[1]-min_overlap)))+1, endpoint=True).astype('int')
            y = np.linspace(0, y_end, int(np.ceil(y_end/np.float64(block_size[0]-min_overlap)))+1, endpoint=True).astype('int')
            
            test_pred = np.zeros(image_size) 
            #image = torch.from_numpy(image).unsqueeze(0).cuda()
            image=TF.to_tensor(image).cuda().unsqueeze(0)
            for j in range(len(x)):    
                for k in range(len(y)):            
                    r_start,c_start = (y[k],x[j])
                    r_end,c_end = (r_start+block_size[0],c_start+block_size[1])               
                    image_part = image[0,:,r_start:r_end, c_start:c_end].unsqueeze(0).cuda()
                    
                    with torch.no_grad():    
                        output1 = model(image_part)
                        #pred = ((torch.sigmoid(output1)>0.5)*1).data.cpu().numpy() 
                        #test_pred1 = test_pred1*1
                        pred = torch.argmax(output1,1).squeeze(0).cpu().numpy()   
                        #pred = pred[0,0]
                    if (j==0)and(k==0):
                        test_pred[r_start:r_end, c_start:c_end] = pred
                    elif (j==0)and(k!=0):
                        test_pred[r_start+int(min_overlap/2):r_end, c_start:c_end] = pred[int(min_overlap/2):,:]
                    elif (j!=0)and(k==0):
                        test_pred[r_start:r_end, c_start+int(min_overlap/2):c_end] = pred[:,int(min_overlap/2):]
                    elif (j!=0)and(k!=0):
                        test_pred[r_start+int(min_overlap/2):r_end, c_start+int(min_overlap/2):c_end] = pred[int(min_overlap/2):,int(min_overlap/2):]

            test_pred = np.asarray(test_pred, dtype=np.uint8)

            test_pred[test_pred==1]=255

            cv2.imwrite(save_root+img_path.replace('jpg','png'), test_pred)
            
if __name__ == '__main__':
    main()
