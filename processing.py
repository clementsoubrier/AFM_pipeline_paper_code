#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:18:29 2022

@author: c.soubrier
"""

#%%  Imports


import os
import datetime
import cv2
import numpy as np
from numba import njit
import tqdm
import re
import copy
from multiprocessing import Pool

from cellpose import utils, io, models, denoise

import matplotlib.pyplot as plt
from shutil import rmtree #erasing a whole directory
from skimage.morphology import skeletonize #,medial_axis
from pySPM import Bruker,SPM_image
from  Centerlines.complete_centerlines import  complete_one_centerline
from scaled_parameters import get_scaled_parameters




''' Paths of the data and to save the results'''


# =============================================================================
# #Inputs
# #directory of the original dataset composed of a sequence of following pictures of the same bacterias, and with log files with .001 or no extension
# Main_directory=   #the directory you chos to work on
# dir_name=Main_directory+"Height/"             #name of dir
# my_data = "../data/"+dir_name       #path of dir
# #directory with the usefull information of the logs, None if there is no.
# data_log="../data/"+Main_directory+"log/"             #       #path and name
# 
# #Temporary directories
# #directory of cropped data each image of the dataset is cropped to erase the white frame
# cropped_data=dir_name+"cropped_data/" 
# #directory with the usefull info extracted from the logs
# cropped_log=dir_name+"log_cropped_data/" 
# #directory for output data from cellpose 
# segments_path = dir_name+"cellpose_output/"
# 
# 
# #Outputs
# #directory of the processed images (every image has the same pixel size and same zoom)
# final_data=dir_name+"final_data/" 
# #dictionnary and dimension directory
# Dic_dir=dir_name+"Dic_dir/" 
# #Saving path for the dictionnary
# saving_path=Dic_dir+'Main_dictionnary'
# #dimension and scale of all the final data
# dimension_data=Dic_dir+'Dimension'
# =============================================================================











''' main dictionnary main_dict: each image contain a dictionnary with
time : the time of the image beginning at 0
adress : location of the cropped file
masks : the masks as numpy array
resolution : physical resolution of a pixel
outlines : the outlines as a list of points
angle: rotation angle of the image compared to a fixed reference
centroid : an array containing the position of the centroid of each mask the centroid[i] is the centroid of the mask caracterized by i+1
area : an array containing the position of the area (number of pixels) of each mask
mask_error : True if cellpose computes no mask, else False
# convex_hull : convex hull of each mask
main_centroid : the centroid of all masks
parent / child : previous / next acceptable  file
mask_list : index of the mask in the masks
centerlines :
vertical_resolution :
'''



#%% Preparation of the data (sorting, resizing). Here data is a log file with infos and pictures of the cell (as np float arrays).


def data_prep_logs_only(log_dir,dir_im,temp_dir_info, thres_scars_direc):
    '''
    

    Parameters
    ----------
    log_dir : TYPE
        DESCRIPTION.
    dir_im : TYPE
        DESCRIPTION.
    temp_dir_info : TYPE
        DESCRIPTION.
    thres_scars_direc : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Load up a list of log files (name as strings) from our example data. Extracting and saving the data
    files = os.listdir(log_dir)
    
    # Sort files by timepoint.
    files.sort(key = natural_keys)    
    
    #clean or create the cropped directory
    if os.path.exists(dir_im):
        for file in os.listdir(dir_im):
            os.remove(os.path.join(dir_im, file))
    else:
        os.makedirs(dir_im)
    #clean or create the cropped directory
    if os.path.exists(temp_dir_info):
        for file in os.listdir(temp_dir_info):
            os.remove(os.path.join(temp_dir_info, file))
    else:
        os.makedirs(temp_dir_info)
        
        
    for i in tqdm.trange(len(files)):
        if  files[i][0]!='.':         #avoiding .DS_Store files
            try:                    #SOMETIMES pySPM.Bruker function crashes if the aspect ratios are not integers
                #loading and saving the data-files and dimension
                Scan = Bruker(os.path.join(log_dir, files[i]))
                channel_list=get_channel_list(Scan)
                back_for=np.zeros(len(channel_list),dtype=int)
                for j in range(1,len(channel_list)):
                    back_for[j]=int(channel_list[j] in channel_list[:j-1])
                subdic={}
                unitdic={}
                for j in range(len( channel_list)):
                    channel=channel_list[j]
                    backward=back_for[j]
                    if 'Height' in channel:
                        name='Height'
                    else:
                        name=channel
                    try:
                        data_array=get_channel(Scan,channel,lazy=False,backward=(1-backward))
                    except:
                        data_array=get_channel(Scan,channel,lazy=False,backward=backward)
                    if name in thres_scars_direc.keys():
                        image=filter_scars_removal(data_array.pixels[::-1,:], thres_scars_direc[name])
                    else:
                        image=filter_scars_removal(data_array.pixels[::-1,:], thres_scars_direc['other'])
                    
                    
                    name=name+'_bwd'*backward+'_fwd'*(1-backward)
                    
                              #filtering the scars on the image
                    subdic[name]=image
                    unitdic[name]=str(data_array.zscale)
                
                filename=re.split(r'\W+',files[i])[0]# taking care of extensions in the filename
                
                newsubdic=realign_bwd_fwd_images(subdic)  #aligning backward and forward pictures
                np.savez_compressed(os.path.join(dir_im,filename),newsubdic)
                
                
                #extracting the angle
                text_file = open(os.path.join(log_dir, files[i]), "r", errors='ignore' )
                #read whole file to a string
                text = text_file.read()
                
                #close file
                text_file.close()
                
                # get rid of the linebreaks
                text=re.sub(r'\n', ' ', text)
                
                match1=re.match(r"^.*Rotate Ang(.*?|\n)\\.*$",text).group(1)
                match1=re.findall(r'\d+\.\d+|\d+',match1)
                
                if data_array.size['real']['unit']=='um':
                    np.save(os.path.join(temp_dir_info, filename + 'dimp.npy'),np.array([data_array.size['real']['y'],data_array.size['real']['x'],np.shape(data_array.pixels)[0],np.shape(data_array.pixels)[1],float(match1[0])])) #format : physical y len, physical x len, pixel number y, pixel number x, rotation angle
                elif data_array.size['real']['unit']=='nm':
                    np.save(os.path.join(temp_dir_info, filename + 'dimp.npy'),np.array([data_array.size['real']['y']/1000,data_array.size['real']['x']/1000,np.shape(data_array.pixels)[0],np.shape(data_array.pixels)[1],float(match1[0])]))
                else: 
                    print('error unit format', data_array.size['real']['unit'])
                np.save(os.path.join(temp_dir_info, filename + 'unit.npy'),unitdic,allow_pickle=True)
            
            except:
                print(files[i],'error in log parameters')
                continue



def realign_bwd_fwd_images(subdic):
    # aligning the forward and backward pictures to counter piezo sensor hysteresis
    if "Height_bwd" in subdic.keys():
        fwd_img=subdic["Height_fwd"].astype(np.float32)
        
        bwd_img=subdic["Height_bwd"].astype(np.float32)
        init_mat=np.eye(2,M=3, dtype=np.float32)
        warpMatrix=cv2.findTransformECC(bwd_img,fwd_img,init_mat,motionType=0)[1]
        
        for channel in subdic.keys():
            if channel[-3:]=='bwd':
                subdic[channel]=cv2.warpAffine(subdic[channel],warpMatrix,np.shape(subdic[channel])[::-1])
    return subdic
    
    
        
def natural_keys(text):
    '''
    tool function to sort files by timepoints

    Parameters
    ----------
    text : string
        
        

    Returns
    -------
    list
        list of the digits

    '''
    return [ int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text) ]


        
def filter_scars_removal(image, thresh):
    '''

    Filter function to remove scars from images.
    adapted from pySPM library


    Parameters
    ----------
    image : numpy array
        AFM image to improve, as a 1 channel image
    thresh : TYPE
        threshold for yhe scars detection

    Returns
    -------
    image : numpy array
        filtered image

    '''
    
    for y in range(1, image.shape[0] - 1):
        b = image[y - 1, :]
        c = image[y, :]
        a = image[y + 1, :]
        mask = np.abs(b - a) < thresh * (np.abs(c - a))
        image[y, mask] = b[mask]
    return image  
        


def correct_plane(img, mask=None):
    '''
    

    Parameters
    ----------
    img : numpy array
        1 channel image
    mask : None or 2D numpy array. 
        If not None define on which pixels the data should be taken.

    Returns
    -------
    img : numpy array
        1 channel image

    '''
    
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    X0, Y0 = np.meshgrid(x, y)
    Z0 = img
    if mask is not None:
        X = X0[mask]
        Y = Y0[mask]
        Z = Z0[mask]
    else:
        X = X0
        Y = Y0
        Z = Z0
    A = np.column_stack((np.ones(Z.ravel().size), X.ravel(), Y.ravel()))
    c, resid, rank, sigma = np.linalg.lstsq(A, Z.ravel(), rcond=-1)

    img -= c[0] * np.ones(img.shape) + c[1] * X0 + c[2] * Y0 #
    return img
       
    

def get_channel_list(elem,encoding='latin1'):
    """
    extracting the list of channels of a bruker SPM image
    adapted from pySPM library

    """
    channellist=[]
    for i in range(len(elem.layers)):
            layer_name = elem.layers[i][b'@2:Image Data'][0].decode(encoding)
            result = re.match( r'([^ ]+) \[([^\]]*)\] "([^"]*)"', layer_name).groups()
            channellist.append(result[2])
    return np.array(channellist,dtype=str)



#copy of function in Bruker (fixed errors concerning certain log files)
def get_channel(self, channel="Height Sensor", backward=False, corr=None, debug=False, encoding='latin1',
                lazy=True):
    """
    Load the SPM image contained in a channel
    adapted from pySPM library
    """
    for i in range(len(self.layers)):
        layer_name = self.layers[i][b'@2:Image Data'][0].decode(encoding)
        result = re.match(
            r'([^ ]+) \[([^\]]*)\] "([^"]*)"', layer_name).groups()
        if result[2] == channel:
            if debug:
                print("channel " + channel + " Found!")
            bck = False
            try:
                if self.layers[i][b'Line Direction'][0] == b'Retrace':
                    bck = True
            except:
                if self.layers[i][b'Line direction'][0] == b'Retrace':
                    bck = True
            if bck == backward:
                if debug:
                    print("Direction found")
                var = self.layers[i][b'@2:Z scale'][0].decode(encoding)
                if debug:
                    print("@2:Z scale", var)
                if '[' in var:
                    result = re.match(r'[A-Z]+\s+\[([^\]]+)\]\s+\(-?[0-9\.]+ .*?\)\s+(-?[0-9\.]+)\s+(.*?)$',
                                      var).groups()
                    if debug:
                        print(result)
                    bpp = int(self.layers[i][b'Bytes/pixel'][0])
                    if debug:
                        print("BPP", bpp)
                    # scale = float(result[1])
                    scale = float(result[1]) / 256 ** bpp

                    result2 = self.scanners[0][b'@' + result[0].encode(encoding)][0].split()
                    if debug:
                        print("result2", result2)
                    scale2 = float(result2[1])
                    if len(result2) > 2:
                        zscale = result2[2]
                    else:
                        zscale = result2[0]
                    if b'/V' in zscale:
                        zscale = zscale.replace(b'/V', b'')
                    if debug:
                        print("scale: {:.3e}".format(scale))
                        print("scale2: {:.3e}".format(scale2))
                        print("zscale: " + str(zscale))
                    var = self.layers[i][b'@2:Z offset'][0].decode(encoding)
                    result = re.match(r'[A-Z]+\s+\[[^\]]+\]\s+\(-?[0-9\.]+ .*?\)\s+(-?[0-9\.]+)\s+.*?$',
                                      var).groups()
                    offset = float(result[0])
                else:
                    if debug:
                        print("mode 2")
                    result = re.match(r'[A-Z]+ \(-?[0-9\.]+ [^\)]+\)\s+(-?[0-9\.]+) [\w]+', var).groups()
                    scale = float(result[0]) / 65536.0
                    scale2 = 1
                    zscale = b'V'
                    result = re.match(r'[A-Z]+ \(-?[0-9\.]+ .*?\)\s+(-?[0-9\.]+) .*?',
                                      self.layers[i][b'@2:Z offset'][0].decode(encoding)).groups()
                    offset = float(result[0])
                if debug:
                    print("Offset:", offset)
                data = self._get_raw_layer(i, debug=debug) * scale * scale2
                xres = int(self.layers[i][b"Samps/line"][0])
                yres = int(self.layers[i][b"Number of lines"][0])
                if debug:
                    print("xres/yres", xres, yres)
                try:
                    scan_size = self.layers[i][b'Scan Size'][0].split()
                except:
                    scan_size = self.layers[i][b'Scan size'][0].split()
                try :
                    aspect_ratio = [int(x) for x in self.layers[i][b'Aspect Ratio'][0].split(b":")]
                except:
                    aspect_ratio = [int(x) for x in self.layers[i][b'Aspect ratio'][0].split(b":")]
                if debug:
                    print("aspect ratio", aspect_ratio)
                if scan_size[2][0] == 126:
                    scan_size[2] = b'u' + scan_size[2][1:]
                size = {
                    'x': float(scan_size[0]) / aspect_ratio[1],
                    'y': float(scan_size[1]) / aspect_ratio[0],
                    'unit': scan_size[2].decode(encoding)}
                image = SPM_image(
                    channel=[channel, 'Topography'][channel == 'Height Sensor'],
                    BIN=data,
                    real=size,
                    _type='Bruker AFM',
                    zscale=zscale.decode(encoding),
                    corr=corr)
                return image
    if lazy:
        return self.get_channel(channel=channel, backward=not backward, corr=corr, debug=debug, encoding=encoding,
                                lazy=False)
    raise Exception("Channel {} not found".format(channel))
    
#%%Preparation of the data so that each image has the same dimension 
def dimension_def_logs_only(dir_im, temp_dir_info, dimensiondata, maxsize): #dir of the images, dir of the numpy info of logs

        
    #determining the main dimension and the best vertical/horizontal precision
    phys_dim=[]
    for file in os.listdir(temp_dir_info):
        if file[-8:]=='dimp.npy':
            log_para=np.load(os.path.join(temp_dir_info, file))
            if log_para[0]==0 or log_para[1]==0:
                raise TypeError('Error in file '+file+', please exculde of dataset')
            phys_dim.append([log_para[0],log_para[1],log_para[0]/log_para[2],log_para[1]/log_para[3]])#format : physical y len, physical x len, resolution y, resolution x
    phys_dim=np.array(phys_dim)
    prec=np.min(phys_dim[:,2:4][phys_dim[:,2:4]>0])
    dim_height=int(np.max(phys_dim[:,0])/prec)
    dim_len=int(np.max(phys_dim[:,1])/prec)

    #avoiding too big pictures with a max size
    ratio=0
    if dim_height>=dim_len and dim_height>maxsize:
        ratio=maxsize/dim_height
    elif dim_len > dim_height and dim_len>maxsize:
        ratio=maxsize/dim_len
    
    # scailing all the images to get same resolution and dimension
    listdir=os.listdir(dir_im)
    for i in tqdm.trange(len(listdir)):
        filez=listdir[i]
        datadir=np.load(os.path.join(dir_im, filez),allow_pickle=True)['arr_0'].item()
        
        for channel in datadir.keys():
            log_para=np.load(os.path.join(temp_dir_info, filez[:-4] + 'dimp.npy'))
            if not 'Error' in channel:
                
                newimg=datadir[channel]
                minimg=np.min(newimg)
                newimg=newimg-minimg*np.ones(np.shape(newimg))
                
            else :
                newimg=datadir[channel]
                
            newimg=bili_scale_image(newimg,log_para[0]/log_para[2],log_para[1]/log_para[3],dim_height,dim_len,prec,prec)
            if ratio>0:
                newimg=cv2.resize(newimg,(int(dim_len*ratio),int(dim_height*ratio)), interpolation=cv2.INTER_CUBIC)

            datadir[channel]=newimg

        np.savez_compressed(os.path.join(dir_im, filez), datadir)
    if ratio==0:
        np.save(os.path.join(temp_dir_info, dimensiondata),np.array([dim_height,dim_len,prec]))#saving dimensions of the images and physical dimension of a pixel for future plots
    else :
        np.save(os.path.join(temp_dir_info, dimensiondata),np.array([int(dim_height*ratio),int(dim_len*ratio),prec/ratio]))
    



    

#%% Running cellpose and saving the results
def run_cel_logs_only(dir_im,seg,temp_dir_info,dimensiondata, pref_channel=None):
    params=get_scaled_parameters(cellpose=True)

    mod = params['cel_model_type']
    denoise_mod = params['cel_denoise_model']
    chan = params['cel_channels']
    param_dia = params['cel_diameter_param']
    thres = params['cel_flow_threshold']
    celp = params['cel_prob_threshold']
    gpuval = params['cel_gpu']

    #clean or create the output directory
    if os.path.exists(seg):
        for file in os.listdir(seg):
            os.remove(os.path.join(seg, file))
    else:
        os.makedirs(seg)
        
    # computation of the expected cell diameter in pixels
    scale=np.load(os.path.join(temp_dir_info, dimensiondata))[2]
    dia=2/scale*param_dia
    
    # Loop over all of our image files and run Cellpose on each of them. 
    listdir=os.listdir(dir_im)
    for i in tqdm.trange(len(listdir)):
        filez=listdir[i]
        log_para=np.load(os.path.join(temp_dir_info, filez[:-4] + 'dimp.npy'))
        dimi=int(log_para[0])/scale
        dimj=int(log_para[1])/scale
        datadir=np.load(os.path.join(dir_im, filez),allow_pickle=True)['arr_0'].item()
        
        keys=datadir.keys()
        if pref_channel is not None:
            img=datadir[pref_channel]
            print( f'Using {pref_channel} for segmentation')
        elif "Height_fwd" in keys and "Peak Force Error_fwd" in keys and "Peak Force Error_bwd" in keys:
            img=renorm_img(datadir["Height_fwd"])/3*2+renorm_img(np.maximum(datadir["Peak Force Error_fwd"],np.zeros(np.shape(datadir["Peak Force Error_fwd"])))+np.maximum(datadir["Peak Force Error_bwd"],np.zeros(np.shape(datadir["Peak Force Error_bwd"]))))/3
        elif "Height_fwd" in keys and "Peak Force Error_fwd" in keys:
            img=renorm_img(datadir["Height_fwd"])/3*2+renorm_img(datadir["Peak Force Error_fwd"])/3
        elif "Height_fwd" in keys:
            img=datadir["Height_fwd"]
        else: 
            print('No Height profile in '+str(filez))
        
        
        if gpuval:
            try:
                # Specify that the cytoplasm Cellpose model for the segmentation. 
                if denoise_mod is None:
                    model = models.Cellpose(gpu=True, model_type=mod)
                else:
                    model = denoise.CellposeDenoiseModel(gpu=True, model_type=mod, restore_type=denoise_mod)
                masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
            except:
                if denoise_mod is None:
                    model = models.Cellpose(gpu=False, model_type=mod)
                else:
                    model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
                masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)
        else :
            if denoise_mod is None:
                model = models.Cellpose(gpu=False, model_type=mod)
            else:
                model = denoise.CellposeDenoiseModel(gpu=False, model_type=mod, restore_type=denoise_mod)
            masks, flows, st, diams = model.eval(img, diameter = dia, channels=chan, flow_threshold = thres, cellprob_threshold=celp)

        # save results
        io.masks_flows_to_seg(img, masks, flows, os.path.join(seg, filez[:-4]), channels=chan, diams=diams)
        
        
        
        #flattening the Height background using the masks
        for channel in keys:
            if channel=="Height_fwd" or channel=="Height_bwd":
                maskheight=np.logical_not(masks)
                dim1,dim2=np.shape(maskheight)
                maskheight=np.logical_and(maskheight,falseframe(dim1,dim2,dimi,dimj))
                goodimg=correct_plane(datadir[channel], mask=maskheight)
                if np.max(falseframe(dim1,dim2,dimi,dimj))==0:
                    cst_val=np.max(datadir[channel])
                elif np.min(falseframe(dim1,dim2,dimi,dimj))!=1:
                    cst_val=min(np.max(datadir[channel][np.logical_not(falseframe(dim1,dim2,dimi,dimj))]),np.min(datadir[channel][falseframe(dim1,dim2,dimi,dimj)]))
                else:
                    cst_val=np.min(datadir[channel][falseframe(dim1,dim2,dimi,dimj)])
                goodimg[np.logical_not(falseframe(dim1,dim2,dimi,dimj))]=cst_val*np.ones((dim1,dim2))[np.logical_not(falseframe(dim1,dim2,dimi,dimj))]
                datadir[channel]=goodimg

        np.savez_compressed(os.path.join(dir_im, filez),**datadir)

@njit    
def renorm_img(img):
    dim1,dim2=np.shape(img)
    newimg=np.zeros((dim1,dim2))
    maxi=np.max(img)
    mini=np.min(img)
    if maxi==mini:
        return newimg
    else:
        for i in range(dim1):
            for j in range(dim2):
                newimg[i,j]=(img[i,j]-mini)/(maxi-mini)
        return newimg

@njit 
def falseframe(dim1,dim2,dimi,dimj):
    img=np.ones((dim1,dim2),dtype=np.bool_)
    for i in range(dim1):
        for j in range(dim2):
            if i<=(dim1-dimi)//2 or i>=dimi+(dim1-dimi)//2 or j<=(dim2-dimj)//2 or j>=dimj+(dim2-dimj)//2:
                img[i,j]=False
    return img

#%% Creating the final dictionnary with all the properties of the images
def download_dict_logs_only(dir_im,temp_dir_info,segmentspath,dimensiondata,year,saving=False ,savingpath='dict'):
    
    
    files = os.listdir(dir_im)
    dic={}
    dimen=np.load(os.path.join(temp_dir_info, dimensiondata))
    
    
    # Sort files by timepoint.
    files.sort(key = natural_keys)
    init_time=datetime.datetime(int(year), int(files[0][:2]), int(files[0][2:4]), int(files[0][4:6]), int(files[0][6:8]), 0, 0)
    for fichier in files:
        fichier=fichier[:-4]
        dat = np.load(os.path.join(segmentspath, fichier+'_seg.npy'), allow_pickle=True).item()
        if np.max(dat['masks'])!=0:

            dic[fichier]={}
            
            time=datetime.datetime(int(year), int(fichier[:2]), int(fichier[2:4]), int(fichier[4:6]), int(fichier[6:8]), 0, 0)
            t=(time-init_time).total_seconds()
            
            dic[fichier]['time']=int(t/60)      #time in minutes after the beginning of the experiment
            dic[fichier]['adress']=os.path.join(dir_im,fichier+'.npz')
            dic[fichier]['masks']=dat['masks']
            dic[fichier]['outlines']=utils.outlines_list(dat['masks'],multiprocessing=False)
            dic[fichier]['angle']=np.load(os.path.join(temp_dir_info, fichier + 'dimp.npy'))[-1]
            dic[fichier]['resolution']=dimen[2]
            dic[fichier]['units']=np.load(os.path.join(temp_dir_info, fichier + 'unit.npy'), allow_pickle=True).item()
            
    #deleting temporary dir
    rmtree(segmentspath)
    rmtree(temp_dir_info)
    if saving:
        np.savez_compressed(savingpath,dic)
    return dic


    
# scaling algorithm     
@njit
def bili_scale_image(img,preci_h,preci_l,dim_h,dim_l,precf_h,precf_l): #scailing an image with vertical resolution preci_h and horizontal resolution preci_l into an image with resolution precf_h,precf_l and shape (dim_h,dim_l)
    
    
    newimage=np.zeros((dim_h,dim_l))
    dim_img_h,dim_img_l=np.shape(img)
    ratio_h=preci_h/precf_h
    ratio_l=preci_l/precf_l
    dim_zoom_h=int(ratio_h*dim_img_h)
    dim_zoom_l=int(ratio_l*dim_img_l)
    drift_h=int((dim_h-dim_zoom_h)/2)
    drift_l=int((dim_l-dim_zoom_l)/2)
    for i in range(dim_zoom_h-1):
        for j in range(dim_zoom_l-1):
            
            ih1=int((i/ratio_h))
            ih2=int(((i+1)/ratio_h+precf_h))
            
            if ih1==ih2:
                frac_h=1
            else:
                frac_h=(i/ratio_h-ih1)
            jl1=int((j/ratio_l))
            jl2=int(((j+1)/ratio_l))
            if jl1==jl2:
                frac_l=1
            else:
                frac_h=(j/ratio_l-jl1)
            
            res=frac_h*frac_l*img[ih1,jl1]+(1-frac_h)*frac_l*img[ih2,jl1]+frac_h*(1-frac_l)*img[ih1,jl2]+(1-frac_h)*(1-frac_l)*img[ih2,jl2]
            newimage[i+drift_h,j+drift_l]=res
    return(newimage)




#%% Computing the area and the centroid of each mask, saving them in the dictionnary and creating child-parent link
def centroid_area(dic):
    diclist=list(dic.keys())
    for i in tqdm.trange(len(diclist)):
        fichier=diclist[i]
        masks=dic[fichier]['masks']
        mask_number=np.max(masks)
        if mask_number>0:
            (centroid,area)=numba_centroid_area(masks,mask_number)
            dic[fichier]['centroid']=centroid
            dic[fichier]['area']=area
        else:
            del dic[fichier]

#to improve computation speed
@njit
def numba_centroid_area(masks,mask_number):
    centroid=np.zeros((mask_number,2),dtype=np.int32)
    area=np.zeros(mask_number,dtype=np.int32)
    (l,L)=np.shape(masks)
    for i in range(mask_number):
        count=0
        vec1=0
        vec2=0
        for j in range(L):
            for k in range(l):
                if masks[k,j]==i+1:
                    vec1+=j
                    vec2+=k
                    count+=1
        area[i]=count
        centroid[i,:]=np.array([vec2//count,vec1//count],dtype=np.int32)
    return(centroid,area)


def main_parenting(dic):
    parent=''
    list_key=list(dic.keys())
    for fichier in list_key:
            dic[fichier]['parent']=parent
            parent=fichier
    child=''
    key=list(dic.keys())
    key.reverse()
    for fichier in key:
            dic[fichier]['child']=child
            child=fichier


#%% Erasing too small masks (less than the fraction frac_mask of the largest mask), and mask with a ratio of saturated area superior to frac_satur . Creating the centroid of the union of acceptable  mask and saving as main_centroid
def clean_masks(frac_mask,frac_satur,dic,saving=False,savingpath='dict'): 
    #Erase the masks that are too small (and the centroids too)
    diclist=list(dic.keys())
    for j in tqdm.trange(len(diclist)):
        fichier=diclist[j]
        masks=dic[fichier]['masks']
        img=np.load(dic[fichier]['adress'])['Height_fwd']
        area=dic[fichier]['area']
        centroid=dic[fichier]['centroid']
        outlines=dic[fichier]['outlines']
        
        
        max_area=np.max(area)
        L=len(area)
        non_saturated=np.zeros(L) #detection of the non saturated masks
        for i in range(L):
            if satur_area(img,masks,i+1)<frac_satur*area[i]:
                non_saturated[i]=1
        
        max_area=max([area[i]*non_saturated[i] for i in range(L)])
        non_defect=np.zeros(L) #classification of the allowed masks
        non_defect_count=0
        newoutlines=[]
        
        
        for i in range(L):
            if area[i]>=frac_mask*max_area and non_saturated[i]==1:
                non_defect_count+=1
                non_defect[i]=non_defect_count
                newoutlines.append(outlines[i][:,::-1].astype(np.int32))

        if non_defect_count==0:
            del dic[fichier]
        else:       
            #update the outlines
            dic[fichier]['outlines']=newoutlines
            #new value of the area and the centroid
            area2=np.zeros(non_defect_count).astype(np.int32)
            centroid2=np.zeros((non_defect_count,2),dtype=np.int32)
            for i in range(L):
                if non_defect[i]!=0:
                    area2[int(non_defect[i]-1)]=area[i]
                    centroid2[int(non_defect[i]-1),:]=centroid[i,:]
            (m,n)=masks.shape
            for i in range(m):
                for k in range(n):
                    if masks[i,k]!=0:
                        masks[i,k]=non_defect[masks[i,k]-1]
            
            dic[fichier]['area']=area2
            dic[fichier]['centroid']=centroid2
            #constructing the main centroid
            main_centroid0=0
            main_centroid1=0
            for i in range (non_defect_count):
                main_centroid0+=area2[i]*centroid2[i,0]
                main_centroid1+=area2[i]*centroid2[i,1]
            
            dic[fichier]['main_centroid']=np.array([main_centroid0//sum(area2),main_centroid1//sum(area2)],dtype=np.int32)

    if saving:
        np.savez_compressed(savingpath,dic)


@njit
def satur_area(img,masks,number,thres=0.95):
    score=-np.inf
    non_zero=np.nonzero(masks==number)
    tot_sum=0
    count=0
    for i,j in zip(non_zero[0],non_zero[1]):
        tot_sum+=img[i,j]
        count+=1
        if img[i,j]>score:          #selecting the maximum value in the mask of label number
            score=img[i,j]
    avg_val=tot_sum/count
    area=0
    for i,j in zip(non_zero[0],non_zero[1]):
            if masks[i,j]==number and img[i,j]>=thres*(score-avg_val)+avg_val:
                area+=1
    return area
 
# creating a new mask with changed values
@njit
def update_masks(mask,new_values):
    (l,L)=np.shape(mask)
    for j in range(l):
        for k in range(L):
            if mask[j,k]!=0:
                mask[j,k]=new_values[mask[j,k]-1]
    return mask

#%%Contructing a list containing all the masks of the dataset with this structure : list_index,dataset, frame,mask_index

def construction_mask_list(dic,dataset,listsavingpath):
   index=0
   final_list=[]
   diclist=list(dic.keys())
   for i in tqdm.trange(len(diclist)):
        fichier=diclist[i]
        masks=dic[fichier]['masks']
        mask_number=np.max(masks)
        list_index=[]
        for i in range(mask_number):
            final_list.append([index,dataset,fichier,i+1])
            list_index.append(index)
            index+=1
        dic[fichier]["mask_list"]=list_index
   np.savez_compressed(listsavingpath,np.array(final_list,dtype=object))



#%%     Computing the centerline of each mask and saving them in the dictionnary

#main function to save the centerlines in the dictionnary
def centerline_mask(dic,alpha_erasing_branch,saving=False,savingpath='dict'):

    diclist=list(dic.keys())
    for j in tqdm.trange(len(diclist)):
        fichier=diclist[j]
        
        masks=dic[fichier]['masks']
        mask_number=np.max(masks)
        centerlines=[]
        for i in range(mask_number):
            mask_i=transfo_bin(masks,i+1) #isolating the i+1-th mask
            first_line=construc_center(mask_i,alpha_erasing_branch)
            if len(first_line)>5:          #avoiding too short centerlines
                try:
                    extended_line=complete_one_centerline(first_line, mask_i)
                except:
                    print('error complete centerline ' ,fichier,i)
                    extended_line=first_line
            else:
                extended_line=first_line
            centerlines.append(extended_line)
        dic[fichier]['centerlines']=centerlines
    if saving:
        np.savez_compressed(savingpath,dic,allow_pickle=True)

#isolating the mask with number num
@njit
def transfo_bin(mask,num):
    (dim1,dim2)=np.shape(mask)
    newmask=np.zeros((dim1,dim2),dtype=np.int32)
    for i in range(dim1):
        for j in range(dim2):
            if mask[i,j]==num:
                newmask[i,j]=1
    return newmask

#Constructing the centerline given a unique mask
def construc_center(mask,alpha):
    skel=padskel(mask)              #runing the skeleton algorithm and obtaining a boolean picture
    _,div=find_extremal_division_point(skel)               #extracting the extremal and division points of the skel
    extr_seg,int_seg=division_path_detek(skel)          #EXTRACTING THE DIFFERENT SEGMENTS
    extr_seg=fusion_erasing(extr_seg,div,alpha)          #fusioning/erasing the external segments if needed
    len_int=len(int_seg)
    len_extr=len(extr_seg)
    
    if len_int==0:
        if len_extr==1:
            return np.array(one_D_line(fil_gaps_1D(extr_seg[0]))).astype(np.int32)
        elif len_extr==2:
            compo1=extr_seg[0]
            compo2=extr_seg[1][-1::-1]
            if compo1[-1]==compo2[0]:
                del compo1[-1]
            sol=compo1+compo2
            return np.array(one_D_line(fil_gaps_1D(sol))).astype(np.int32)
        else:
            print("Error skel format,len_int==0,len_extr="+str(len_extr))
            return np.array([[]],dtype=np.int32)
        
        
    elif len_int==1 and len_extr==2:
        compo1=extr_seg[0]
        compo3=extr_seg[1][::-1]
        if max(abs(compo1[-1][0]-int_seg[0][0][0]),abs(compo1[-1][1]-int_seg[0][0][1]))<=3:
            compo2=int_seg[0]
        else:
            compo2=int_seg[0][::-1]
        if compo1[-1]==compo2[0]:
            del compo1[-1]
        if compo2[-1]==compo3[0]:
            del compo2[-1]
        sol=compo1+compo2+compo3
        return np.array(one_D_line(fil_gaps_1D(sol))).astype(np.int32)
    
        
    #we can do the case with 2 internal segments
    elif len_int==2 and len_extr==3:
        int0=int_seg[0]
        int1=int_seg[1]
        
        #selecting the intersection point between the two internal branches
        if max(abs(int0[0][0]-int1[0][0]),abs(int0[0][1]-int1[0][1]))<=3:
            point=int0[0]
            new_int_seg1=int0[-2::-1]
            new_int_seg2=int1
        elif max(abs(int0[0][0]-int1[-1][0]),abs(int0[0][1]-int1[-1][1]))<=3:
            point=int0[0]
            new_int_seg1=int1[:-1]
            new_int_seg2=int0
        elif max(abs(int0[-1][0]-int1[-1][0]),abs(int0[-1][1]-int1[-1][1]))<=3:
            point=int0[-1]
            new_int_seg1=int0
            new_int_seg2=int1[-2::-1]
        else :
            point=int0[-1]
            new_int_seg1=int0
            new_int_seg2=int1[1:]
        #simplifying by erasing the external branch linked to both internal branch
        for elem in extr_seg:
            if max(abs(elem[-1][0]-point[0]),abs(elem[-1][1]-point[1]))<=3:
                extr_seg.remove(point)
        #previous case len_int==1 and len_extr==2
        if max(abs(extr_seg[1][-1][0]-new_int_seg1[0][0]),abs(extr_seg[1][-1][1]-new_int_seg1[0][1]))<=3:
            new_int_seg1,new_int_seg2=copy.deepcopy(new_int_seg2[::-1]),copy.deepcopy(new_int_seg1[::-1])
        if extr_seg[0][-1]==new_int_seg1[0]:
            del new_int_seg1[0]
        if new_int_seg1[-1]==new_int_seg2[0]:
            del new_int_seg2[0]
        if new_int_seg2[-1]==extr_seg[1][-1]:
            del extr_seg[1][-1]
        sol=extr_seg[0]+new_int_seg1+new_int_seg2+extr_seg[1][::-1]
        
        return np.array(one_D_line(fil_gaps_1D(sol))).astype(np.int32)
    else:
        print("Error skel format, len_int="+str(len_int)+"len_extr="+str(len_extr))
        return np.array([[]],dtype=np.int32)
    

#construct an integer line between 2 points in 2 D
def linear_interpolation(point1,point2): 
    len_list= max(np.abs(point1[0]-point2[0]),np.abs(point1[1]-point2[1]))-1
    if len_list<=0:
        return np.zeros(0)
    elif np.abs(point1[0]-point2[0])!=np.abs(point1[1]-point2[1]):
        if point1[0]-point2[0]-1==len_list:
            val1=np.linspace(point1[0]-1,point2[0]+1,len_list,dtype=np.int32)
            val2=np.linspace(point1[1],point2[1],len_list,dtype=np.int32)

        elif point2[0]-point1[0]-1==len_list:
            val1=np.linspace(point1[0]+1,point2[0]-1,len_list,dtype=np.int32)
            val2=np.linspace(point1[1],point2[1],len_list,dtype=np.int32)

        elif point1[1]-point2[1]-1==len_list:
            val1=np.linspace(point1[0],point2[0],len_list,dtype=np.int32)
            val2=np.linspace(point1[1]-1,point2[1]+1,len_list,dtype=np.int32)

        else:
            val1=np.linspace(point1[0],point2[0],len_list,dtype=np.int32)
            val2=np.linspace(point1[1]+1,point2[1]-1,len_list,dtype=np.int32)
    else:
        i=1-2*(point1[0]>point2[0])
        j=1-2*(point1[1]>point2[1])
        val1=np.linspace(point1[0]+i,point2[0]-i,len_list,dtype=np.int32)
        val2=np.linspace(point1[1]+j,point2[1]-j,len_list,dtype=np.int32)
        
    final=np.zeros((len_list,2),dtype=np.int32)
    final[:,0]=val1
    final[:,1]=val2
    return final


def fil_gaps_1D(line):
    if len(line)<2:
        return line
    else:
        oldelem=line[0]
        newseg=[oldelem]
        for j in range(1,len(line)):
            newelem=line[j]
            if oldelem!=newelem:
                if max(abs(oldelem[0]-newelem[0]),abs(oldelem[1]-newelem[1]))==1:
                    newseg.append(newelem)
                    oldelem=newelem
                else:
                    newseg+=list(linear_interpolation(oldelem,newelem))
                    newseg.append(newelem)
                    oldelem=newelem
        return newseg
                     
            
# skeletonization algorithm
def padskel(mask):
    ''' 
    Runs skimage.morphology.skeletonize on a padded version of the mask, to avoid errors.
    
    Parameters
    ----------
    mask = an opencv image
    
    Returns
    -------
    skel = a skeleton (boolean array)
    '''
    mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)#add a border to the skel
    skel = skeletonize(mask,method='lee')              #best skeletonization algorithms
    #skel = skeletonize(mask)           #second skeletonization algorithms
    #skel= medial_axis(mask)            #third skeletonization algorithms
    skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20]
    return skel

# Because the skeleton is 1D, the extremal points have only one neighbour, the division points have 3, we don't take into account the cases with 4 or more neighbour (bad quality skeleton)
def find_extremal_division_point(skel):
    ex=[]
    div=[]
    dim1,dim2=np.shape(skel)
    for i in range(dim1):
        for j in range(dim2):
            if skel[i,j]:
                count=neighbourcount(i,j,dim1,dim2,skel)
                if count==1:
                    ex.append([i,j])
                elif count==3:
                    div.append([i,j])
    return ex,div


#counting the neighbour of a point that are also in the skeleton
@njit
def neighbourcount(i,j,dim1,dim2,skel):
    count=0
    for k in [-1,0,1]:
        for l in [-1,0,1]:
            if (k,l)!=(0,0) and 0<=i+k<dim1 and 0<=j+l<dim2:
                if skel[i+k,j+l]:
                    count+=1
    return count                  

#computing the different segment composing the skeleton : int_seg the list of the internal segment (between two points of div) the list of the external segment (between a points of ext and another point). Each segment is a list of points.
def division_path_detek(skel):#computing the different segment composing the skeleton : int_seg the list of the internal segment (between two points of div) the list of the external segment (between a points of ext and another point)
    ex,div=find_extremal_division_point(skel)
    dim1,dim2=np.shape(skel)
    if len(ex)<=1 :
        return [],[]
    elif len(ex)==2 : #simple case without branching
        old_point=ex[0]
        tot_seg=[old_point]
        for k in [-1,0,1]:
            for l in [-1,0,1]:
                if (k,l)!=(0,0) and 0<=old_point[0]+k<dim1 and 0<=old_point[1]+l<dim2:
                    if skel[old_point[0]+k,old_point[1]+l]:
                        new_point=[old_point[0]+k,old_point[1]+l]
                        tot_seg.append(new_point)
        while new_point not in ex :
            pot_new_point=[]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if (k,l)!=(0,0) and 0<=new_point[0]+k<dim1 and 0<=new_point[1]+l<dim2 and (new_point[0]+k,new_point[1]+l)!=(old_point[0],old_point[1]) and skel[new_point[0]+k,new_point[1]+l]:
                        pot_new_point=[new_point[0]+k,new_point[1]+l]
            if pot_new_point==[]:
                break
            old_point,new_point=new_point,pot_new_point
            tot_seg.append(new_point)         
        return [tot_seg],[]
    
    elif len(ex)==3 :#simple case three connected points
        extr_seg=[] #computation of the external case
        for i in range(3):
            old_point=ex[i]
            skel[old_point[0],old_point[1]]=False
            new_seg=[old_point]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if  0<=old_point[0]+k<dim1 and 0<=old_point[1]+l<dim2:
                        if skel[old_point[0]+k,old_point[1]+l]:
                            new_point=[old_point[0]+k,old_point[1]+l]
                            new_seg.append(new_point)
                            skel[new_point[0],new_point[1]]=False  
            while (new_point not in div) :
                pot_new_point=[]
                for k in [-1,0,1]:
                    for l in [-1,0,1]:
                        if  0<=new_point[0]+k<dim1 and 0<=new_point[1]+l<dim2 and skel[new_point[0]+k,new_point[1]+l]:
                                pot_new_point=[new_point[0]+k,new_point[1]+l]
                if pot_new_point==[]:
                    break
                old_point,new_point=new_point,pot_new_point
                if new_point not in div:
                    new_seg.append(new_point)
                    skel[new_point[0],new_point[1]]=False 
                else:
                    close_neighbour=False
                    for elem in div:
                        if max(abs(new_point[0]-elem[0]),abs(new_point[1]-elem[1]))==1:
                            close_neighbour=True
                    if close_neighbour:
                        new_seg.append(new_point)
                        skel[new_point[0],new_point[1]]=False 
                    else:
                        new_seg.append(new_point)
            extr_seg.append(new_seg)
        return extr_seg,[]
    
    
     
    else: #case with branching
        extr_seg=[] #computation of the external case
        for i in range(len(ex)):
            old_point=ex[i]
            skel[old_point[0],old_point[1]]=False
            new_seg=[old_point]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if  0<=old_point[0]+k<dim1 and 0<=old_point[1]+l<dim2:
                        if skel[old_point[0]+k,old_point[1]+l]:
                            new_point=[old_point[0]+k,old_point[1]+l]
                            new_seg.append(new_point)
                            skel[new_point[0],new_point[1]]=False  
            while (new_point not in div) :
                pot_new_point=[]
                for k in [-1,0,1]:
                    for l in [-1,0,1]:
                        if  0<=new_point[0]+k<dim1 and 0<=new_point[1]+l<dim2 and skel[new_point[0]+k,new_point[1]+l]:
                                pot_new_point=[new_point[0]+k,new_point[1]+l]
                if pot_new_point==[]:
                    break
                old_point,new_point=new_point,pot_new_point
                if new_point not in div:
                    new_seg.append(new_point)
                    skel[new_point[0],new_point[1]]=False 
                else:
                    close_neighbour=False
                    for elem in div:
                        if max(abs(new_point[0]-elem[0]),abs(new_point[1]-elem[1]))<=2:
                            close_neighbour=True
                    if close_neighbour:
                        new_seg.append(new_point)
                        skel[new_point[0],new_point[1]]=False 
                    else:
                        new_seg.append(new_point)
            extr_seg.append(new_seg)
            
        newextr_seg,newint_seg=division_path_detek(skel)
        return extr_seg, newextr_seg+newint_seg



#simplify a segment into a 1D line (so that each pixel has at most 2 neighbour), we suppose that the segment goes in one direction ( it can't go back to point where it was, but can stay one the same pixel). This function is only used in simple_fusion
def one_D_line(seg) : 
    if len(seg)<2:
        return seg
    else:
        #selecting the two first element, by construction, the two first elements should always be different
        newseg=[seg[0],seg[1]]
        if 2<len(seg):
            for j in range(2,len(seg)):
                parent=newseg[-1]
                grandparent=newseg[-2]
                child=seg[j]
                if max(abs(child[0]-parent[0]),abs(child[1]-parent[1]),abs(child[0]-grandparent[0]),abs(child[1]-grandparent[1]))<=1:
                    del newseg[-1]
                    newseg.append(child)
                else :
                    newseg.append(child)
        if newseg[-1][0]==newseg[-2][0] and newseg[-1][1]==newseg[-2][1]:
            del newseg[-1]
        return newseg


# Erasing the shortest branch if it is alpha>1 shorter than the longuest. Else fusioning two external segments linked to the same division point to get just a 1D line, works on simple cases only
def fusion_erasing(extr_seg,div,alpha):
    if len(extr_seg)==1: #case with just one segment
        return extr_seg
    else: #each external segment has a connection to a division point
        new_seg=[]
        newdiv=[] #selection only one of the three division point if three are touching
        
        for point in div:
            close_neighbour=False
            for elem in newdiv:
                if max(abs(point[0]-elem[0]),abs(point[1]-elem[1]))<=3:
                    close_neighbour=True
            if not close_neighbour:
                newdiv.append(point)
        
        for i in range(len(newdiv)):
            temp=[]
            for seg in extr_seg:
                if max(abs(seg[-1][0]-newdiv[i][0]),abs(seg[-1][1]-newdiv[i][1]))<=3:
                    temp.append(seg)
                    
            if len(temp)==3:# only possible case for a 3 branches graph only
                maxi=np.argmax(np.array([len(temp[0]),len(temp[1]),len(temp[2])]))
                body=temp.pop(maxi)
                fusi=[]
                if len(temp[0])>alpha*len(temp[1]):
                    return [body+temp[0][-2::-1]]
                elif len(temp[1])>alpha*len(temp[0]):
                    return [body+temp[1][-2::-1]]
                else :
                    debut=min(len(temp[0]),len(temp[1]))
                    for i in range(debut):
                        a=(temp[0][len(temp[0])-debut+i][0]+temp[1][len(temp[1])-debut+i][0])//2
                        b=(temp[0][len(temp[0])-debut+i][1]+temp[1][len(temp[1])-debut+i][1])//2
                        fusi.append([a,b]) 
                        
                    return [body+one_D_line(fusi)[-2::-1]]
            
            elif len(temp)==2:
                if len(temp[0])>alpha*len(temp[1]):
                    new_seg.append(temp[0]) 
                elif len(temp[1])>alpha*len(temp[0]):
                    new_seg.append(temp[1]) 
                else :
                    fusi=[]
                    debut=min(len(temp[0]),len(temp[1]))
                    for i in range(debut):
                        a=(temp[0][len(temp[0])-debut+i][0]+temp[1][len(temp[1])-debut+i][0])//2
                        b=(temp[0][len(temp[0])-debut+i][1]+temp[1][len(temp[1])-debut+i][1])//2
                        fusi.append([a,b]) 
                    new_seg.append(one_D_line(fusi))
            elif len(temp)==1:
                new_seg.append(temp[0])
                
        return new_seg





def export_dataset(datset_name):
    # Plot and save segmentation results
    params = get_scaled_parameters(data_set=True,paths_and_names=True)
    if datset_name in params.keys():
        datasets = params[datset_name]
    elif isinstance(datasets, str): 
        raise NameError('This directory does not exist')
    
    data_direc = params["main_data_direc"]
    dicname = params["main_dict_name"]
    
    
    
    for direc in datasets:
        dir = os.path.join('single_cell', direc)
        
        if os.path.exists(dir):
            rmtree(dir)
        
        
        dic = np.load(os.path.join(data_direc, direc, dicname), allow_pickle=True)['arr_0'].item()
        frame = 1
        for fichier in dic:
            # plot image with masks overlaid
            img = np.load(dic[fichier]['adress'])['Height_fwd']
            masks = dic[fichier]['masks']
            fin_dir = os.path.join(dir, f'frame_{frame}')
            frame +=1
            os.makedirs(fin_dir)
            np.save(os.path.join(fin_dir, 'image.npy'), img)
            np.save(os.path.join(fin_dir, 'segmented_masks.npy'), masks)
      




#%% main function

def run_one_dataset_logs_only(Main_directory):
    print(Main_directory)
    
    year=Main_directory[-4:]
    
    print(year)
    today = datetime.date.today()

    if not 1950 <int(year)<=today.year:
        raise NameError('The name of your directory should end with the year of the experiment')
    
    params=get_scaled_parameters(paths_and_names=True)
    initial_data = params["initial_data_direc"]
    data_direc = params["main_data_direc"]
    final_data_direc = params["final_data_direc"]
    main_dict_name = params["main_dict_name"]
    masks_list_name = params["masks_list_name"]

    my_data = os.path.join(initial_data,Main_directory)     #path of dir
    #directory with the usefull information of the logs, None if there is no.


    data_direc='data/datasets/'


    #Temporary directories and files
    temp_dir_info=os.path.join(data_direc, Main_directory, "temporary_info")
    segments_path = os.path.join(data_direc, Main_directory, "cellpose_output")
    dimension_data = 'Dimension.npy'

    #Outputs
    
    #directory of the processed images (every image has the same pixel size and same zoom)
    final_data=os.path.join(data_direc, Main_directory, final_data_direc)
    #Saving path for the dictionnary
    saving_path=os.path.join(data_direc, Main_directory, main_dict_name)
    #dimension and scale of all the final data
    list_savingpath=os.path.join(data_direc, Main_directory, masks_list_name)
    


    
    ''' Parameters'''

    param_img = get_scaled_parameters(img_treatment=True)

    ratio_erasing_masks = param_img['ratio_erasing_masks']
    ratio_saturation = param_img['ratio_saturation']
    max_pixel_im = param_img['max_pixel_im']
    centerline_crop_param = param_img['centerline_crop_param']
    threshold_scars_directory = param_img['threshold_scars_directory']
    

    step=0
    ''''''
    
    print(Main_directory,"data_prep",step)
    step+=1
    data_prep_logs_only(my_data,final_data,temp_dir_info,threshold_scars_directory)
    
    print(Main_directory,"dimension_def",step)
    step+=1
    dimension_def_logs_only(final_data,temp_dir_info,dimension_data,max_pixel_im)
    
    print(Main_directory,"run_cel",step)
    step+=1
    if Main_directory == "WT_mc2_55/06-10-2015":
        run_cel_logs_only(final_data, segments_path, temp_dir_info, dimension_data, pref_channel='DMTModulus_fwd')
    else:
        run_cel_logs_only(final_data,segments_path,temp_dir_info,dimension_data)
    
    print(Main_directory,"download_dict",step)
    step+=1
    main_dict=download_dict_logs_only(final_data,temp_dir_info,segments_path,dimension_data,year)
    
    print(Main_directory,"centroid_area",step)
    step+=1
    centroid_area(main_dict)

    print(Main_directory,"clean_masks",step)
    step+=1
    clean_masks(ratio_erasing_masks,ratio_saturation, main_dict)
    
    print(Main_directory,"main_parenting",step)
    step+=1
    main_parenting(main_dict) 

    print(Main_directory,"construction_mask_list",step)
    step+=1
    construction_mask_list(main_dict,Main_directory,list_savingpath)
    
    print(Main_directory,"centerline_mask",step)
    step+=1
    centerline_mask(main_dict,centerline_crop_param,saving=True,savingpath=saving_path)
    
    
    
def main(Directory= "all"):
    params = get_scaled_parameters(data_set=True)
    if Directory in params.keys():
        with Pool(processes=8) as pool:
            for _ in pool.imap_unordered(run_one_dataset_logs_only, params[Directory]):
                pass
    elif isinstance(Directory, list): 
        with Pool(processes=8) as pool:
            for _ in pool.imap_unordered(run_one_dataset_logs_only, Directory):
                pass
    elif isinstance(Directory, str): 
        raise NameError('This directory does not exist')
    
    

    
    
#%% running main function   

if __name__ == "__main__":
    Directory= 'good'
    run_one_dataset_logs_only( Directory )
    # main()
    

