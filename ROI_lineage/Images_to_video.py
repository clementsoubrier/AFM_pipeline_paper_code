#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:08:29 2023

@author: c.soubrier
"""
import sys
import cv2
import os
import numpy as np

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters
from ROI_lineage.plot_final_lineage_tree import plot_image_lineage_tree





def create_video(direc):

    params=get_scaled_parameters(paths_and_names=True,plot=True,video=True)
    
    dicname = params["main_dict_name"]
    data_direc = params["main_data_direc"]
    indexlistname = params['roi_masks_list_name']
    res_path = params["results_direc"]
    colormask = params["masks_colors"]
    dir_res = os.path.join(res_path, direc,)

    if os.path.exists(dir_res):
        for file in os.listdir(dir_res):
            os.remove(os.path.join(dir_res, file))
    else:
        os.makedirs(dir_res)

    image_folder = os.path.join(params["image_folder_video"], direc)
    video_name = os.path.join(dir_res, params["video_name"])
    print(dir_res)

    

    main_dict=np.load(os.path.join(data_direc, direc, dicname), allow_pickle=True)['arr_0'].item()
    indexlist=np.load(os.path.join(data_direc, direc, indexlistname), allow_pickle=True)['arr_0']

    plot_image_lineage_tree(main_dict,colormask,indexlist,direc,saving=True,img_dir=image_folder)
    

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, 0,2, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()



if __name__ == "__main__":
    Directory = "good" 
    create_video(Directory)