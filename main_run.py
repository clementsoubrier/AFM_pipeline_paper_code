#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:18:29 2022

@author: c.soubrier
"""
import os
import shutil
from multiprocessing import Pool


from scaled_parameters import get_scaled_parameters

import processing 
from ROI_lineage import final_graph
from ROI_lineage import plot_final_lineage_tree
from ROI_lineage import Images_to_video
from peaks_troughs import group_by_cell


def run_one_direc(direc):
    processing.run_one_dataset_logs_only(direc)
    final_graph.Final_lineage_tree(direc)
    plot_final_lineage_tree.run_whole_lineage_tree(direc,False)
    Images_to_video.create_video(direc)
    group_by_cell.compute_dataset(direc)
    
    if direc == 'WT_INH_700min_2014':
        params = get_scaled_parameters(paths_and_names=True)
        data_direc = params["main_data_direc"]
        or_dir = os.path.join(data_direc, direc)
        dir_im_a = os.path.join(data_direc, "INH_after_700")
        dir_im_b = os.path.join(data_direc, "INH_before_700")
        
        if os.path.exists(dir_im_a):
            for file in os.listdir(dir_im_a):
                os.remove(os.path.join(dir_im_a, file))
        else:
            os.makedirs(dir_im_a)
        if os.path.exists(dir_im_b):
            for file in os.listdir(dir_im_b):
                os.remove(os.path.join(dir_im_b, file))
        else:
            os.makedirs(dir_im_b)
            
        for file in os.listdir(or_dir) :
            file_path = os.path.join(or_dir, file)
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(dir_im_a, file))
                shutil.copy(file_path, os.path.join(dir_im_b, file))
            




def main(Directory= "all"):
    params = get_scaled_parameters(data_set=True)
    if Directory in params.keys():
        with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(run_one_direc, params[Directory]):
                pass
    elif isinstance(Directory, list)  : 
        with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(run_one_direc, Directory):
                pass
    elif isinstance(Directory, str)  : 
        raise NameError('This directory does not exist')
    
    
    

if __name__ == "__main__":
    #  Running the whole processing pipeline (until tracking) over all the datasets, saving results plot, creating films and cell structure
    main()

    