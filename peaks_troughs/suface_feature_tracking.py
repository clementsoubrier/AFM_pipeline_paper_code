import sys
import os

import numpy as np
from numba import njit

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters


# implementation of tracking of the peaks and troughs over time 
    


def create_peaks_troughs_list(cell):
    pnt_list=[]     # format : index, frame_number, peak=1 and trough = 0, xposition, yposition, timestamp, peaks_trough_number
    j=0
    for i, frame_data in enumerate(cell):
        time = frame_data['timestamp']
        peaks_index = []
        troughs_index = []
        for elem in frame_data["peaks"]:
            peaks_index.append(j)
            pnt_list.append([j, i, 1, frame_data["xs"][elem], frame_data["ys"][elem],time, -1]) 
            j+=1
        for elem in frame_data["troughs"]:
            troughs_index.append(j)
            pnt_list.append([j, i, 0, frame_data["xs"][elem], frame_data["ys"][elem],time, -1]) 
            j+=1
        frame_data['peaks_index'] = peaks_index
        frame_data['troughs_index'] = troughs_index
        np.savez(frame_data["filename"],**frame_data)
    return np.array(pnt_list)



@njit
def pnt_link_matrix(pnt_list, max_xdrift, max_ydrift, max_time):
    matlen = len(pnt_list)
    time_mat = np.zeros((matlen, matlen), dtype=np.int32)
    dist_mat = np.zeros((matlen, matlen))
    for i in range(matlen-1):
        for j in range(i+1,matlen):
            time=np.abs(pnt_list[i][5]-pnt_list[j][5]) 
            dist_x= np.abs(pnt_list[i][3]-pnt_list[j][3])
            
            cond =  dist_x < max_xdrift \
                and np.abs(pnt_list[i][4]-pnt_list[j][4]) < max_ydrift \
                and 0 < time < max_time \
                and pnt_list[i][2] == pnt_list[j][2]
            if cond :
                time_mat[j,i] = time 
                dist_mat[j,i] = dist_x
            
    return time_mat, dist_mat


def final_pnt_link_matrix(new_pnt_list, pnt_ROI, final_max_xdrift, max_ydrift, max_time):
    matlen = len(new_pnt_list)
    time_mat = np.zeros((matlen, matlen), dtype=np.int32)
    dist_mat = np.zeros((matlen, matlen))
    for i in range(matlen-1):
        for j in range(i+1,matlen):
            time=np.abs(new_pnt_list[i][5]-new_pnt_list[j][5]) 
            dist_x= np.abs(new_pnt_list[i][3]-new_pnt_list[j][3])
            
            cond =  dist_x < final_max_xdrift \
                and np.abs(new_pnt_list[i][4]-new_pnt_list[j][4]) < max_ydrift \
                and 0 < time < max_time \
                and new_pnt_list[i][2] == new_pnt_list[j][2]
                
            if end_of_lineage(i, new_pnt_list, pnt_ROI) == -1 or end_of_lineage(j, new_pnt_list, pnt_ROI) == 1:
                cond = False
            if cond :
                time_mat[j,i] = time 
                dist_mat[j,i] = dist_x
            
    return time_mat, dist_mat


def end_of_lineage(i, new_pnt_list, pnt_ROI):
    lineage = int(new_pnt_list[i,-1])
    if lineage == -1:
        return 0
    if new_pnt_list[i,0] == pnt_ROI[lineage][0]:
        return -1
    return 1
    

def update_list(pnt_list, time_mat, dist_mat):
    index = 0
    generations = int(np.max(pnt_list[:,1]))
    last_gen = find_gen(pnt_list,generations)
    for elem in last_gen:
        pnt_list[int(elem[0]),6] = index
        index+=1
    for gen_num in range(generations-1,-1,-1):
        index = first_update_pnt_list(pnt_list, gen_num, time_mat, dist_mat,index)
    return pnt_list
        
    
    
    
def find_gen(pnt_list, generations):
    mask = pnt_list[:,1] == generations
    new_list = pnt_list[mask]
    return new_list[np.argsort(new_list[:, 3])]



def first_update_pnt_list(pnt_list, gen_num, time_mat, dist_mat, index): # create ROIs for elements that are the closest to eachother in both directions
    gen = find_gen(pnt_list, gen_num)
    for elem in gen:
        
        feature_number = int(elem[0])
        compar_ind, closest_ind_compar = closest_neighbour(feature_number, time_mat, dist_mat)
        
        if closest_ind_compar == feature_number:
            if pnt_list[compar_ind, -1] == -1:
                pnt_list[compar_ind, -1] = index
                index += 1
            pnt_list[feature_number, -1] = pnt_list[compar_ind, -1] 
                    
    return index



def closest_neighbour(feature_number, time_mat, dist_mat):
    compar_ind, closest_ind_compar = -1, -1
    mask = time_mat[:,feature_number]>0
    if np.any(mask):
        min_time = np.min(time_mat[:,feature_number][mask])
        tent_ind = np.nonzero(time_mat[:,feature_number] == min_time)[0]
        closest_pt = np.argmin(dist_mat[:,feature_number][tent_ind])
        
        compar_ind = tent_ind[closest_pt]   # index of closest point linked with elem
        compar_mask = np.ravel(time_mat[compar_ind,:]>0)
        
        if np.any(compar_mask):
            compar_line = np.ravel(time_mat[compar_ind,:])
            compar_min_time = np.min(compar_line[compar_mask])
            compar_tent_ind = np.nonzero(compar_line  == compar_min_time)[0]
            compar_closest_pt = np.argmin(np.ravel(dist_mat[compar_ind,:])[compar_tent_ind])
            closest_ind_compar = compar_tent_ind[compar_closest_pt]
    return compar_ind, closest_ind_compar  # closest point to feature_number, closest point to closest_ind


def reconstruct_ROI(pnt_list):
    ROI_dict = {}
    indices = list(set([elem[-1] for elem in pnt_list]))
    for index in indices:
        if index != -1:
            ROI_ind = np.nonzero(pnt_list[:,-1] == index)[0]
            ROI_dict[index] = ROI_ind
    return ROI_dict



def update_list_non_crossing(new_pnt_list, pnt_list, new_time_mat, new_dist_mat, final_max_xdrift):
    possible_link = []
    for feature_number in range(len(new_pnt_list)):
        compar_ind, closest_ind_compar = closest_neighbour(feature_number, new_time_mat, new_dist_mat)
        if closest_ind_compar == feature_number:
            possible_link.append([new_pnt_list[compar_ind, 0], new_pnt_list[feature_number, 0], new_dist_mat[compar_ind, feature_number]])
                    
    
    if len(possible_link) >= 1 :
        possible_link = np.array(possible_link)
        mask_to_erase = np.ones(len(possible_link), dtype=bool)
        
        for elem in possible_link[:,0]:
            arg_mask=np.logical_or(possible_link[:,0]==elem,possible_link[:,1]==elem)
            args = np.nonzero(arg_mask)[0]
            if sum(possible_link[args,2]) > final_max_xdrift:
                mask_to_erase[args] = False
                
        possible_link = possible_link[mask_to_erase]
        if len(possible_link) >= 1:
            possible_link = possible_link[np.argsort(possible_link[:,-1])]
            pnt_list = glue_non_crossing(pnt_list, possible_link)
    
    return pnt_list





def update_index_number(new_pnt_list, pnt_list):
    mask = new_pnt_list[:,-1] != -1
    for elem in new_pnt_list[mask]:
        old_index = pnt_list[elem[0],-1]
        if elem[-1]!= old_index:
            pnt_list[pnt_list[:,-1] == old_index][:,-1] = elem[-1]
    return pnt_list



def glue_non_crossing(pnt_list, possible_link):
    for elem in possible_link:
        left_feature = []
        right_feature = []
        for i in [0,1]:
            first_ind = int(elem[i])
            if pnt_list[first_ind,-1] == -1:
                xposition = pnt_list[first_ind,3]
                gen = pnt_list[first_ind,1]
                for point in pnt_list[pnt_list[:,1] == gen]:
                    if point[3] < xposition and point[-1] != -1:
                        left_feature.append(point[-1])
                    if point[3] > xposition and point[-1] != -1:
                        right_feature.append(point[-1])
                        
            else:
                val = pnt_list[first_ind,-1]
                for subelem in np.nonzero(pnt_list[:,-1] == val)[0]:
                    gen = pnt_list[subelem,1]
                    xposition = pnt_list[subelem,3]
                    for point in pnt_list[pnt_list[:,1] == gen]:
                        if point[3] < xposition and point[-1] != -1:
                            left_feature.append(point[-1])
                        if point[3] > xposition and point[-1] != -1:
                            right_feature.append(point[-1])
                
        
        gluing = True
        if (set(left_feature) & set(right_feature)): #checking if a same feature appears at both sides of the reglued feature (crossing)
            gluing = False
            
        if gluing:                              # gluing elements by updating indices
            old_index = pnt_list[int(elem[1]),-1]
            new_index = pnt_list[int(elem[0]),-1]
            
            if old_index == new_index == -1:
                ind = np.max(pnt_list[:,-1])+1
                pnt_list[int(elem[1]),-1] = ind
                pnt_list[int(elem[0]),-1] = ind
            elif old_index == -1:
                pnt_list[int(elem[1]),-1] = pnt_list[int(elem[0]),-1]
            elif new_index == -1:
                pnt_list[int(elem[0]),-1] = pnt_list[int(elem[1]),-1]
            else :
                mask = pnt_list[:,-1] == old_index
                pnt_list[:,-1][mask] = new_index 
            
    return pnt_list





def peak_troughs_lineage(dataset, cell, roi_dir):
    params=get_scaled_parameters(paths_and_names=True, pnt_tracking=True)
    max_time = params["max_time"]
    fist_max_xdrift = params["first_max_xdrift"]
    max_ydrift =params["max_ydrift"]
    final_max_xdrift = params["final_max_xdrift"]
    saving_dic = os.path.join(params["dir_cells"], dataset, roi_dir, params["dir_cells_list"])
    
    if os.path.exists(saving_dic):
        for file in os.listdir(saving_dic):
            os.remove(os.path.join(saving_dic, file))
    else:
        os.makedirs(saving_dic)
        
    pnt_list_path = os.path.join(saving_dic, params['pnt_list_name'])
    pnt_ROI_path = os.path.join(saving_dic, params['pnt_ROI_name'])
    
    pnt_list = create_peaks_troughs_list(cell)
    if len(pnt_list) >= 1:
        time_mat, dist_mat = pnt_link_matrix(pnt_list, fist_max_xdrift, max_ydrift, max_time)
        update_list(pnt_list, time_mat, dist_mat)
        
        pnt_ROI = reconstruct_ROI(pnt_list)
        mask = pnt_list[:,-1] == -1
        for key in pnt_ROI:
            mask[pnt_ROI[key][0]] = True
            mask[pnt_ROI[key][-1]] = True
        new_pnt_list = pnt_list[mask]
        
        new_time_mat, new_dist_mat = final_pnt_link_matrix(new_pnt_list, pnt_ROI, final_max_xdrift, max_ydrift, max_time)
        pnt_list = update_list_non_crossing(new_pnt_list, pnt_list, new_time_mat, new_dist_mat, final_max_xdrift)
        
        pnt_ROI = reconstruct_ROI(pnt_list)
        
        np.savez_compressed(pnt_list_path, pnt_list)
        np.savez_compressed(pnt_ROI_path, pnt_ROI)

