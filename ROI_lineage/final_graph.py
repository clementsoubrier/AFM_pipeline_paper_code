#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:43:22 2023

@author: c.soubrier
"""

#%%  Imports
import os
import sys
import numpy as np
from numba import njit
from copy import deepcopy
import tqdm
from multiprocessing import Pool

from scipy.signal import fftconvolve

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters




#Construction of the lineage matrices 



#%% Functions


def trans_vector_matrix(dic,max_diff_time):
    dic_list=list(dic.keys())
    fichier=dic_list[-1]
    maxtime=dic[fichier]['time']
    mat_vec=np.zeros((maxtime+1,2*max_diff_time+1,2),dtype=np.int32)
    mat_ang=np.zeros((maxtime+1,2*max_diff_time+1))
    
    for number in tqdm.trange(len(dic_list)):
        fichier1=dic_list[number]
        if number<len(dic_list)-1 :
            time1=dic[fichier1]['time']
            masks1=main_mask(dic[fichier1]['masks'])
            angle1=dic[fichier1]['angle']
            i=1
            timer=dic[dic_list[number+i]]['time']
            while abs(timer-time1)<=max_diff_time:
                fichier2=dic_list[number+i]
                masks2=main_mask(dic[fichier2]['masks'])
                angle2=dic[fichier2]['angle']
                update_trans_vector_matrix(mat_vec,mat_ang,masks1,angle1,time1,masks2,angle2,timer,max_diff_time)
                i+=1
                if number+i>=len(dic_list):
                    break
                else:
                    timer=dic[dic_list[number+i]]['time']
            
    return mat_vec, mat_ang


def update_trans_vector_matrix(mat_vec,mat_ang,masks1,angle1,time1,masks2,angle2,timer,max_diff_time):      #enlever le rayon et le vecguess
    angle=angle2-angle1
    if angle==0:
        res=opt_trans_vec(masks1,masks2)
        mat_vec[time1,timer-time1+max_diff_time]=res
        mat_vec[timer,time1-timer+max_diff_time]=-res
    else:
        dim1,dim2=np.shape(masks1)
        centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
        masks1=rotation_img(angle,masks1,centerpoint)
        res=opt_trans_vec(masks1,masks2).astype(np.int32)
        mat_vec[time1,timer-time1+max_diff_time]=res
        mat_ang[time1,timer-time1+max_diff_time]=angle
        mat_ang[timer,time1-timer+max_diff_time]=-angle
        mat_vec[timer,time1-timer+max_diff_time]=(rotation_vector(-angle,-res,np.array([0,0],dtype=np.int32))).astype(np.int32)



def lineage_matrix(dic,maskslist,mat_vec,mat_ang,max_diff_time,threshold):
    

    mat_dim=len(maskslist)
    mat=np.zeros((mat_dim,mat_dim))
    diclist=list(dic.keys())
    for i in tqdm.trange(len(diclist)-1):
        fichier=diclist[i]
        base_time=dic[fichier]['time']
        child_fichier=dic[fichier]['child']
        timer=dic[child_fichier]['time']
        while abs(timer-base_time)<=max_diff_time:
            transfert=mat_vec[base_time,timer-base_time+max_diff_time]
            angle=mat_ang[base_time,timer-base_time+max_diff_time]
            mask_c=dic[child_fichier]['masks']
            area_c=dic[child_fichier]['area']
            mask_p=dic[fichier]['masks']
            area_p=dic[fichier]['area']
            
            mask_c=mask_transfert(mask_c,transfert)
            
            if angle!=0:
                dim1,dim2=np.shape(mask_p)
                centerpoint=np.array([dim1//2,dim2//2],dtype=np.int32)
                mask_p=rotation_img(angle,mask_p,centerpoint)
            
            links_p,links_c =comparision_mask_score(mask_c,mask_p,area_c,area_p,threshold)
            len_p,len_c=np.shape(links_p)
            # for the matrix the entry [i,j] means the intersection of mask i and j divided by area of mask i
            
            for k in range(len_p):
                for l in range(len_c):
                    i=dic[fichier]["mask_list"][k]
                    j=dic[child_fichier]["mask_list"][l]
                    mat[i,j]=links_p[k,l]
                    j=dic[fichier]["mask_list"][k]
                    i=dic[child_fichier]["mask_list"][l]
                    mat[i,j]=links_c[l,k]
                    
            child_fichier=dic[child_fichier]['child']
            if  child_fichier=='':
                break
            timer=dic[child_fichier]['time']
            
    return mat

@njit 
def comparision_mask_score(mask_c,mask_p,area_c,area_p,threshold):
    number_mask_c=len(area_c)
    number_mask_p=len(area_p)
    dim1,dim2=np.shape(mask_p)
    result_p=np.zeros((number_mask_p,number_mask_c))
    result_c=np.zeros((number_mask_c,number_mask_p))
    for j in range(1,number_mask_c+1):
        for i in range(1,number_mask_p+1):
            area=0
            for k in range(dim1):
                for l in range(dim2):
                    if mask_c[k,l]==j and mask_p[k,l]==i:
                        area+=1
            if area/area_c[j-1]>=threshold:
                result_c[j-1,i-1]=area/area_c[j-1]
            if area/area_p[i-1]>=threshold:
                result_p[i-1,j-1]=area/area_p[i-1]
    return result_p,result_c



# Effective computation of the translation vector : returns the translation vector that optimizes the score of the intersection of the two main shapes

def opt_trans_vec(img_1, img_2):
    corr = fftconvolve(img_1, img_2[::-1, ::-1])
    argmax = np.unravel_index(corr.argmax(), corr.shape)
    vec = np.array(argmax) - np.array(img_1.shape) + 1
    return vec

# Translation of the masks by a vector
@njit 
def mask_transfert(mask,vector):
    (l1,l2)=np.shape(mask)
    new_mask=np.zeros((l1,l2),dtype=np.int32)
    for i in range(l1):
        for j in range(l2):
            if (0<=i+vector[0]<=l1-1) and (0<=j+vector[1]<=l2-1) and mask[i,j]>0:
                new_mask[int(i+vector[0]),int(j+vector[1])]=mask[i,j]
    return new_mask


#rotation of a vector around a point, the angle is in radian (float as output)
@njit 
def rotation_vector(angle,vec,point):
    mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    newvec=vec-point
    var=np.array([mat[0,0]*newvec[0]+mat[0,1]*newvec[1],mat[1,0]*newvec[0]+mat[1,1]*newvec[1]])
    return point+var


def rotation_line(angle,line,point):    #format n by 2, ij representation
    mat=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    newline=line-point
    var= newline @ np.transpose(mat)  
    return point+var

#rotation of an image around a point (int32 as input)
@njit 
def rotation_img(angle,img,point):
    dim1,dim2=np.shape(img)
    new_img=np.zeros((dim1,dim2),dtype=np.int32)
    for i in range(dim1):
        for j in range(dim2):
            trans_vec=rotation_vector(-angle,np.array([i,j]),point)#sign of the rotation : definition of the angle in the logs
            i_n,j_n=trans_vec[0],trans_vec[1]
            i_t=int(i_n)
            j_t=int(j_n)
            if 0<=i_t<dim1-1 and 0<=j_t<dim2-1:
                frac_i=i_n-i_t
                frac_j=j_n-j_t
                new_img[i,j]=np.int32(frac_i*frac_j*img[i_t,j_t]+frac_i*(1-frac_j)*img[i_t,j_t+1]+(1-frac_i)*frac_j*img[i_t+1,j_t]+(1-frac_i)*(1-frac_j)*img[i_t+1,j_t+1])
    return new_img


# Tranform all masks into one shape (the main shape)
@njit
def main_mask(mask):
    (l,L)=np.shape(mask)
    new_mask=np.zeros((l,L))
    for i in range(l):
        for j in range(L):
            if mask[i,j]!=0:
                new_mask[i,j]=1
    return new_mask


def clean_matrix(lin_mat,dic,maskslist,max_diff_time,thres):
    mat_dim=len(maskslist)
    newmat=np.zeros((mat_dim,mat_dim),dtype=np.int8)
    for i in tqdm.trange(mat_dim):
        base_time=dic[maskslist[i][2]]['time']
        time_list=[[] for k in range(max_diff_time)]
        if i<mat_dim:
            for j in range(i+1,mat_dim):
                if lin_mat[i,j]>thres/4:      #no to put 0, float precision problem can appear
                    time_list[dic[maskslist[j][2]]['time']-base_time-1].append(j)
            
            for index in range(len(time_list)):
                element=time_list[index]
                if len(element)==1:
                    if lin_mat[i,element[0]]>thres and lin_mat[element[0],i]>thres:
                        newmat[i,element[0]]= index+1
                elif len(element)==2:
                    if lin_mat[i,element[0]]+lin_mat[i,element[1]]>thres and lin_mat[element[0],i]>thres and lin_mat[element[1],i]>thres:
                        newmat[i,element[0]]= index+1
                        newmat[i,element[1]]= index+1

     
    return newmat






#%%Deriving the boolean matrix from the link matrix. The bollean matrix maximizes the size of trees


def Bool_from_linkmatrix(linkmat,max_diff_time):
    dim=len(linkmat)
    newmat=np.zeros((dim,dim),dtype=bool)
    print('Roots and links detection')
    forwardlinks,backwardlinks,rootslist=detekt_roots(linkmat,dim) 
    print('Leafs detection')
    endpoints=detekt_end_leafs(forwardlinks,backwardlinks,linkmat,dim,max_diff_time)
    print('Computing longest paths')
    final_links=longuest_path(backwardlinks,rootslist,dim)
    for point in endpoints:
        update_bool_mat(final_links[point],newmat)
    return newmat


# update the boolean matrix of the lineage tree to add the leaf link_list
def update_bool_mat(link_list,mat):
    len_lis=len(link_list)
    if len_lis>=2:
        links=deepcopy(link_list)
        links.reverse()
        for num in range(1,len_lis):
            i=links[num]
            j=links[num-1]
            if mat[i,j]:
                break
            else:
                mat[i,j]=True
        
    
#detekts roots of the tree and transform the matrix into an adjacence list
def detekt_roots(linkmat,dim):
    
    forwardlinks=[]
    backwardlinks=[]
    for i in range(dim):
        forwardlinks.append([])
        backwardlinks.append([])
    rootslist=[]
    
    
    for i in tqdm.trange(dim):
        if np.max(linkmat[i,:])>0:
            for j in range(i+1,dim):
                if linkmat[i,j]>0:
                    forwardlinks[i].append(j)
                    backwardlinks[j].append(i)
            if np.max(linkmat[:,i])==0:
                rootslist.append(i)
        elif np.max(linkmat[:,i])==0:
            rootslist.append(i)
            
            
    return forwardlinks,backwardlinks,rootslist


# detect the true leafs of the tree (it comes from a real division and is not the result of bad detection)
def detekt_end_leafs(forwardlinks,backwardlinks,linkmat,dim,depth):
    res=[]
    for i in tqdm.trange(dim):
        
        
        if forwardlinks[i]==[] and backwardlinks[i]!=[]:
            #print(i)
            ancestors=list_ancestors(i,backwardlinks,linkmat,depth)
            #print(ancestors)
            children=list_children(ancestors,forwardlinks,linkmat,2*depth)
            #print(children)
            if len(ancestors) >= 2 and not max(children)>i:          #checking that a real division has been detected
                res.append(i)
            elif len(ancestors)==1:                                     #case when a problem occurs just after division
                div_point=backwardlinks[i][0]
                num=linkmat[div_point,i]
                successor=np.argwhere(linkmat[div_point,:]==num)
                if successor[0,0]==i:
                    sister_cell=successor[1,0]
                else:
                    sister_cell=successor[0,0]
                
                if not forwardlinks[sister_cell]==[]:
                    res.append(i)
                
    return res



# detection of the leafs of the tree
def end_leafs(linkmat,dim):
    leafslist=[]
    for i in range(dim):
        if np.max(linkmat[i,:])==0:
            leafslist.append(i)
    return leafslist

# create the list of ancestors of a node in the tree up to a certain time
def list_ancestors(i,backwardlinks,linkmat,depth):
    res=[i]
    if depth<1:
        return res
    else:
        for indiv in backwardlinks[i]:
            subdepth=linkmat[indiv ,i]
            if subdepth>=1 and subdepth<=depth and np.count_nonzero(linkmat[indiv,:]==subdepth)==1:
                res+=list_ancestors(indiv, backwardlinks,linkmat,depth-subdepth)
        return list(set(res))


# create the list of children of a node in the tree up to a certain time
def list_children(ancestors,forwardlinks,linkmat,depth):
    res=deepcopy(ancestors)
    if depth<1:
        return res
    else:
        for indiv in ancestors:
            for link in forwardlinks[indiv]:
                subdepth=linkmat[indiv,link]
                if subdepth>=1 and subdepth<=depth:
                    res+=list_children([link],forwardlinks,linkmat,depth-subdepth)
            
        return list(set(res))
            

def longuest_path(backwardlinks,rootslist,dim):
    path=[[] for i in range(dim)]
    value=np.zeros(dim,dtype=np.int32)

    for root in rootslist:
        path[root]=[root]
        value[root]=1
        
        
    for iteration in tqdm.trange(dim):
        value_iter,path_iter=update_longest_path(iteration,backwardlinks,value,path)
        value[iteration]=value_iter
        path[iteration]=path_iter
        
    return path
        
    
def update_longest_path(iteration,backwardlinks,value,path):
    if value[iteration]>0:
        return value[iteration],path[iteration]
    else:
        parents=backwardlinks[iteration]
        parent_number=len(parents)
        count=0
        finalpath=[]
        for i in range(parent_number):
            parent_value,parent_path=update_longest_path(parents[i],backwardlinks,value,path)
            if parent_value>=count:
                count,finalpath=parent_value,parent_path
        value[iteration]=count+1
        path[iteration]=finalpath+[iteration]
        return value[iteration],path[iteration]
        
#%% running the whole lineage tree algorithm    


def Final_lineage_tree(direc):

    params=get_scaled_parameters(lineage_tree=True)
    # max time (minutes) between two comparable frames
    max_diff_time = params["max_diff_time"]

   
    surfthresh = params["child_div_thres"]

    #fraction of the preserved area to consider child and parent relation for masks (fusioning of 2 masks after division)
    finthres = params["final_thresh"] 


    params = get_scaled_parameters(paths_and_names=True)

    dicname = params["main_dict_name"]
    listname = params["masks_list_name"]
    linmatname = params['lineage_matrix_name']
    boolmatname = params['bool_matrix_name']
    data_direc = params["main_data_direc"]
    linkmatname = params['link_matrix_name']

    
    masks_list = np.load(os.path.join(data_direc, direc, listname), allow_pickle=True)['arr_0']
    #print(masks_list)
    main_dict=np.load(os.path.join(data_direc, direc, dicname), allow_pickle=True)['arr_0'].item()
    #print(main_dict)
    print(direc,'trans_vector_matrix 1')
    vector_matrix, angle_matrix=trans_vector_matrix(main_dict,max_diff_time) #translation vector and rotation angle between the different frames
    print(direc,'lineage_matrix 2')
    lin_mat=lineage_matrix(main_dict,masks_list,vector_matrix, angle_matrix,max_diff_time,surfthresh)
    print(direc,'Link_matrix 3')
    Link_mat=clean_matrix(lin_mat,main_dict,masks_list,max_diff_time,finthres)
    print(direc,'Bool_matrix  4')
    Bool_mat=Bool_from_linkmatrix(Link_mat,max_diff_time)
    print(direc,'saving 5')
    np.save(os.path.join(data_direc, direc, boolmatname), Bool_mat)
    np.save(os.path.join(data_direc, direc, linkmatname), Link_mat)
    np.save(os.path.join(data_direc, direc, linmatname), lin_mat)
    


    
def main(Directory= "all"):
    params = get_scaled_parameters(data_set=True)
    if Directory in params.keys():
        with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(Final_lineage_tree, params[Directory]):
                pass
    elif isinstance(Directory, list)  : 
        with Pool(processes=8) as pool:
            for direc in pool.imap_unordered(Final_lineage_tree, Directory):
                pass
    elif isinstance(Directory, str)  : 
        raise NameError('This directory does not exist')
    
#%% running main function   

if __name__ == "__main__":

    main()
    
    



