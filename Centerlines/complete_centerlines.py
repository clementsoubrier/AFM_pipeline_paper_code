# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:48:04 2023

@author: shawn
"""
import numpy as np
from PIL import Image #dealing with .tif images


#%% Centerline completion helper functions ####################################

def get_neighbors(coords,coords_lst):
    neighborhood = coords + np.array([[-1, -1],[-1, 0],[-1, 1],[0, -1],[0, 1],[1, -1],[1, 0],[1, 1]]) # get the possible neighbors of the current node
    neighbors = [list(nn) for nn in neighborhood for cc in coords_lst if np.all(nn==cc)] # get a list of all the neighbors 
    return neighbors

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

def find_extremal_div_pts(coords_lst):
    ex_pts = []
    div_pts = []
    for coordinates in coords_lst:
        neighbors = get_neighbors(coordinates,coords_lst)
        if len(neighbors)==1:
            ex_pts.append(coordinates)
        elif len(neighbors)>2:
            div_pts.append(coordinates)
    return ex_pts, div_pts


def dda_line(end_segment, mask):
    dy, dx = end_segment[-1][0] - end_segment[0][0], end_segment[-1][1] - end_segment[0][1]
    
    if abs(dx)>=abs(dy):
        step = abs(dx)
    else:
        step = abs(dy)
        
    dx = dx/step
    dy = dy/step
    x = end_segment[-1][1]
    y = end_segment[-1][0]
    
    line = []
    while True:
        x = x + dx
        y = y + dy
        x_coord = int(np.round(x))
        y_coord = int(np.round(y))
        
        if x_coord>=mask.shape[1] or y_coord>=mask.shape[0] or x_coord<0 or y_coord<0:
            break
        elif mask[y_coord,x_coord]!=0:
            line.append([y_coord,x_coord])
        else:
            break
        
    return line



#%% Alternative wrapper function (accepts one centerline numpy array and corresponding binarized mask numpy array)

def complete_one_centerline(centerline, mask):
    
    """
    Extrapolates the end segments of the centerline to the boundary of the given mask.

    Parameters:
        centerline (numpy.ndarray): Array of y-x coordinates representing the midline of a masked cell.
        
        mask (numpy.ndarray): Binarized array with non-zero values indicating the cell mask coordinates.
                              All other mask labels and background pixels should be set to 0.

    Returns:
            numpy.ndarray: Extended centerline with end segments extrapolated to the mask boundary.
    """
    # if len(centerline)<5: # skip if the centerline is too small or absent
        # print("Warning: centerline too small or missing")
        
    end_points = find_extremal_div_pts(centerline)[0]
    end_points = [list(e)for e in end_points]
    
    # if len(end_points)!=2: # skip if there are fewer or more than 2 terminal pixels in the centerline
        # print("Warning: fewer than 2 (looped) or more than 2 (branched) centerline terminal pixels")
    
    extended_centerline = np.copy(centerline)
    
    for end in end_points:
        end_segment = [end]
        centerline_copy = np.ndarray.tolist(np.copy(centerline))
        centerline_copy.remove(end)
        
        for ii in range(5):
            neighbor = get_neighbors(end,centerline_copy)
            
            if len(neighbor)!=1: # stop adding coordinates to the end segment if discontinuous or branched centerline
                # print("Warning: end segment branched or broken")
                break
            else:
                neighbor = list(neighbor[0])
            
            # print(neighbor)
            end_segment.append(neighbor)
            # print(end_segment)
            centerline_copy.remove(neighbor)
            end = neighbor
            
        end_segment = end_segment[::-1]
        
        if len(end_segment)>1: # extrapolate the end line segment if a slope can be calculated
            line = dda_line(end_segment, mask)
        # else:
            # print("Warning: end segment too short for extrapolation")
            
        
        if len(line)>0: # extend the centerline if the extension exists
            extended_centerline = np.concatenate((extended_centerline, np.array(line)))
        # else: 
            # print("Warning: centerline not extended from one end")
    
    centerline_copy = np.copy(extended_centerline).tolist()
    terminal_coordinates = find_extremal_div_pts(centerline_copy)[0][0]
    centerline_copy.remove(terminal_coordinates)
    reordered = [terminal_coordinates]
    while len(centerline_copy)>0:
        neighbors = list(get_neighbors(reordered[-1],centerline_copy))
        if len(neighbors)>1:
            for n in neighbors:
                next_neighbors = get_neighbors(n, centerline_copy)
                centerline_copy.remove(n)
                if np.any([nn for nn in next_neighbors if nn not in neighbors]):
                    reordered.append(n)
        elif len(neighbors)==0:
            break
        else:
            reordered.append(neighbors[0])
            centerline_copy.remove(neighbors[0])
            
    extended_centerline = np.array(one_D_line(reordered), dtype=np.int64)
    
    return extended_centerline





