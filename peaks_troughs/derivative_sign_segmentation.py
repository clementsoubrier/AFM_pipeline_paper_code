
from enum import IntEnum, auto

import numpy as np
from scipy.ndimage import gaussian_filter1d
from numba import njit


# peaks and trough detection


class Feature(IntEnum):
    PEAK = auto()
    TROUGH = auto()


def find_extrema(der):
    is_pos = der >= 0
    sign_change = np.logical_xor(is_pos[1:], is_pos[:-1])
    intern_extrema = 1 + np.flatnonzero(sign_change)
    return intern_extrema



@njit
def first_filter_extrema(xs,extrema, values, classif, min_width):
    extr_len=len(extrema)

    bool_arr=np.zeros(extr_len,dtype=np.bool_)
        
    bool_arr[0],bool_arr[-1]=True,True
    for i in range(1,extr_len-1):
        bool_arr[i]= (np.abs(xs[extrema[i-1]]-xs[extrema[i]]>min_width) or np.abs(xs[extrema[i+1]]-xs[extrema[i]])>min_width)
        
                    
    return extrema[bool_arr],values[bool_arr], classif[bool_arr]

@njit
def second_filter_extrema(xs,ys,extrema, values, classif, min_depth):           #excluding flat extrema
    extr_len=len(extrema)
    bool_arr=np.zeros(extr_len,dtype=np.bool_)
    if extr_len<=2:
        return extrema, values, classif
    bool_arr[0],bool_arr[-1]=True,True
    for i in range(1,extr_len-1):               
        a_orth=np.array([[values[i-1]-values[i+1],1000*xs[extrema[i+1]]-1000*xs[extrema[i-1]]]])  #vector joining two minima or two maxima
        len_pts=extrema[i+1]-extrema[i-1]
        vect_list=np.ones((2,len_pts))*np.array([[1000*xs[extrema[i-1]]],[values[i-1]]])
        vect_list[0,:]-=xs[extrema[i-1]:extrema[i+1]]
        vect_list[1,:]-=ys[extrema[i-1]:extrema[i+1]]
        proj=np.abs(a_orth@vect_list)           
        bool_arr[i]=np.max(proj)/np.linalg.norm(a_orth)>min_depth   # distance to the line between the optima
    return extrema[bool_arr], values[bool_arr], classif[bool_arr]


@njit
def alternating_feature(extrema, values, classif):
    extr_len=len(extrema)
    bool_arr=np.ones(extr_len,dtype=np.bool_)
    if extr_len<=1:
        return extrema, values, classif
    for i in range(extr_len-1):               #excluding twice the same feature
        if classif[i]==classif[i+1]:
            bool_arr[i]=False
            match classif[i]:
                case Feature.PEAK:
                    j=i+(values[i+1]>values[i])
                    classif[i+1]=classif[j]
                    values[i+1]=values[j]
                    extrema[i+1]=extrema[j]
                case Feature.TROUGH:
                    j=i+(values[i+1]<values[i])
                    classif[i+1]=classif[j]
                    values[i+1]=values[j]
                    extrema[i+1]=extrema[j]
    return extrema[bool_arr], values[bool_arr], classif[bool_arr]
            
    
    
    


def resample_extrema(intern_extrema, xs, ys, min_width,min_depth):
    extrema = []
    values = []
    classif = []
    for i in intern_extrema:
        a = (ys[i + 1] - 2 * ys[i] + ys[i - 1]) / 2     #second derivative
        y = ys[i]
        extrema.append(i)
        values.append(y)
        if a < 0:
            classif.append(Feature.PEAK)
        else:
            classif.append(Feature.TROUGH)
    
    if len(extrema)<=2:
        return extrema, values, classif
    argsort = sorted(range(len(extrema)), key=extrema.__getitem__)
    extrema = np.array([extrema[i] for i in argsort])
    values = np.array([values[i] for i in argsort])
    classif = np.array([classif[i] for i in argsort])

    extrema, values, classif = first_filter_extrema(xs,extrema, values, classif,min_width)
    extrema, values, classif = alternating_feature(extrema, values, classif)
    extrema, values, classif = second_filter_extrema(xs,ys,extrema, values, classif, min_depth)

    return alternating_feature(extrema, values, classif)        



def find_peaks_troughs(
    xs,
    ys,
    smooth_std,
    min_width,
    min_depth,
):



    xs_smooth = gaussian_filter1d(xs, smooth_std, mode="nearest")
    ys_smooth = gaussian_filter1d(ys, smooth_std, mode="nearest")
    der = (ys_smooth[1:] - ys_smooth[:-1]) / (xs_smooth[1:] - xs_smooth[:-1])

   

    intern_extrema = find_extrema(der)
    extrema, _, classif = resample_extrema(intern_extrema,xs_smooth, ys_smooth, min_width,min_depth)


    peaks = []
    troughs = []


    for i in range(len(extrema)):
        match classif[i]:
            case Feature.PEAK:
                peaks.append(extrema[i])
            case Feature.TROUGH:
                troughs.append(extrema[i])
            case _:
                raise ValueError(
                    f"Unknown feature {classif[i]}, feature should "
                    f"be a {Feature.PEAK} or a {Feature.TROUGH}."
                )
    peaks = np.array(peaks)
    troughs = np.array(troughs)
    
    
    return xs, ys, peaks, troughs
