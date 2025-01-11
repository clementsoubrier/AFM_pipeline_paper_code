import math
import sys

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.preprocess import evenly_spaced_resample, preprocess_centerline

#Alignment of height profiles

@njit
def get_sliding_windows(y_1, y_2, h_offset, window_len): 
    assert len(y_1) >= len(y_2)
    start_1 = max(0, h_offset)
    start_2 = max(0, -h_offset)
    end_1 = min(len(y_1), len(y_2) + h_offset)
    end_2 = min(len(y_2), len(y_1) - h_offset)
    windows_1 = np.lib.stride_tricks.sliding_window_view(y_1[start_1:end_1], window_len)
    windows_2 = np.lib.stride_tricks.sliding_window_view(y_2[start_2:end_2], window_len)
    windows_1 = np.ascontiguousarray(windows_1)
    windows_2 = np.ascontiguousarray(windows_2)
    return windows_1, windows_2


@njit
def numba_mean_last_axis(arr):
    means = []
    for x in arr:
        means.append(np.mean(x))
    return np.array(means)


@njit
def compute_v_offset_range(y_1, y_2, h_offset, window_len):
    windows_1, windows_2 = get_sliding_windows(y_1, y_2, h_offset, window_len)
    mean_1 = numba_mean_last_axis(windows_1)
    mean_2 = numba_mean_last_axis(windows_2)
    v_offsets = mean_1 - mean_2
    min_v_offset = v_offsets.min()
    max_v_offset = v_offsets.max()
    return min_v_offset, max_v_offset



@njit
def quantile_error(y_1, y_2, h_offset, window_len, quantile):
    windows_1, windows_2 = get_sliding_windows(y_1, y_2, h_offset, window_len)
    diff = windows_1 - windows_2
    errors = np.sum(diff**2, axis=1)
    final_error = np.quantile(errors, quantile)
    return final_error

@njit
def min_error(y_1, y_2, h_offset, window_len):
    start_1 = max(0, h_offset)
    start_2 = max(0, -h_offset)
    windows_1=y_1[start_1:start_1+window_len]
    windows_2=y_2[start_2:start_2+window_len]
    diff = windows_1 - windows_2
    final_error = np.sum(diff**2)
    return final_error


@njit
def quantile_error_no_window(y_1, y_2, h_offset, quantile):
    start_1 = max(0, h_offset)
    start_2 = max(0, -h_offset)
    window_len=np.min(np.array([len(y_2)+h_offset,len(y_2),len(y_1)-h_offset]))
    windows_1=y_1[start_1:start_1+window_len]
    windows_2=y_2[start_2:start_2+window_len]
    diff = windows_1 - windows_2
    final_error = np.quantile(diff**2, quantile)
    return final_error


@njit
def align_quantile(y_1, y_2, max_translation, penalty, window, quantile, pixel_size,use_quantile, division, penalty_division):
    if len(y_2) > len(y_1):
        best_h_offset, best_v_offset, best_error = align_quantile(
            y_2, y_1, max_translation, penalty, window, quantile, pixel_size,use_quantile, division, penalty_division
        )
        return -best_h_offset, -best_v_offset, best_error
    y_1 = y_1.astype(np.float32)
    y_2 = y_2.astype(np.float32)
    window_len = round(window * (len(y_2) - 1)) + 1
    physical_length = pixel_size * window_len #np.min(np.array([len(y_2)+h_offset,len(y_2),len(y_1)-h_offset]))
    v_offset_window_len = window_len*(window_len >1 )+ (round(0.2 * (len(y_2) - 1)) + 1)*(window_len == 1)
    min_h_offset = -round(max_translation * (len(y_2) - 1))
    max_h_offset = len(y_1) - len(y_2) + round(max_translation * (len(y_2) - 1))
    best_error = np.inf
    best_h_offset = 0
    best_v_offset = 0

    for h_offset in range(min_h_offset, max_h_offset + 1):
         #physical superimposition zone
        
        min_v_offset, max_v_offset = compute_v_offset_range(
            y_1, y_2, h_offset, v_offset_window_len
        )
        v_offset_sampling = math.ceil(max_v_offset - min_v_offset) + 1
        v_offsets = np.linspace(min_v_offset, max_v_offset, v_offset_sampling)
        stick_out = max(0, -h_offset) + max(0, len(y_2) + h_offset - len(y_1))
        physical_stick_out = stick_out * pixel_size
        stick_out_error = penalty * physical_stick_out

        for v_offset in v_offsets:
            if division:
                error = quantile_error(y_1, y_2 + v_offset, h_offset, window_len, quantile)
                stick_in = min(max(0, h_offset), max(0, len(y_1) - len(y_2) - h_offset))
                total_error = error / physical_length + stick_out_error + penalty_division * stick_in * pixel_size
            else :
                if window_len==1:
                    error = quantile_error_no_window(y_1, y_2, h_offset, quantile)
                elif use_quantile : 
                    error = quantile_error(y_1, y_2 + v_offset, h_offset, window_len, quantile)
                else :
                    error = min_error(y_1, y_2 + v_offset, h_offset, window_len)
                total_error = error / physical_length + stick_out_error
                
            if total_error < best_error:
                best_error = total_error
                best_h_offset = h_offset
                best_v_offset = v_offset

    return best_h_offset, best_v_offset, best_error


def align_with_reference(xs, ys, reference_xs, reference_ys, params, pixel_size, use_quantile_instead_of_min=True, smoothed=False, division=False):
    

    max_translation = params["max_translation"]
    penalty = params["penalty"]
    max_relative_translation=params["max_relative_translation"]
    window_relative_size=params["window_relative_size"]
    quantile_size=params["quantile_size"]
    smooth_std=params["smooth_std"]
    
    if division:
        penalty_division = params["penalty_division"]
        window_relative_size=params["window_relative_size_division"]
        quantile_size=params["quantile_size_division"]
    else:
        penalty_division = 0
        window_relative_size=params["window_relative_size"]
        quantile_size=params["quantile_size"]
    
    physical_max_translation = max_translation * pixel_size
    if reference_xs == [] or reference_ys == []:
        return xs, ys
    xs_p, ys_p = preprocess_centerline(
        xs, ys, params["kernel_len"], params["std_cut"], params["window"], pixel_size
    )
    if xs_p[-1] - xs_p[0] < physical_max_translation:
        xs_p, ys_p = evenly_spaced_resample(xs, ys, pixel_size)
    if smoothed:
        ys_p = gaussian_filter1d(ys_p, smooth_std, mode="nearest")
    if len(ys_p) < 5:
        return xs, ys
    
    ref_data = {'delta' : [], 'v_delta' : [], 'score' : [], 'bad_quality' : [], 'x0_1st' : []}
    for ref_cent in zip(reference_xs, reference_ys):
        
        ref_xs_p, ref_ys_p = preprocess_centerline(
            ref_cent[0], ref_cent[1], params["kernel_len"], params["std_cut"], params["window"], pixel_size
        )
        
        if ref_xs_p[-1] - ref_xs_p[0] < physical_max_translation:
            ref_xs_p, ref_ys_p = evenly_spaced_resample(
                ref_cent[0], ref_cent[1], pixel_size
            )
        
        if smoothed:
            ref_ys_p = gaussian_filter1d(ref_ys_p, smooth_std, mode="nearest")

        relative_translation=min(max_relative_translation, physical_max_translation/min(abs(xs_p[-1] - xs_p[0]),abs(ref_xs_p[-1] - ref_xs_p[0])))

        if len(ref_ys_p) < 5 :
            ref_data['bad_quality'].append(True)
            ref_data['delta'].append(0)
            ref_data['v_delta'].append(0)
            ref_data['score'].append(0)
            ref_data['x0_1st'].append(0)
        else :    
            # add max translation
            delta, v_delta, score = align_quantile(ref_ys_p, ys_p, relative_translation, penalty, window_relative_size, quantile_size, pixel_size, use_quantile_instead_of_min, division, penalty_division)
            
            ref_data['bad_quality'].append(False)
            ref_data['delta'].append(delta)
            ref_data['v_delta'].append(v_delta)
            ref_data['score'].append(score)
            ref_data['x0_1st'].append(ref_xs_p[0])
            
    ref_data['bad_quality'] = np.array(ref_data['bad_quality'], dtype=bool)
    ref_data['delta'] = np.array(ref_data['delta'])
    ref_data['v_delta'] = np.array(ref_data['v_delta'])
    ref_data['score'] = np.array(ref_data['score'])
    ref_data['x0_1st'] = np.array(ref_data['x0_1st'])
    
    
    
    if np.all(ref_data['bad_quality']):
        return xs, ys
    
    ref_data['score'][ref_data['bad_quality']] = np.max(ref_data['score'])+1
    best_parent_ind = np.argmin(ref_data['score'])
    
    
    h_translation = ref_data['x0_1st'][best_parent_ind] - xs_p[0] + ref_data['delta'][best_parent_ind] * pixel_size
    v_delta  = ref_data['v_delta'][best_parent_ind]

    return xs + h_translation, ys + v_delta    


# def align_with_reference_division(xs, ys, xs2, ys2, reference_xs, reference_ys, params, pixel_size,use_quantile_instead_of_min=True,smoothed=False):
#     params = params.copy()
#     max_translation = params["max_translation")
#     penalty = params["penalty")
#     max_relative_translation=params["max_relative_translation")
#     window_relative_size=params["window_relative_size")
#     quantile_size=params["quantile_size")
#     smooth_std=params["smooth_std")

#     physical_max_translation = max_translation * pixel_size
#     if reference_xs is None or reference_ys is None:
#         return xs, ys
#     xs_p, ys_p = preprocess_centerline(xs, ys, **params, resolution=pixel_size)
#     ref_xs_p, ref_ys_p = preprocess_centerline(
#         reference_xs, reference_ys, **params, resolution=pixel_size
#     )
#     if xs_p[-1] - xs_p[0] < physical_max_translation:
#         xs_p, ys_p = evenly_spaced_resample(xs, ys, pixel_size)
#     if ref_xs_p[-1] - ref_xs_p[0] < physical_max_translation:
#         ref_xs_p, ref_ys_p = evenly_spaced_resample(
#             reference_xs, reference_ys, pixel_size
#         )
#     if smoothed:
#         ys_p = gaussian_filter1d(ys_p, smooth_std, mode="nearest")
#         ref_ys_p = gaussian_filter1d(ref_ys_p, smooth_std, mode="nearest")

#     relative_translation=min(max_relative_translation,physical_max_translation/min(abs(xs_p[-1] - xs_p[0]),abs(ref_xs_p[-1] - ref_xs_p[0])))

#     if len(ref_ys_p) < 5 or len(ys_p) < 5:
#         return xs, ys

#     # add max translation
#     delta, v_delta = align_quantile(ref_ys_p, ys_p, relative_translation, penalty, window_relative_size, quantile_size, pixel_size,use_quantile_instead_of_min)
#     h_translation = ref_xs_p[0] - xs_p[0] + delta * pixel_size

#     return xs+ h_translation, ys+ v_delta    
