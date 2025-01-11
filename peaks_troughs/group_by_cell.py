import glob
import math
import operator
import os
import sys
import statistics
from enum import IntEnum
from shutil import rmtree
from multiprocessing import Pool

import numpy as np
import tqdm



package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.align import align_with_reference
from peaks_troughs.derivative_sign_segmentation import find_peaks_troughs
from peaks_troughs.preprocess import keep_centerline
from scaled_parameters import get_scaled_parameters
from ROI_lineage.final_graph import rotation_line
from peaks_troughs.suface_feature_tracking import peak_troughs_lineage


# Creating the cell structure for data analysis and computing tracking, alignment of height profiles, peaks and troughs detection and tracking

class Orientation(IntEnum):
    UNKNOWN = 0
    OLD_POLE_NEW_POLE = 1
    NEW_POLE_OLD_POLE = 2


def find_all_datasets():
    datasets_dir = os.path.join("data", "datasets")
    pattern = os.path.join(datasets_dir, "**", "final_data", "")
    datasets = glob.glob(pattern, recursive=True)
    datasets = [
        os.path.dirname(os.path.relpath(path, datasets_dir)) for path in datasets
    ]
    return datasets


def load_data(dataset, log_progress):
    path = os.path.join( "data", "datasets", dataset)
    if log_progress:
        print("Loading main dictionary.", end="")
    main_dict_path = os.path.join(path, "main_dictionnary.npz")
    main_dict = np.load(main_dict_path, allow_pickle=True)["arr_0"].item()
    if log_progress:
        print(" Done.")
        print("Loading masks list.", end="")
    masks_list_path = os.path.join(path, "masks_list.npz")
    masks_list = np.load(masks_list_path, allow_pickle=True)["arr_0"]
    if log_progress:
        print(" Done.")
        print("Loading ROI dictionary.", end="")
    roi_dict_path = os.path.join(path, "ROI_dict.npz")
    roi_dict = np.load(roi_dict_path, allow_pickle=True)["arr_0"].item()
    if log_progress:
        print(" Done.")
    frame_dicts = main_dict.values()
    if log_progress:
        frame_dicts = tqdm.tqdm(
            frame_dicts, total=len(main_dict), desc="Loading images"
        )
    for img_dict in frame_dicts:
        img_path = os.path.join(img_dict["adress"])
        contents = np.load(img_path)
        fwd_img = contents["Height_fwd"]
        try:
            bwd_img = contents["Height_bwd"]
        except KeyError:
            bwd_img = None
        pixel_size = img_dict["resolution"]
        height_profiles_fwd = []
        height_profiles_bwd = []
        for centerline in img_dict["centerlines"]:
            if centerline.size:
                xs_fwd, ys_fwd = extract_height_profile(centerline, fwd_img, pixel_size)
                if bwd_img is None:
                    xs_bwd = None
                    ys_bwd = None
                else:
                    xs_bwd, ys_bwd = extract_height_profile(
                        centerline, bwd_img, pixel_size
                    )
            else:
                xs_fwd = np.array([]).reshape((0, 2))
                ys_fwd = np.array([]).reshape((0, 2))
                xs_bwd = np.array([]).reshape((0, 2))
                ys_bwd = np.array([]).reshape((0, 2))
            height_profiles_fwd.append((xs_fwd, ys_fwd))
            height_profiles_bwd.append((xs_bwd, ys_bwd))
        img_dict["height_profiles_fwd"] = height_profiles_fwd
        img_dict["height_profiles_bwd"] = height_profiles_bwd
    return main_dict, masks_list, roi_dict


def get_roi_dir(cell_name, dataset):
    match (cell_name, dataset):
        case (str() | bytes() | os.PathLike(), None):
            dataset = ""
        case (int(), str() | bytes() | os.PathLike()):
            cell_name = f"ROI {cell_name}"
        case (str() | bytes() | os.PathLike(), str() | bytes() | os.PathLike()):
            pass
        case _:
            raise ValueError(
                f"The combination ({cell_name}, {dataset}) does not describe a ROI."
            )
    roi_dir = os.path.join( "data", "cells", dataset, cell_name)
    return roi_dir


def load_cell(
    cell_name,
    dataset=None,
    return_defects=False,
    cut_before_700=False,
    cut_after_700=False,
):
    if dataset == 'INH_before_700':
        cut_before_700 = True
        dataset = 'WT_INH_700min_2014'
    roi_dir = os.path.join(get_roi_dir(cell_name, dataset),"lines")
    centerlines = os.listdir(roi_dir)
    roi = []
    for filename in centerlines:
        filename = os.path.join(roi_dir, filename)
        with np.load(filename) as centerline:
            line = centerline["line"]
            xs = centerline["xs"]
            ys = centerline["ys"]
            timestamp = centerline["timestamp"].item()
            orientation = centerline["orientation"].item()
            no_defect = centerline["no_defect"].item()
            peaks = centerline["peaks"]
            troughs = centerline["troughs"]
            peaks_index = centerline['peaks_index']
            troughs_index = centerline['troughs_index']
            peaks_ROI = centerline["peaks_ROI"]
            troughs_ROI = centerline["troughs_ROI"]
            mask_index = centerline["mask_index"]
            dataset_name = centerline["dataset_name"]
            
            frame_data = {
                "filename" : filename,
                "line": line,
                "xs": xs,
                "ys": ys,
                "timestamp": timestamp,
                "orientation": Orientation(orientation),
                "no_defect": no_defect,
                "peaks": peaks,
                "troughs": troughs,
                "peaks_index": peaks_index,
                "troughs_index": troughs_index,
                "peaks_ROI": peaks_ROI,
                "troughs_ROI": troughs_ROI,
                "mask_index": mask_index,
                "dataset_name": dataset_name
            }
        if cut_before_700 and timestamp >= 700:
            continue
        if cut_after_700 and timestamp < 700:
            continue
        if no_defect or return_defects:
            roi.append(frame_data)
    roi.sort(key=operator.itemgetter("timestamp"))
    return roi


def _load_dataset(dataset, progress_bar, return_defects):
    if dataset == "INH_before_700":
        dataset = "WT_INH_700min_2014"
        cut_before_700 = True
        cut_after_700 = False
    elif dataset == "INH_after_700":
        dataset = "WT_INH_700min_2014"
        cut_before_700 = False
        cut_after_700 = True
    else:
        cut_before_700 = False
        cut_after_700 = False
    path = os.path.join( "data", "cells", dataset)
    directories = os.listdir(path)
    if progress_bar:
        directories = tqdm.tqdm(directories, desc=dataset)
    for roi_dir in directories:
        roi = load_cell(roi_dir, dataset, return_defects, cut_before_700, cut_after_700)
        roi_id =  roi_dir #int(roi_dir.split("ROI ")[-1])
        if roi:
            yield roi_id, roi


def load_dataset(dataset, progress_bar=True, return_defects=False):
    if dataset == "all_wild_types":
        datasets = [
            os.path.join("WT_mc2_55", "03-09-2014"),
            os.path.join("WT_mc2_55", "06-10-2015"),
            os.path.join("WT_mc2_55", "30-03-2015"),
            "INH_before_700",
        ]
    elif dataset is None or dataset == "all_datasets":
        datasets = find_all_datasets()
    else:
        datasets = [dataset]
    for dataset in datasets:
        for roi_id, roi in _load_dataset(dataset, progress_bar, return_defects):
            yield roi_id, roi


def topological_sort(roi_dict):
    rois = set(roi_dict.keys())
    order = []
    while rois:
        roi = rois.pop()
        while parent := roi_dict[roi]["Parent"]:
            roi = parent
        stack = [roi]
        while stack:
            roi = stack.pop()
            rois.discard(roi)
            order.append(roi)
            children = roi_dict[roi]["Children"]
            stack.extend(children)
    return order


def use_same_direction(reference, line, reference_angle, angle, xs, ys, xs_bwd, ys_bwd): 
    mean_ref= np.average(reference, axis=0).astype(np.int_)
    mean= np.average(line, axis=0).astype(np.int_)
    if reference_angle != angle:
        reference = rotation_line(angle - reference_angle, reference, mean_ref)
        
    newline = line - mean
    newreference = reference - mean_ref
    
    diffs = newreference[:, np.newaxis] - newline[np.newaxis]
    dists_sq = np.sum(diffs**2, axis=-1)
    if len(line) <= len(reference):
        order = dists_sq.argmin(axis=0)
    else:
        order = dists_sq.argmin(axis=1)
    if statistics.linear_regression(range(len(order)), order).slope < 0:
        line = line[::-1]
        xs = xs[-1] - xs[::-1]
        ys = ys[::-1]
        if xs_bwd is not None:
            xs_bwd = xs_bwd[-1] - xs_bwd[::-1]
            ys_bwd = ys_bwd[::-1]
    return line, xs, ys, xs_bwd, ys_bwd


def determine_orientation(reference, line):  
    if reference is None:
        return Orientation.UNKNOWN
    dist_start_start = math.dist(reference[0], line[0])
    dist_end_end = math.dist(reference[-1], line[-1])
    if dist_start_start < dist_end_end:
        return Orientation.OLD_POLE_NEW_POLE
    if dist_start_start > dist_end_end:
        return Orientation.NEW_POLE_OLD_POLE
    return Orientation.UNKNOWN


def extract_height_profile(centerline, img, pixel_size):
    xs = [0]
    for i in range(1, len(centerline)):
        x = xs[-1] + math.dist(centerline[i], centerline[i - 1])
        xs.append(x)
    xs = pixel_size * np.array(xs, dtype=np.float64)
    ys = [img[i, j] for i, j in centerline]
    ys = np.array(ys, dtype=np.float64)
    return xs, ys


def save_mask(          
    img_dict,
    mask_id,
    reference_line,
    reference_xs,
    reference_ys,
    reference_angle,
    orientation,
    quality,
    roi_dirname,
    mask_index,
    dataset_name,
    division_break=False
):
    line = img_dict["centerlines"][mask_id]
    angle = img_dict['angle']
    timestamp = img_dict["time"]
    pixel_size = img_dict["resolution"]
    xs, ys = img_dict["height_profiles_fwd"][mask_id]
    xs_bwd, ys_bwd = img_dict["height_profiles_bwd"][mask_id]
    
    params =  get_scaled_parameters(pnt_aligning=True, pixel_size=pixel_size)
    alignment_depth = params["depth_comparison_align"]
    
    
    if not line.size:
        return reference_line, reference_xs, reference_ys, reference_angle, orientation
    if reference_line != []:
        line, xs, ys, xs_bwd, ys_bwd = use_same_direction(
            reference_line[-1], line, reference_angle[-1], angle, xs, ys, xs_bwd, ys_bwd
        )
    else : 
        reference_line = []
        reference_xs = []
        reference_ys = []
        reference_angle = []
    if orientation is None:
        orientation = determine_orientation(reference_line[-1], line)
    
    params = get_scaled_parameters(pixel_size=pixel_size, pnt_preprocessing=True, pnt_filtering=True)
    no_defect = keep_centerline(xs, ys, pixel_size, **params)
    if not no_defect and xs_bwd is not None:
        no_defect_bwd = keep_centerline(xs_bwd, ys_bwd, pixel_size, **params)
        if no_defect_bwd:
            xs = xs_bwd
            ys = ys_bwd
            no_defect = no_defect_bwd
    no_defect = no_defect and quality
    params = get_scaled_parameters(pixel_size=pixel_size, pnt_preprocessing=True, pnt_aligning=True)
    xs, ys = align_with_reference(
        xs, ys, reference_xs, reference_ys, params, pixel_size, division=division_break
    )
    params = get_scaled_parameters(pixel_size=pixel_size, pnt_peaks_troughs=True)
    _, _, peaks, troughs = find_peaks_troughs(xs, ys, **params)
    mask_num = len(os.listdir(roi_dirname))
    filename = f"{mask_num:03d}.npz"
    path = os.path.join(roi_dirname,filename)

    np.savez(
        path,
        line=line,
        xs=xs,
        ys=ys,
        timestamp=timestamp,
        no_defect=no_defect,
        orientation=orientation.value,
        peaks=peaks,
        troughs=troughs,
        troughs_index=np.array([]),
        peaks_index=np.array([]),
        troughs_ROI=np.array([]),
        peaks_ROI=np.array([]),
        mask_index=mask_index,
        dataset_name=dataset_name
    )
    length_current = xs[-1] - xs[0]
    if  reference_xs == []:
        length_reference = None
    else:
        length_reference = max([elem[-1]-elem[0] for elem in reference_xs])
    if no_defect or (
        len(line) >= 10
        and (length_reference is None or length_current >= length_reference)
    ):
        if len(reference_line) >= alignment_depth:
            del(reference_line[0])
            del(reference_xs[0])
            del(reference_ys[0])
            del(reference_angle[0])
        reference_line.append(line)
        reference_xs.append(xs)
        reference_ys.append(ys)
        reference_angle.append(angle)
        
    return reference_line, reference_xs, reference_ys, reference_angle, orientation


def save_roi(
    roi,
    reference_line,
    reference_xs,
    reference_ys,
    reference_angle,
    masks_list,
    main_dict,
    roi_dirname,
    dataset_name
):
    
    if os.path.exists(roi_dirname):
        for file in os.listdir(roi_dirname):
            os.remove(os.path.join(roi_dirname, file))
    else:
        os.makedirs(roi_dirname)
    masks = roi["Mask IDs"]
    try:
        masks_quality = roi["masks_quality"]
        missing_masks_quality = False
    except KeyError:
        missing_masks_quality = True
        masks_quality = [True] * len(masks)
    if reference_line == []:
        orientation = Orientation.UNKNOWN
    else:
        orientation = None
    first_iter = True
    for mask_index, quality in zip(masks, masks_quality, strict=True):
        _, _, frame, mask_label = masks_list[mask_index]
        img_dict = main_dict[frame]
        if first_iter and reference_line != []:
            reference_line = [reference_line[-1]]
            reference_xs = [reference_xs[-1]]
            reference_ys = [reference_ys[-1]]
            reference_angle = [reference_angle[-1]]
            reference_line, reference_xs, reference_ys, reference_angle, orientation = save_mask(
                img_dict,
                mask_label - 1,
                reference_line,
                reference_xs,
                reference_ys,
                reference_angle,
                orientation,
                quality,
                roi_dirname,
                mask_index,
                dataset_name,
                division_break=True
            )
            if len(reference_line)>1:
                del(reference_line[0])
                del(reference_xs[0])
                del(reference_ys[0])
                del(reference_angle[0])
        
        else:
            reference_line, reference_xs, reference_ys, reference_angle, orientation = save_mask(
                img_dict,
                mask_label - 1,
                reference_line,
                reference_xs,
                reference_ys,
                reference_angle,
                orientation,
                quality,
                roi_dirname,
                mask_index,
                dataset_name
            )
        first_iter = False
    return (reference_line, reference_xs, reference_ys, reference_angle), missing_masks_quality


def save_dataset(dataset, log_progress=True):
    if os.path.exists(os.path.join( "data", "cells", dataset,'')):
        rmtree(os.path.join( "data", "cells", dataset,''))
    dataset_missing_masks_quality = False
    if log_progress:
        print("Processing dataset", dataset)
    main_dict, masks_list, roi_dict = load_data(dataset, log_progress)
    roi_names = topological_sort(roi_dict)
    if log_progress:
        roi_names = tqdm.tqdm(roi_names, desc="Processing ROIs")
    references = {}
    for roi_name in roi_names:
        roi = roi_dict[roi_name]
        roi_dir = os.path.join( "data", "cells", dataset, roi_name, "lines")
        reference = references.pop(roi_name, ([], [], [], []))
        reference, missing_masks_quality = save_roi(
            roi, *reference, masks_list, main_dict, roi_dir, dataset
        )
        dataset_missing_masks_quality = (
            dataset_missing_masks_quality or missing_masks_quality
        )
        for child in roi["Children"]:
            if reference != [] and len(reference[0]) >= 1:
                references[child] = [[elem[-1]] for elem in reference]
    if dataset_missing_masks_quality:
        global datasets_missing_masks_quality
        datasets_missing_masks_quality.append(dataset)
    if log_progress:
        print("\n\n")



def get_peak_troughs_lineage_lists(dataset, roi_id):
    
    params=get_scaled_parameters(paths_and_names=True)
    direct = os.path.join(params["dir_cells"], dataset, roi_id, params["dir_cells_list"])
    pnt_list_name = os.path.join(direct, params['pnt_list_name'])
    pnt_ROI_name = os.path.join(direct, params['pnt_ROI_name'])
    pnt_list = np.load(pnt_list_name, allow_pickle=True)['arr_0']
    pnt_ROI = np.load(pnt_ROI_name, allow_pickle=True)['arr_0'].item()
    
    return pnt_list, pnt_ROI

def compute_dataset(dataset):
    save_dataset(dataset)
        
    for roi_dir, cell in load_dataset(dataset, False):
        peak_troughs_lineage(dataset, cell, roi_dir)


def main(datasets=None):
     
    if datasets is None:
        datasets = find_all_datasets()
    else:
        params = get_scaled_parameters(data_set=True)
        if datasets in params.keys():
            datasets = params[datasets]
        elif isinstance(datasets, str): 
            raise NameError('This directory does not exist')


    with Pool(processes=8) as pool:
        for _ in pool.imap_unordered(compute_dataset, datasets):
            pass
            
    global datasets_missing_masks_quality
    if datasets_missing_masks_quality:
        print(
            'The following dataset(s) do not have the "masks_quality" attribute.\n'
            f"{datasets_missing_masks_quality}"
        )


if __name__ == "__main__":
    main('all') 
