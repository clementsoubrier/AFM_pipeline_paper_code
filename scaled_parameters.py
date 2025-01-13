import os
import numpy as np

'''
Paths
'''
# data_sets

DATA_SET = ["WT_mc2_55/06-10-2015",
            "WT_mc2_55/30-03-2015",
            "WT_mc2_55/03-09-2014",
            'WT_INH_700min_2014'
            ] 



DATA_SET_WT_NO_DRUG = ["WT_mc2_55/06-10-2015",
                       "WT_mc2_55/30-03-2015",
                       "WT_mc2_55/03-09-2014",
                       "INH_before_700"]


DATA_SET_GOOD_QUAL = ["WT_mc2_55/30-03-2015"
                      ] 

                        
INH_BEF = ["INH_before_700"]
INH_AFT = ["INH_after_700"]

                
            

# path_and_names = 
INITIAL_DATA = os.path.join("..","data2")       # directory with the initial data (logs)
DATA_DIREC = os.path.join('Processed_data', 'datasets')   # main processed directory
FINAL_DATA = "final_data"                       # sub directory with the final images
RESULT_DIR = os.path.join('Processed_data', 'results')    # result directory (stats, videos, images)
DIR_PLOT_STAT = os.path.join(RESULT_DIR,'plot_stat') # result directory stats
DATA_CELL = os.path.join('Processed_data', 'cells')       # directory of the ROIs
LINE_DIR = "lines"                              # data of the ROIs (centerlines, peaks and troughs)
FEATURES_TRACKING = "features_tracking"         # peaks and troughs tracking directory
FINAL_IMG_DIR = os.path.join("..","Images","All_images")

MAIN_DICTIONNARY_NAME = 'main_dictionnary.npz'
MASKS_LIST_NAME = 'masks_list.npz'
ROI_DICT_NAME = 'ROI_dict.npz'
ROI_MASKS_LIST_NAME = 'masks_ROI_list.npz'

LINEAGE_MATRIX_NAME = 'lineage_matrix.npy'      
BOOL_MATRIX_NAME = "bool_matrix.npy"
LINK_MATRIX_NAME = 'link_matrix.npy'


PNT_LIST_NAME = 'peaks_troughs_list.npz'
PNT_ROI_NAME = 'peaks_troughs_ROI_list.npz'


'''
Processing 
'''


# cellpose
CEL_MODEL_TYPE = 'cyto'                 # Cellpose segmentation algo
DENOISING_MODEL = "denoise_cyto3"       # Cellpose denoizing model

CEL_CHANNELS = [0,0]                    # Define channels to run segementation on grayscale=0, R=1, G=2, B=3; channels = [cytoplasm, nucleus]; nucleus=0 if no nucleus
CEL_DIAMETER_PARAM = 1                  # parameter to adjust the expected size (in pixels) of bacteria. Incease if cellpose detects too small masks, decrease if it don't detects small mask/ fusion them. Should be around 1 
CEL_FLOW_THRESHOLD = 0.8                # segmentation parameter
CEL_PROB_THRESHOLD = 0.4                # segmentation parameter
CEL_GPU = True                          # if you have access to a GPU

# img_treament
RATIO_ERASING_MASKS = 0.2               # erasing small masks That have a smaller relative size
RATIO_SATURATION = 0.1                  # erasing masks that have a ratio of saturated surface bigger than this fraction
MAX_PIXEL_IM = 1024                     # maximum height or len of a picture
CENTERLINE_CROP_PARAM = 2               # minimum ratio of two brench length to erase the small branch
THRESHOLD_SCARS_DIRECTORY = {'Height':0.9,
                             "Peak Force Error": 0.1,
                             "other":0.5} # specific parameters for initial image tratment

# plot
MASKS_COLORS = [[255,0,0],
                [0,255,0],
                [0,0,255],
                [255,255,0],
                [255,0,255],
                [0,255,255],
                [255,204,130],
                [130,255,204],
                [130,0,255],
                [130,204,255]]

# video
IMAGE_FOLDER_VIDEO = '../Python_code/img/'
VIDEO_NAME = 'video.avi'

# lineage_tree
FINAL_THRESH = 0.75                     # minimal threshold to consider a daughter-mother relation 
THRES_MIN_DIVISION = 0.7                # threshold to consider that there is a division and not just a break in the ROI 
COMPARISON_THRES = 0.7                  # threshold to consider that two children are actually linked
MIN_LEN_ROI = 3                         # minimum number of frames in an ROI
CHILD_DIV_THRES = 0.34                  # fraction of the preserved area to consider child and parent relation for masks (should be less than 0.5 to take into account the division)
MAX_DIFF_TIME = 70                      # maximum time between 2 frames to compore them (in seconds)


# physical_feature
PHYS_DERIV_PREC = 0.1                   # mu m  length of the centerline to compute the vectors tangent to the centerlines
PHYS_NORMAL_PREC =  0.2                 # mu m  length of the averaged portion along the normal to the centerlines
PHYS_TANGENT_PREC = 0.1                 # mu m  length of the averaged portion along the tangent to the centerlines



'''
Peaks and troughs parameters (pnt_)
'''

# pnt_preprocessing
KERNEL_SIZE = 0.1575                    # micrometers
SLOPE_STD_CUT = 2.5                     # dimensionless
WINDOW_SIZE = 0.1575                    # micrometers

# pnt_peaks_troughs
SMOOTH_KERNEL_STD = 0.118               # micrometers
MIN_FEATURE_WIDTH = 0.315               # micrometers
MIN_FEATURE_DEPTH = 11.75               # nanometers

# pnt_filtering                         # filtering bad centerlines
MIN_CENTERLINE_LEN = 2.5                # micrometers
MIN_PREPROCESSED_CENTERLINE_LEN = 1.5   # micrometers
MAX_SLOPE_STD = 3                       # dimensionless
MAX_SLOPE = 1500                        # dimensionless (nanometers per micrometers)
MAX_VAR_SLOPE = 4000                    # dimensionless (nanometers per micrometers)

# pnt_aligning                          # aligning centerlines of a same ROI 
MAX_TRANSLATION = 1.18                  # micrometers
MAX_RELATIVE_TRANSLATION = 0.5            # maximum relative translation for centerlines comparision
ALIGNMENT_PENALTY = 10000               # dimensionless (nanometers per micrometers) 
WINDOW_RELATIVE_SIZE = 0.1              # size of the window relative to the centerline size
if MAX_RELATIVE_TRANSLATION+WINDOW_RELATIVE_SIZE>1:
    raise ValueError("MAX_RELATIVE_TRANSLATION+WINDOW_RELATIVE_SIZE<=1 is required") 
QUANTILE_SIZE = 0.3                     # Quantile value to select alignment
QUANTILE_SIZE_DIVISION = 0.4            # Quantile value to select alignment after division
WINDOW_RELATIVE_SIZE_DIVISION = 0.2     # size of the window relative to the centerline size, for alignment after division
if MAX_RELATIVE_TRANSLATION+WINDOW_RELATIVE_SIZE_DIVISION>1:
    raise ValueError("MAX_RELATIVE_TRANSLATION+WINDOW_RELATIVE_SIZE_DIVISION<=1 is required") 
ALIGNMENT_PENALTY_DIVISION = 20000      # dimensionless (nanometers per micrometers), penalty when daughter cells are ticking inside mother cells 
DEPTH_COMPARISON_ALIGN = 5              # int, number of centerline to compare for alignment

# pnt_tracking                          # tracking of the suface features (peaks and troughs over generations)
MAX_TIME = 70                           # mn  max time between 2 frames to compare them
FIRST_MAX_XDRIFT = 0.5                  # mu m  max x coordinate difference between 2 frames to compare them
FINAL_MAX_XDRIFT = 0.8                  # mu m  max x coordinate difference between 2 frames to compare them
MAX_YDRIFT = 500                        # nm  max y coordinate difference between 2 frames to compare them

#stats
BIN_NUMBER_HIST_FEAT_CREA = 20          # number of bins for the histogram for the feature creation plot
BIN_NUMBER_HIST_COUNT = 40              # number of bins for the histogram for the feature count plot
SMOOTHING_HIST_FEAT_CREA = 400          # smoothing parameter for the feature creation plot
SMOOTHING_HIST_COUNT = 3                # smoothing parameter for the feature count plot
POLE_REGION_SIZE = 1.5                  # mu m physical size of the pole region
DIV_MAX_SUPERPOSITION = 0.5             # mu m maximum superposition of the 2 daughter centerlines admissible
DIV_MAX_DIST_FROM_MOTH = 1              # mu m maximum distance from mother closest boundary
DIV_MIN_DAUGTHER_SIZE = 2               # mu m min size of other daughter if 1 daughter selected
DIV_CURV_PHY_WINDOW = 0.2               # mu m min size of window for computing local curvature
DIV_CURV_SMOOTH = 0.1                   # mu m min size of guassian smoothing std for computing local curvature



def get_scaled_parameters(
    paths_and_names=False,
    data_set=False,
    cellpose=False,
    img_treatment=False,
    plot=False,
    video=False,
    lineage_tree=False,
    physical_feature=False,
    pixel_size=None,
    pnt_preprocessing=False,
    pnt_peaks_troughs=False,
    pnt_filtering=False,
    pnt_aligning=False,
    pnt_tracking=False,
    stats=False
):
    params = {}

    if paths_and_names:
        params["initial_data_direc"] = INITIAL_DATA
        params["main_data_direc"] = DATA_DIREC
        params["final_data_direc"] = FINAL_DATA
        params["results_direc"] = RESULT_DIR
        params["dir_plot_stat"] = DIR_PLOT_STAT
        params["dir_cells"] = DATA_CELL    
        params["dir_cells_data"] = LINE_DIR 
        params["dir_cells_list"] = FEATURES_TRACKING 
        params["final_img_dir"] = FINAL_IMG_DIR

        params["main_dict_name"] = MAIN_DICTIONNARY_NAME
        params["masks_list_name"] = MASKS_LIST_NAME
        params["roi_dict_name"] = ROI_DICT_NAME
        params['roi_masks_list_name'] = ROI_MASKS_LIST_NAME

        params['lineage_matrix_name'] = LINEAGE_MATRIX_NAME
        params['bool_matrix_name'] = BOOL_MATRIX_NAME
        params['link_matrix_name'] = LINK_MATRIX_NAME


        params['pnt_list_name'] = PNT_LIST_NAME
        params['pnt_ROI_name'] = PNT_ROI_NAME

    if data_set:
        for data in DATA_SET:
            params[data] = [data]
        params['all'] = DATA_SET
        params['WT_no_drug'] = DATA_SET_WT_NO_DRUG
        params['good'] = DATA_SET_GOOD_QUAL
        params["INH_before_700"] = INH_BEF
        params["INH_after_700"] = INH_AFT


    if cellpose:
        params['cel_model_type'] = CEL_MODEL_TYPE
        params['cel_denoise_model'] = DENOISING_MODEL
        params['cel_channels'] = CEL_CHANNELS
        params['cel_diameter_param'] = CEL_DIAMETER_PARAM
        params['cel_flow_threshold'] = CEL_FLOW_THRESHOLD
        params['cel_prob_threshold'] = CEL_PROB_THRESHOLD
        params['cel_gpu'] = CEL_GPU

    if img_treatment:
        params["ratio_erasing_masks"] = RATIO_ERASING_MASKS
        params["ratio_saturation"] = RATIO_SATURATION
        params["max_pixel_im"] = MAX_PIXEL_IM
        params["centerline_crop_param"] = CENTERLINE_CROP_PARAM
        params["threshold_scars_directory"] = THRESHOLD_SCARS_DIRECTORY
        
    if plot:
        params["masks_colors"] = MASKS_COLORS

    if video:
        params["image_folder_video"] = IMAGE_FOLDER_VIDEO
        params["video_name"] = VIDEO_NAME

    if lineage_tree:
        params["final_thresh"] = FINAL_THRESH
        params["thres_min_division"] = THRES_MIN_DIVISION
        params["comparison_thres"] = COMPARISON_THRES
        params["min_len_ROI"] = MIN_LEN_ROI
        params["child_div_thres"] = CHILD_DIV_THRES
        params["max_diff_time"] = MAX_DIFF_TIME


    if physical_feature:
        params["phys_deriv_prec"] = PHYS_DERIV_PREC 
        params["phys_normal_prec"] = PHYS_NORMAL_PREC
        params["phys_tangent_prec"] = PHYS_TANGENT_PREC
        
    if pnt_preprocessing:
        params["kernel_len"] = 1 + round(KERNEL_SIZE / pixel_size)
        params["std_cut"] = SLOPE_STD_CUT
        params["window"] = 1 + round(WINDOW_SIZE / pixel_size)

    if pnt_peaks_troughs:
        params["smooth_std"] = SMOOTH_KERNEL_STD / pixel_size
        params["min_width"] = MIN_FEATURE_WIDTH
        params["min_depth"] = MIN_FEATURE_DEPTH

    if pnt_filtering:
        params["min_len"] = MIN_CENTERLINE_LEN
        params["min_prep_len"] = MIN_PREPROCESSED_CENTERLINE_LEN
        params["max_der_std"] = MAX_SLOPE_STD
        params["max_der"] = MAX_SLOPE
        params["max_var_der"] = MAX_VAR_SLOPE

    if pnt_aligning:
        params["max_translation"] = max(1, round(MAX_TRANSLATION / pixel_size))
        params["max_relative_translation"] = MAX_RELATIVE_TRANSLATION
        params["penalty"] = ALIGNMENT_PENALTY
        params["window_relative_size"] = WINDOW_RELATIVE_SIZE
        params["quantile_size"] = QUANTILE_SIZE
        params["smooth_std"] = SMOOTH_KERNEL_STD / pixel_size
        params["quantile_size_division"] = QUANTILE_SIZE_DIVISION
        params["window_relative_size_division"] = WINDOW_RELATIVE_SIZE_DIVISION
        params["penalty_division"] = ALIGNMENT_PENALTY_DIVISION
        params["depth_comparison_align"] = DEPTH_COMPARISON_ALIGN
    
    if pnt_tracking:
        params["max_time"] = MAX_TIME
        params["first_max_xdrift"] = FIRST_MAX_XDRIFT 
        params["final_max_xdrift"] = FINAL_MAX_XDRIFT 
        params["max_ydrift"] = MAX_YDRIFT
    
    if stats:
         params["bin_number_hist_feat_crea"] = BIN_NUMBER_HIST_FEAT_CREA
         params["bin_number_hist_count"] = BIN_NUMBER_HIST_COUNT
         params["smoothing_hist_feat_crea"] = SMOOTHING_HIST_FEAT_CREA
         params["smoothing_hist_fcount"] = SMOOTHING_HIST_FEAT_CREA
         params["pole_region_size"] = POLE_REGION_SIZE
         params["div_max_superposition"] = DIV_MAX_SUPERPOSITION
         params["div_max_dist_from_moth"] = DIV_MAX_DIST_FROM_MOTH
         params['div_min_daugther_size'] = DIV_MIN_DAUGTHER_SIZE
         params['div_curv_phy_window'] = DIV_CURV_PHY_WINDOW
         params['div_curv_smooth'] = DIV_CURV_SMOOTH


    return params
