import itertools
import os
import sys
import statistics
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)
from scaled_parameters import get_scaled_parameters
from peaks_troughs.group_by_cell import Orientation, load_cell, load_dataset


#Computing statistics for cell growth with different datasets


def p_value_to_str(p):
    if math.isnan(p):
        return 'NaN'
    elif p>=0.05:
        return f"P={p:.2e}"
    elif p>=0.01:
        return f"P={p:.2e} *"
    elif p>=0.001:
        return f"P={p:.2e} **"
    else :
        return f"P={p:.2e} ***"

def print_stats(data_list):
    for i,elem in enumerate(data_list):
        print(f'data nr {i}')
        print(f'mean : {np.mean(elem)}')
        print(f'1st quartile : {np.quantile(elem,0.25)}')
        print(f'median : {np.median(elem)}')
        print(f'3rd quartile : {np.quantile(elem,0.75)}')
        print(f'std : {np.std(elem)}')
       

def extract_growth(roi):
    t_0 = roi[0]["timestamp"]
    roi = iter(roi)
    while True:
        try:
            frame_data = next(roi)
        except StopIteration:
            return [], [], [], []
        orientation = Orientation(frame_data["orientation"])
        xs = frame_data["xs"]
        match orientation:
            case Orientation.UNKNOWN:
                continue
            case Orientation.NEW_POLE_OLD_POLE:
                old_pole_0 = xs[-1]
                new_pole_0 = xs[0]
            case Orientation.OLD_POLE_NEW_POLE:
                old_pole_0 = xs[0]
                new_pole_0 = xs[-1]
        break
    
    
    
    timestamps = []
    old_pole_growth = []
    new_pole_growth = []
    overall_growth = []
    for frame_data in roi:
        timestamp = frame_data["timestamp"]
        orientation = Orientation(frame_data["orientation"])
        xs = frame_data["xs"]
        match orientation:
            case Orientation.UNKNOWN:
                continue
            case Orientation.NEW_POLE_OLD_POLE:
                old_pole_var = xs[-1] - old_pole_0
                new_pole_var = new_pole_0 - xs[0]
            case Orientation.OLD_POLE_NEW_POLE:
                old_pole_var = old_pole_0 - xs[0]
                new_pole_var = xs[-1] - new_pole_0
        timestamps.append(timestamp - t_0)
        old_pole_growth.append(old_pole_var)
        new_pole_growth.append(new_pole_var)
        overall_growth.append(abs(xs[-1]-xs[0]))


    return np.array(timestamps), np.array(old_pole_growth), np.array(new_pole_growth), np.array(overall_growth)

def str_orient(orientation):
    match orientation:
        case Orientation.UNKNOWN:
            ori_str = "unknown orientation"
        case Orientation.NEW_POLE_OLD_POLE:
            ori_str = "new to old orientation"
        case Orientation.OLD_POLE_NEW_POLE:
            ori_str = "old new orientation"
    return ori_str



def plot_piecewise_linear_reg_old_new(timestamps, old_pole_growth, new_pole_growth, roi_name, outlier_detection=False):
    plt.figure()

    new_times = timestamps
    old_times = timestamps
    
    
    if outlier_detection:
        new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
        old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
        
    a_1, a_2, b, t = piecewise_pointwise_linear_regression(new_times, new_pole_growth)
    t_0 = new_times[0]
    t_f = new_times[-1]
    piecewise_times = [t_0, t, t_f]
    p_0 = b - a_1 * t
    p_f = b + a_2 * (t_f - t)
    piecewise_y = [p_0, b, p_f]
    
    plt.scatter(new_times, new_pole_growth, label="new pole length", color="C2")
    plt.plot(
        piecewise_times, piecewise_y, color="C2"
    )
    
    plt.scatter(old_times, old_pole_growth, label="old pole length", color="C0")
    slope, intercept = statistics.linear_regression(old_times, old_pole_growth)
    lin_int = slope*old_times+intercept
    plt.plot(old_times, lin_int,color="C0")
    
    
    y_min = min(min(new_pole_growth), min(piecewise_y), min(old_pole_growth), min(lin_int))
    y_max = max(max(new_pole_growth), max(piecewise_y), max(old_pole_growth), max(lin_int))
    plt.plot(
        [t, t],
        [y_min, y_max],
        linestyle="dashed",
        label=f"NETO: {int(t)} minutes",
        color="C1",
    )
    
    
    plt.plot([], [], " ", label=f"Slope ratio: {a_2 / a_1:.2f}")
    plt.annotate(
        f"{1000 * a_1:.1f} nm / minute", (t_0 + (0.05 * (t_f - t_0)), p_0)
    )
    plt.annotate(
        f"{1000 * a_2:.1f} nm / minute",
        (t_f - (0.05 * (t_f - t_0)), p_f),
        ha="right",
        va="top",
    )
    
    
    plt.legend()
    plt.xlabel("Time since cell birth (minutes)")
    plt.ylabel("Pole length (µm)")
    plt.title(f"Growth of {roi_name}")
    plt.show()


def plot_piecewise_linear_reg(timestamps,old_pole_growth, new_pole_growth, overall_growth, roi_name, outlier_detection=False):
    plt.figure()
    overall_growth = np.array(overall_growth)
    overall_times = np.array(timestamps)
    
    
    if outlier_detection:
        overall_times, overall_growth =   remove_length_outliers(overall_times , overall_growth)
        
    a_1, a_2, b, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
    t_0 = overall_times[0]
    t_f = overall_times[-1]
    piecewise_times = [t_0, t, t_f]
    p_0 = b - a_1 * t
    p_f = b + a_2 * (t_f - t)
    piecewise_y = [p_0, b, p_f]
    y_min = min(min(overall_growth), min(piecewise_y))
    y_max = max(max(overall_growth), max(piecewise_y))
    plt.scatter(overall_times, overall_growth, label="cell length variation", color="C2")
    plt.plot(
        piecewise_times, piecewise_y, label="piecewise linear fit", color="C0"
    )
    plt.plot(
        [t, t],
        [y_min, y_max],
        linestyle="dashed",
        label=f"NETO: {int(t)} minutes",
        color="C1",
    )
    plt.plot([], [], " ", label=f"Slope ratio: {a_2 / a_1:.2f}")
    plt.annotate(
        f"{1000 * a_1:.1f} nm / minute", (t_0 + (0.05 * (t_f - t_0)), p_0)
    )
    plt.annotate(
        f"{1000 * a_2:.1f} nm / minute",
        (t_f - (0.05 * (t_f - t_0)), p_f),
        ha="right",
        va="top",
    )
    plt.legend()
    
    plt.xlabel("Time since cell birth (minutes)")
    plt.ylabel("Length variation (µm)")
    plt.title(f"Growth of {roi_name}")
    plt.show()
    

def remove_worst_length_outliers(times, y):
    dy = y[1:] - y[:-1]
    dt = times[1:] - times[:-1]
    dydt = dy / dt
    d2ydt = dydt[1:] - dydt[:-1]
    dtbis = (times[2:] - times[:-2]) / 2
    d2ydt2 = d2ydt / dtbis
    q1, q3 = np.percentile(d2ydt2, [25, 75])
    clipped = np.clip(d2ydt2, q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    errors = np.abs(d2ydt2 - clipped)
    i = np.argmax(errors)
    if errors[i] > 0:
        times = np.concatenate([times[: i + 1], times[i + 2 :]])
        y = np.concatenate([y[: i + 1], y[i + 2 :]])
    return times, y


def remove_length_outliers(times, y):
    while len(times) > 0:
        new_times, new_y = remove_worst_length_outliers(times, y)
        if len(new_times) == len(times):
            return np.array(times), np.array(y)
        times = new_times
        y = new_y
    raise RuntimeError


def piecewise_pointwise_linear_regression(times, y):
    assert times[-1] - times[0] >= 75
    best_err = np.inf
    best_a_1 = None
    best_a_2 = None
    best_b = None
    best_t = None
    errs = []
    for i, (t_1, t_2) in enumerate(itertools.pairwise(times), 1):
        if t_2 < 45 or t_1 > times[-1] - 30:
            continue
        t_1 = max(t_1, 45)
        t_2 = min(t_2, times[-1] - 30)
        for t in np.linspace(t_1, t_2, round(t_2 - t_1) + 1):
            mat = np.zeros((len(times), 3))
            mat[:i, 0] = times[:i] - t
            mat[i:, 1] = times[i:] - t
            mat[:, 2] = 1
            coefs = np.linalg.lstsq(mat, y, None)[0]
            a_1, a_2, b = coefs
            err = np.sum((y - np.dot(mat, coefs)) ** 2)
            errs.append(err)
            if err < best_err:
                best_err = err
                best_a_1 = a_1
                best_a_2 = a_2
                best_b = b
                best_t = t
    return best_a_1, best_a_2, best_b, best_t

def compute_slopes(times, y, t):
    mat = np.zeros((len(times), 3))
    i = np.nonzero(times >=t)[0][0]
    mat[:i, 0] = times[:i] - t
    mat[i:, 1] = times[i:] - t
    mat[:, 2] = 1
    coefs = np.linalg.lstsq(mat, y, None)[0]
    a_1, a_2, b = coefs
    return a_1, a_2, b
    


def plot_growth_all_cent(dataset, old_new=True, overall=True,outlier_detection=False):
    
    
    for roi_name, roi in load_dataset(dataset, False):
        timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
        if len(timestamps)>=5 and timestamps[-1]-timestamps[0] > 75:
            if old_new:
                plot_piecewise_linear_reg_old_new(timestamps, old_pole_growth, new_pole_growth, roi_name, outlier_detection=outlier_detection)
            if  overall:
                plot_piecewise_linear_reg(timestamps, old_pole_growth, new_pole_growth, overall_growth, roi_name, outlier_detection=outlier_detection)



def compute_growth_stats(datasetnames, outlier_detection=False):
    
    tot_growth = []
    overall_neto = []
    old_new_neto = []
    overall_slopes = []  # first, second
    old_new_slopes = []  # first new, second new, old
    params = get_scaled_parameters(data_set=True)
    
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    for dataset in datasets:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                tot_growth.append((overall_growth[-1]-overall_growth[0])/(timestamps[-1]-timestamps[0]))
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)
                    
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(new_times, new_pole_growth)
                
                if 0<a_1<=a_2:
                    
                    old_new_neto.append(t)
                
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
                overall_slopes.append([a_1,a_2])
                if 0<a_1<=a_2:
                    
                    overall_neto.append(t)
                    # the neto is computed from the overall length because it is more stable
                    new_a_1, new_a_2, _, = compute_slopes(new_times, new_pole_growth, t)
                    a_3, a_4, _ = compute_slopes(old_times, old_pole_growth, t)
                    mat = np.zeros((len(old_times), 2))
                    mat[:, 0] = old_times
                    mat[:, 1] = 1
                    a_5, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                    
                    old_new_slopes.append([new_a_1, new_a_2, a_3, a_4, a_5])
                
                
                
    overall_slopes = np.array(overall_slopes)      
    old_new_slopes = np.array(old_new_slopes)
    
    title = f"Neto statistics with 2 methods and dataset {datasetnames}"
    _, ax = plt.subplots()
    ax.boxplot([old_new_neto,overall_neto], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([old_new_neto,overall_neto])
    ax.set_xticklabels(["new pole elongation", "overall elongation"])
    ax.set_ylabel(r"Time $(min)$")
    ax.set_title(title)
    pvalue = stats.ttest_ind(old_new_neto,overall_neto).pvalue
    print(pvalue)
    ax.annotate(p_value_to_str(pvalue),(1.3,120))
    
    title = f"Overall cell elongation speed with dataset {datasetnames}"
    _, ax = plt.subplots()
    ax.boxplot([overall_slopes[:,0],overall_slopes[:,1],overall_slopes[:,1]-overall_slopes[:,0]], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([overall_slopes[:,0],overall_slopes[:,1],overall_slopes[:,1]-overall_slopes[:,0]])
    ax.set_title(title)
    ax.set_ylabel(r"elongation speed ($\mu m (min)^{-1}$)")
    ax.set_xticklabels(["Before NETO", "After NETO","Difference before/after"])
    pvalue = stats.ttest_ind(overall_slopes[:,0],overall_slopes[:,1]).pvalue
    print(pvalue)
    ax.annotate(p_value_to_str(pvalue),(1.3,0.02))

    
    title = f"Comparision of elongation speed for dataset \n {datasetnames} and {len(old_new_slopes[:,1])} cells"
    _, ax = plt.subplots()
    ax.boxplot([1000*old_new_slopes[:,1], 1000*old_new_slopes[:,4]], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([1000*old_new_slopes[:,1], 1000*old_new_slopes[:,4]])
    ax.set_xticklabels([f"new pole \n after NETO", "old pole"])
    ax.set_ylabel(r"elongation speed ($nm (min)^{-1}$)")
    ax.set_title(title)
    pvalue = stats.ttest_ind(old_new_slopes[:,1], old_new_slopes[:,4]).pvalue
    print(pvalue)
    x1 = 1
    x2 = 2 
    y = 12
    h = 0.1
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+ h, p_value_to_str(pvalue), ha='center', va='bottom')
    plt.tight_layout()
    
    
    plt.show()   

def Neto_comparision(data1, data2, outlier_detection=True):
    
    
    
    
    overall_neto1 = []
    overall_neto2 = []
    params = get_scaled_parameters(data_set=True)
    
    if data1 in params.keys():
        datasets1 = params[data1]
    elif isinstance(data1, str): 
        raise NameError('This directory does not exist')
    else :
        datasets1 = data1 
    
    for dataset in datasets1:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)
                    
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(new_times, new_pole_growth)
                
                if 0<a_1<=a_2:
                    
                    overall_neto1.append(t)
    if data2 in params.keys():
        datasets2 = params[data2]
    elif isinstance(data2, str): 
        raise NameError('This directory does not exist')
    else :
        datasets2 = data2
    
    for dataset in datasets2:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)
                    
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(new_times, new_pole_growth)
                
                if 0<a_1<=a_2:
                        
                    overall_neto2.append(t)
                        
                        
    overall_neto1 = np.array(overall_neto1)                    
    overall_neto2 = np.array(overall_neto2)
    
    title = f"Neto statistics with 2 methods and datasets \n  {data1} and {data2}"
    _, ax = plt.subplots()
    ax.boxplot([overall_neto1, overall_neto2], showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([overall_neto1, overall_neto2])
    ax.set_xticklabels(["data1", "data2"])
    ax.set_ylabel(r"Time $(min)$")
    ax.set_title(title)
    pvalue = stats.ttest_ind(overall_neto1, overall_neto2).pvalue
    print(pvalue)
    ax.annotate(p_value_to_str(pvalue),(1.3,120))    
    
    plt.tight_layout()
    plt.show()               
                    
                    

def compute_pole_growth_stats(datasetnames, outlier_detection=False, plot=False):
    
    tot_growth = []
    overall_slopes = []  # first, second
    old_new_slopes = []  # first new, second new, old
    after_neto_slopes = [] # first new, second old
    params = get_scaled_parameters(data_set=True)
    
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    for dataset in datasets:
        for roi_id, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                tot_growth.append((overall_growth[-1]-overall_growth[0])/(timestamps[-1]-timestamps[0]))
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)
 
                # the neto is computed from the overall length because it is more stable
                a_1, a_2, b, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
                overall_slopes.append([a_1,a_2])
                    
                new_a_1, new_a_2, c, = compute_slopes(new_times, new_pole_growth, t)
                a_3, a_4, d = compute_slopes(old_times, old_pole_growth, t)
                mat = np.zeros((len(old_times), 2))
                mat[:, 0] = old_times
                mat[:, 1] = 1
                a_5, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                old_new_slopes.append([new_a_1, new_a_2, a_3, a_4, a_5])
                if plot:
                    
                    plt.figure()
                    plt.scatter(overall_times, overall_growth - overall_growth[0]+1, color = 'k', label = 'overall')
                    plt.scatter(old_times, old_pole_growth +0.5, color = 'r', label = 'old pole')
                    plt.scatter(new_times, new_pole_growth, color = 'b', label = 'new pole')
                    plt.plot([0, t, overall_times[-1]], [b-a_1*t - overall_growth[0]+1, b - overall_growth[0]+1, b - overall_growth[0]+1 + a_2* (overall_times[-1]-t)], color = 'k')
                    plt.plot([0, t, new_times[-1]], [c-new_a_1*t, c, c+ new_a_2* (new_times[-1]-t)], color = 'b')
                    mat = np.zeros((len(old_times), 2))
                    mat[:, 0] = old_times
                    mat[:, 1] = 1
                    sl, b_o = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                    plt.plot([0, old_times[-1]], [b_o+0.5,b_o+0.5+sl*old_times[-1]], color = 'r')
                    plt.plot(
                        [t, t],
                        [0, overall_growth[-1]- overall_growth[0]+1],
                        linestyle="dashed",
                        label=f"NETO",
                        color="k",)
                    plt.legend()
                    plt.title(f'Cell and pole elongation of {roi_id}, \n with dataset {datasetnames}')
                    plt.xlabel(r'time $(min)$')
                    plt.ylabel(r'elongation $(\mu m)$')
                    plt.tight_layout()
                    
                if timestamps[-1] - t >75:
                    maskn = new_times >=  t + (new_times[-1] - t) /2
                    masko = old_times >=  t + (old_times[-1] - t) /2
                    if np.sum(maskn) >= 5 and np.sum(masko) >= 5:
                        o_p = old_pole_growth[masko]
                        n_p = new_pole_growth[maskn]
                        new_times = new_times[maskn]
                        old_times = old_times[masko]
                        mato = np.zeros((len(old_times),2))
                        mato[:, 0] = old_times
                        mato[:, 1] = 1
                        slo, _ = np.linalg.lstsq(mato, o_p, None)[0]
                        
                        matn = np.zeros((len(new_times),2))
                        matn[:, 0] = new_times
                        matn[:, 1] = 1
                        sln, _ = np.linalg.lstsq(matn, n_p, None)[0]
                        after_neto_slopes.append([sln, slo])
                
                
    overall_slopes = 1000*np.array(overall_slopes)      
    old_new_slopes = 1000*np.array(old_new_slopes)
    after_neto_slopes = 1000*np.array(after_neto_slopes)
    
    title = f"Pole elongation speed \n with dataset {datasetnames} and {len(old_new_slopes[:,0])} cells"
    _, ax = plt.subplots()
    ax.boxplot([old_new_slopes[:,0], old_new_slopes[:,1], old_new_slopes[:,2], old_new_slopes[:,3]], showfliers=False, medianprops=dict(color='k'))
    ax.set_title(title)
    print(title)
    print_stats([old_new_slopes[:,0], old_new_slopes[:,1], old_new_slopes[:,2], old_new_slopes[:,3]])
    ax.set_ylabel(r"elongation speed ($n m (min)^{-1}$)")
    ax.set_xticklabels(["New pole \n before NETO", "New pole \n after NETO","Old pole \n before NETO","Old pole \n after NETO"])
    pvalue = stats.ttest_ind(old_new_slopes[:,0],old_new_slopes[:,1]).pvalue
    pvalue2 = stats.ttest_ind(old_new_slopes[:,2],old_new_slopes[:,3]).pvalue
    pvalue3 = stats.ttest_ind(old_new_slopes[:,1],old_new_slopes[:,2]).pvalue
    pvalue4 = stats.ttest_ind(old_new_slopes[:,1],old_new_slopes[:,3]).pvalue
    
    x1 = 1
    x2 = 2 
    y = 11
    h = 0.2
    ax.plot([x1,  x2], [y,  y], color = 'k')
    ax.text((x1+x2)*.5, y + h , p_value_to_str(pvalue), ha='center', va='bottom')
    
    x1 = 3
    x2 = 4
    y = 19
    h = 0.2
    ax.plot([x1, x2], [y,  y],  color = 'k')
    ax.text((x1+x2)*.5, y + h , p_value_to_str(pvalue2), ha='center', va='bottom')
    
    ax.set_ylim(-5,22)
    plt.tight_layout()
    
    
    title = f"Pole elongation after NETO \n with dataset {datasetnames} and {len(old_new_slopes[:,0])} cells"
    _, ax = plt.subplots()
    ax.boxplot([after_neto_slopes[:,0], old_new_slopes[:,4]], showfliers=False, medianprops=dict(color='k'))
    ax.set_title(title)
    print(title)
    print_stats([after_neto_slopes[:,0], after_neto_slopes[:,1]])
    ax.set_ylabel(r"elongation speed ($n m (min)^{-1}$)")
    ax.set_xticklabels(["New pole \n after NETO", "Old pole "])
    pvalue = stats.ttest_ind(after_neto_slopes[:,0], after_neto_slopes[:,1]).pvalue
    
    x1 = 1
    x2 = 2 
    y = 11
    h = 0.2
    ax.plot([x1,  x2], [y,  y], color = 'k')
    ax.text((x1+x2)*.5, y + h , p_value_to_str(pvalue), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()   

def compare_INH_pole_growth(outlier_detection=False):
    datasetname1, datasetname2 = "WT_mc2_55/30-03-2015", "INH_after_700"
    
    old_new_slopes1 = [] 
    old_new_slopes2 = [] # first new, second new, old
    
    params = get_scaled_parameters(data_set=True)
    
    if datasetname1 in params.keys():
        datasets1 = params[datasetname1]
    elif isinstance(datasetname1, str): 
        raise NameError('This directory does not exist')
    else :
        datasets1 = datasetname1
    
    if datasetname2 in params.keys():
        datasets2 = params[datasetname2]
    elif isinstance(datasetname2, str): 
        raise NameError('This directory does not exist')
    else :
        datasets2 = datasetname2
    
    
    for dataset in datasets1:
        for roi_id, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)
 
                # the neto is computed from the overall length because it is more stable
                _, _, _, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)


                
                new_a_1, new_a_2, _, = compute_slopes(new_times, new_pole_growth, t)
                a_3, a_4, _ = compute_slopes(old_times, old_pole_growth, t)
                

                mat = np.zeros((len(old_times), 2))
                mat[:, 0] = old_times
                mat[:, 1] = 1
                sl, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                old_new_slopes1.append([new_a_1, new_a_2, a_3, a_4,sl])
                
                
    for dataset in datasets2:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)
 
                
                mat = np.zeros((len(old_times), 2))
                mat[:, 0] = old_times
                mat[:, 1] = 1
                slo, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                
                mat = np.zeros((len(new_times), 2))
                mat[:, 0] = new_times
                mat[:, 1] = 1
                sln, _ = np.linalg.lstsq(mat, new_pole_growth, None)[0]
                
                old_new_slopes2.append([sln, slo])
            
                        
                        
                        
                    
                
                
                  
    old_new_slopes1 = 1000*np.array(old_new_slopes1)   
    old_new_slopes2 = 1000*np.array(old_new_slopes2)
    

    
    title = f"Old pole elongation speed  with dataset \n{datasetname1} and {datasetname2} \n and {len(old_new_slopes1[:,0])} + {len(old_new_slopes2[:,0])} cells"
    _, ax = plt.subplots()
    ax.boxplot([old_new_slopes1[:,4], old_new_slopes2[:,1], old_new_slopes1[:,0], old_new_slopes1[:,1], old_new_slopes2[:,0]], showfliers=False, medianprops=dict(color='k'))
    ax.set_title(title)
    print(title)
    print_stats([old_new_slopes1[:,4], old_new_slopes2[:,1], old_new_slopes1[:,0], old_new_slopes1[:,1], old_new_slopes2[:,0]])
    ax.set_ylabel(r"elongation speed ($n m (min)^{-1}$)")
    ax.set_xticklabels(["Old pole \n non treated","Old pole\n after INH", " New pole \n pre-Neto \n non treated ", " New pole \n post-Neto \n non treated ","New pole\n after INH"])
    pvalue = stats.ttest_ind(old_new_slopes1[:,4], old_new_slopes2[:,1]).pvalue
    pvalue2 = stats.ttest_ind(old_new_slopes1[:,0], old_new_slopes2[:,0]).pvalue
    pvalue3 = stats.ttest_ind(old_new_slopes1[:,1], old_new_slopes2[:,0]).pvalue
    
    x1 = 1
    x2 = 2 
    y = 16
    h = 0.2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y + h , p_value_to_str(pvalue), ha='center', va='bottom')
    
    
    x1 = 4
    x2 = 5 
    y = 16
    h = 0.2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y + h , p_value_to_str(pvalue3), ha='center', va='bottom')
    
    
    ax.set_ylim(-3,18)
    
    plt.tight_layout()
    plt.show()   




def compare_dataset_pole_growth(datasetname1, datasetname2, y, h, lim, outlier_detection=False):
    tot_growth1 = []
    overall_slopes1 = []  
    old_new_slopes1 = [] 
    tot_growth2 = []
    overall_slopes2 = []  
    old_new_slopes2 = [] 
    
    params = get_scaled_parameters(data_set=True)
    
    if datasetname1 in params.keys():
        datasets1 = params[datasetname1]
    elif isinstance(datasetname1, str): 
        raise NameError('This directory does not exist')
    else :
        datasets1 = datasetname1
    
    if datasetname2 in params.keys():
        datasets2 = params[datasetname2]
    elif isinstance(datasetname2, str): 
        raise NameError('This directory does not exist')
    else :
        datasets2 = datasetname2
    
    
    for dataset in datasets1:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                tot_growth1.append((overall_growth[-1]-overall_growth[0])/(timestamps[-1]-timestamps[0]))
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)

                # the neto is computed from the overall length because it is more stable
                a_1, a_2, b, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
                overall_slopes1.append([a_1,a_2])
                    
                
                
                new_a_1, new_a_2, _, = compute_slopes(new_times, new_pole_growth, t)
                a_3, a_4, _ = compute_slopes(old_times, old_pole_growth, t)
                
                
                mat = np.zeros((len(old_times), 2))
                mat[:, 0] = old_times
                mat[:, 1] = 1
                sl, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                old_new_slopes1.append([new_a_1, new_a_2, a_3, a_4,sl])
                    
                    
    for dataset in datasets2:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                tot_growth2.append((overall_growth[-1]-overall_growth[0])/(timestamps[-1]-timestamps[0]))
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth =   remove_length_outliers(overall_times, overall_growth)

                # the neto is computed from the overall length because it is more stable
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
                overall_slopes2.append([a_1,a_2])

                    
                new_a_1, new_a_2, _, = compute_slopes(new_times, new_pole_growth, t)
                a_3, a_4, _ = compute_slopes(old_times, old_pole_growth, t)
                
                
                mat = np.zeros((len(old_times), 2))
                mat[:, 0] = old_times
                mat[:, 1] = 1
                sl, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                old_new_slopes2.append([new_a_1, new_a_2, a_3, a_4,sl])
            
                        
                        
                        
                    
                
                
                
    overall_slopes1 = 1000*np.array(overall_slopes1)      
    old_new_slopes1 = 1000*np.array(old_new_slopes1)
    overall_slopes2 = 1000*np.array(overall_slopes2)      
    old_new_slopes2 = 1000*np.array(old_new_slopes2)
    
    title = f"New pole elongation speed \n with dataset {datasetname1} and \n {datasetname2} \n and {len(old_new_slopes1[:,0])} + {len(old_new_slopes2[:,0])} cells"
    _, ax = plt.subplots()
    ax.boxplot([old_new_slopes1[:,0], old_new_slopes1[:,1], old_new_slopes2[:,0], old_new_slopes2[:,1]], showfliers=False, medianprops=dict(color='k'))
    ax.set_title(title)
    print(title)
    print_stats([old_new_slopes1[:,0], old_new_slopes1[:,1], old_new_slopes2[:,0], old_new_slopes2[:,1]])
    ax.set_ylabel(r"elongation speed ($n m (min)^{-1}$)")
    ax.set_xticklabels(["Before NETO, \n 1st dataset", "After NETO,\n 1st dataset","Before NETO,\n 2nd dataset", "After NETO,\n 2nd dataset"])
    pvalue = stats.ttest_ind(old_new_slopes1[:,0],old_new_slopes1[:,1]).pvalue
    pvalue2 = stats.ttest_ind(old_new_slopes2[:,0],old_new_slopes2[:,1]).pvalue
    pvalue3 = stats.ttest_ind(old_new_slopes1[:,0],old_new_slopes2[:,0]).pvalue
    pvalue4 = stats.ttest_ind(old_new_slopes1[:,1],old_new_slopes2[:,1]).pvalue
    
    x1 = 1
    x2 = 2 
    ax.plot([x1, x2], [y[0], y[0]], color = 'k')
    ax.text((x1+x2)*.5, y[0] + h , p_value_to_str(pvalue), ha='center', va='bottom')
    
    x1 = 3
    x2 = 4
    ax.plot([x1, x2], [y[1], y[1]],  color = 'k')
    ax.text((x1+x2)*.5, y[1] + h , p_value_to_str(pvalue2), ha='center', va='bottom')
    
    x1 = 1
    x2 = 3
    ax.plot([x1, x2], [y[2], y[2]],  color = 'k')
    ax.text((x1+x2)*.5, y[2] + h , p_value_to_str(pvalue3), ha='center', va='bottom')
    
    x1 = 2
    x2 = 4
    ax.plot([x1, x2], [y[3], y[3]],  color = 'k')
    ax.text((x1+x2)*.5, y[3] + h , p_value_to_str(pvalue4), ha='center', va='bottom')
    
    ax.set_ylim(lim[0],lim[1])

    plt.tight_layout()
    
    
    title = f"Old pole elongation speed \n with dataset {datasetname1} and \n {datasetname2} \n and {len(old_new_slopes1[:,0])} + {len(old_new_slopes2[:,0])} cells"
    _, ax = plt.subplots()
    ax.boxplot([old_new_slopes1[:,2], old_new_slopes1[:,3], old_new_slopes2[:,2], old_new_slopes2[:,3]], showfliers=False, medianprops=dict(color='k'))
    ax.set_title(title)
    print(title)
    print_stats([old_new_slopes1[:,2], old_new_slopes1[:,3], old_new_slopes2[:,2], old_new_slopes2[:,3]])
    ax.set_ylabel(r"elongation speed ($n m (min)^{-1}$)")
    ax.set_xticklabels(["Before NETO, \n 1st dataset", "After NETO,\n 1st dataset","Before NETO,\n 2nd dataset", "After NETO,\n 2nd dataset"])
    pvalue = stats.ttest_ind(old_new_slopes1[:,2],old_new_slopes1[:,3]).pvalue
    pvalue2 = stats.ttest_ind(old_new_slopes2[:,2],old_new_slopes2[:,3]).pvalue
    pvalue3 = stats.ttest_ind(old_new_slopes1[:,3],old_new_slopes2[:,2]).pvalue
    pvalue4 = stats.ttest_ind(old_new_slopes1[:,3],old_new_slopes2[:,3]).pvalue
    
    x1 = 1
    x2 = 2 
    ax.plot([x1, x2], [y[4], y[4]], color = 'k')
    ax.text((x1+x2)*.5, y[4] + h , p_value_to_str(pvalue) , ha='center', va='bottom')
    
    x1 = 3
    x2 = 4
    ax.plot([x1, x2], [y[5], y[5]],  color = 'k')
    ax.text((x1+x2)*.5, y[5] + h , p_value_to_str(pvalue2), ha='center', va='bottom')
    
    x1 = 1
    x2 = 3
    ax.plot([x1, x2], [y[6], y[6]],  color = 'k')
    ax.text((x1+x2)*.5, y[6] + h ,p_value_to_str(pvalue3), ha='center', va='bottom')
    
    x1 = 2
    x2 = 4
    ax.plot([x1, x2], [y[7], y[7]],  color = 'k')
    ax.text((x1+x2)*.5, y[7] + h , p_value_to_str(pvalue4), ha='center', va='bottom')
    
    
    ax.set_ylim(-5,33)
    ax.set_ylim(lim[2],lim[3])
    plt.tight_layout()
    plt.show()   


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 13})
    plot_growth_all_cent('good', outlier_detection=True) 
    compute_pole_growth_stats("good", outlier_detection=False)
    compute_pole_growth_stats("WT_no_drug", outlier_detection=False)
    compare_INH_pole_growth(outlier_detection=False)
    compare_dataset_pole_growth("WT_mc2_55/30-03-2015", "INH_after_700", [12,8,15,17,19,16,21.5,24], 0.4, [-6,19, -13,27], outlier_detection=False)
    
    plt.rcParams.update({'font.size': 10}) 