import glob
import os
import sys
from collections import Counter
import statistics

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import splev, splrep


package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.group_by_cell import Orientation, load_dataset, get_peak_troughs_lineage_lists
from peaks_troughs.growth_stats import p_value_to_str, print_stats, piecewise_pointwise_linear_regression, extract_growth
from scaled_parameters import get_scaled_parameters

'''Computing different geopetrical feature statistics'''
#%% Statistics on peaks and troughs without lineage
def compute_stats(dataset):
    peak_counter = Counter()
    trough_counter = Counter()
    peak_lengths = []
    trough_lengths = []
    for _, cell in load_dataset(dataset):
        for frame_data in cell:
            xs = frame_data["xs"]
            ys = frame_data["ys"]
            
            peaks = frame_data['peaks']
            troughs = frame_data['troughs']
            peak_counter[len(peaks)] += 1
            trough_counter[len(troughs)] += 1
            if len(peaks) + len(troughs) >= 2:
                peaks_dist, troughs_dist = compute_dist(peaks, troughs, xs)
                peak_lengths.extend(peaks_dist)
                trough_lengths.extend(troughs_dist)
    stats = {
        "peak_counter": peak_counter,
        "trough_counter": trough_counter,
        "peak_lengths": peak_lengths,
        "trough_lengths": trough_lengths,
    }
    return stats


def compute_dist(peaks, troughs, xys, array=False): #can use xs or ys for lenght or height
    peaks_dist = []
    troughs_dist = []
    for peak in peaks:
        score = 0
        right = False
        left = False
        if np.any(troughs > peak):
            pos = np.min(troughs[troughs > peak])
            score += np.abs(xys[pos] - xys[peak])
            right = True
        if np.any(troughs < peak):
            pos = np.max(troughs[troughs < peak])
            score += np.abs(xys[pos] - xys[peak])
            right = True
        if right and left:
            peaks_dist.append(score/2)
        else:
            peaks_dist.append(score)
    for trough in troughs:
        score = 0
        right = False
        left = False
        if np.any(peaks > trough):
            pos = np.min(peaks[peaks > trough])
            score += np.abs(xys[pos] - xys[trough])
            right = True
        if np.any(peaks < trough):
            pos = np.max(peaks[peaks < trough])
            score += np.abs(xys[pos] - xys[trough])
            right = True
        if right and left:
            troughs_dist.append(score/2)
        else:
            troughs_dist.append(score)
    if array:
        return np.array(peaks_dist), np.array(troughs_dist)
    return peaks_dist, troughs_dist 
        
        
        
    

def _plot_counts_histogram(ax, counter, feature, dataset):
    n_centerlines = sum(counter.values())
    mini = min(counter)
    maxi = max(counter)
    percentages = [100 * counter[x] / n_centerlines for x in range(mini, maxi + 1)]
    edges = [x - 0.5 for x in range(mini, maxi + 2)]
    xlabel = f"Number of {feature}"
    ylabel = "Proportion of centerlines (%)"
    title = "Repartition of the number of {} ({} | {} centerlines)".format(
        feature, dataset, n_centerlines
    )
    ax.stairs(percentages, edges, fill=True)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def plot_peak_trough_counters(dataset, peak_counter, trough_counter):
    _, (ax1, ax2) = plt.subplots(1, 2)
    _plot_counts_histogram(ax1, peak_counter, "peaks", dataset)
    _plot_counts_histogram(ax2, trough_counter, "troughs", dataset)


def _plot_lengths_histogram(ax, lengths, feature, dataset):
    mean = np.mean(lengths)
    q1, med, q3 = np.percentile(lengths, [25, 50, 75])
    xlabel = f"Length of the {feature}s (µm)"
    title = (
        f"Distribution of the {feature} length ({dataset} | "
        f"{len(lengths)} {feature}s)"
    )
    density, bins = np.histogram(lengths , 40)
    spl = splrep(bins[1:], density, s=3*len(lengths), per=False)
    x2 = np.linspace(bins[0], bins[-1], 200)
    y2 = splev(x2, spl)
    y2[y2<0]=0
    ax.plot(x2, y2, '--', color = 'purple', label= 'smooth approximation', )
    ax.hist(lengths, 40,  color="grey") #density=True,
    ax.axvline(mean, color="red", label="mean")
    ax.axvline(med, color="blue", label="median")
    ax.axvline(q1, color="green", label="quantiles")
    ax.axvline(q3, color="green")
    ax.legend()
    ax.set(xlabel=xlabel, title=title)


def plot_peak_trough_lengths(dataset, peak_lengths, trough_lengths):
    _, (ax1, ax2) = plt.subplots(2, 1)
    _plot_lengths_histogram(ax1, peak_lengths, "peak", dataset)
    _plot_lengths_histogram(ax2, trough_lengths, "trough", dataset)


def plot_stats(dataset, /, peak_counter, trough_counter, peak_lengths, trough_lengths):
    plot_peak_trough_counters(dataset, peak_counter, trough_counter)
    plot_peak_trough_lengths(dataset, peak_lengths, trough_lengths)
    plt.tight_layout()
    plt.show()



#%% Statistics on peaks and troughs with lineage

# Spatial distribution of feature creation 
def feature_creation(dataset_names):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    bin_num = params["bin_number_hist_feat_crea"]
    smooth_param = params["smoothing_hist_feat_crea"]
    
    stat_list = {'peaks':[], 'troughs':[]}
    stat_list_relat = {'peaks':[], 'troughs':[]}
    length_list = []
    stat_list_np = {'peaks':[], 'troughs':[]}
    stat_list_op = {'peaks':[], 'troughs':[]}
    
    
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        frame_data = cell[0] #pnt_ROI[key][1]
                        orientation = Orientation(frame_data['orientation'])
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            if generation >2 :
                                total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                                length_list.append(total_length)
                                x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])
                                feat_ind = int(pnt_list[root][2])
                                feature = feat_ind*'peaks' + (1-feat_ind)*'troughs'
                                match orientation:
                                    case Orientation.UNKNOWN:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            
                                    case Orientation.NEW_POLE_OLD_POLE:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            stat_list_op[feature].append(x_coord)
                                        else:
                                            stat_list_np[feature].append(x_coord)
                                            
                                    case Orientation.OLD_POLE_NEW_POLE:
                                        if x_coord > total_length/2 :
                                            x_coord = total_length - x_coord
                                            stat_list_np[feature].append(x_coord)
                                        else:
                                            stat_list_op[feature].append(x_coord)
                                stat_list[feature].append(x_coord)
 
                                stat_list_relat[feature].append(x_coord/total_length)
                                
                                
                                
    
    
    
    
    _, ax = plt.subplots()
    title = (
        f"Distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list['peaks']+stat_list['troughs'])} features tracked"
    )
    ax.boxplot([stat_list['peaks']+stat_list['troughs'], np.array(length_list)/2],  showfliers=False, medianprops=dict(color='k'))
    print(title)
    print_stats([stat_list['peaks']+stat_list['troughs'], np.array(length_list)/2])
    ax.set_xticklabels([f'Feature creation distance \n to pole', 'half cell length'])
    ax.set_ylabel(r"($\mu m$)")
    pvalue1 = stats.ttest_ind(stat_list['peaks']+stat_list['troughs'], np.array(length_list)/2).pvalue
    x1 = 1
    x2 = 2 
    y = 4
    h=0.01
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    
    ax.set_title(title)
    
    
    # plt.figure()
    # title = (
    #     f"Distribution of feature creation with dataset \'{dataset_names}\',\n and {len(stat_list['peaks']+stat_list['troughs'])} features tracked"
    # )
    # res = stats.kstest(stat_list['peaks'], stat_list['troughs'])
    # print(res.statistic, res.pvalue)
    # plt.hist(stat_list['peaks'], bin_num, color="g", alpha=0.6,  label= 'peaks', density=True)
    # plt.hist(stat_list['troughs'], bin_num, color="b", alpha=0.6, label= 'troughs', density=True)
    # plt.annotate(f'pvalue = {res.pvalue:.2e}', (3,0.6))
    # plt.xlabel(r'Distance to the pole ($\mu m$) ')
    # plt.legend()
    # plt.title(title)

    
    plt.figure()
    title = (
        f"Feature creation with dataset \'{dataset_names}\',\n and {len(stat_list_relat['peaks'])+len(stat_list_relat['troughs'])} features tracked, normalized total lenght"
    )
    plt.hist(stat_list_relat['peaks'], bin_num, color="grey", alpha=0.6, label= 'feature creation peaks')
    plt.hist(stat_list_relat['troughs'], bin_num, color="red", alpha=0.6, label= 'feature creation troughs')
    plt.xlabel(r' $\leftarrow \; \text{pole}\;|\; \text{center}\; \rightarrow$ ')
    print(title)
    print_stats([stat_list_relat['peaks'],stat_list_relat['troughs']])
    plt.legend()
    plt.title(title)
    
    
    stat_list_relat = stat_list_relat['peaks'] + stat_list_relat['troughs']
    plt.figure()
    title = (
        f"Feature creation with dataset \'{dataset_names}\',\n and {len(stat_list_relat)} features tracked, normalized total lenght"
    )
    plt.hist(stat_list_relat, bin_num, color="grey", alpha=0.6, label= 'feature creation peaks')
    plt.hist(stat_list_relat, bin_num, color="red", alpha=0.6, label= 'feature creation troughs')
    plt.xlabel(r' $\leftarrow \; \text{pole}\;|\; \text{center}\; \rightarrow$ ')
    print(title)
    print_stats([stat_list_relat])
    density, bins = np.histogram(stat_list_relat , bin_num)
    spl = splrep(bins[1:], density,s=3*len(stat_list_relat), per=False)
    x2 = np.linspace(bins[0], bins[-1], 100)
    y2 = splev(x2, spl)
    y2[y2<0]=0
    plt.plot(x2, y2, '--', color = 'k', label= 'smooth approximation', )
    
    plt.legend()
    plt.title(title)

    
    # _, ax = plt.subplots()
    # ax.boxplot([stat_list_np['peaks'],
    #             stat_list_op['peaks'],
    #             stat_list_np['troughs'],
    #             stat_list_op['troughs']]
    #            ,  showfliers=False, medianprops=dict(color='k')) 
    # print(title)
    # print_stats([stat_list_np['peaks'],
    #             stat_list_op['peaks'],
    #             stat_list_np['troughs'],
    #             stat_list_op['troughs']])
    # ax.set_xticklabels([f"Peaks new pole \n creation", f"Peaks old pole \n creation", f"Troughs new pole \n creation", f"Troughs old pole \n creation"])
    # ax.set_ylabel(r"Distance to the pole ($\mu m$)")
    # pvalue1 = stats.ttest_ind(stat_list_np['peaks'], stat_list_op['peaks']).pvalue
    # x1 = 1
    # x2 = 2 
    # y = 2.8
    # h=0.01
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    # pvalue1 = stats.ttest_ind(stat_list_np['peaks'], stat_list_np['troughs']).pvalue
    # x1 = 1
    # x2 = 3 
    # y = 3.1
    # h=0.01
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    # pvalue1 = stats.ttest_ind(stat_list_op['peaks'], stat_list_op['troughs']).pvalue
    # x1 = 2
    # x2 = 4
    # y = 2.5
    # h=0.01
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    # pvalue1 = stats.ttest_ind(stat_list_np['troughs'], stat_list_op['troughs']).pvalue
    # x1 = 3
    # x2 = 4 
    # y = 2.1
    # h=0.01
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    # ax.set_ylim(0,3.5)
    
    plt.show()




def feature_creation_time_vs_NETO(dataset_names):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    
    stat_list_np = []
    stat_list_op = []
    
    
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                
                timestamps, _, new_pole_growth, overall_growth = extract_growth(cell)
                if len(timestamps)>=6 and timestamps[-1]-timestamps[0] > 100:
                    timestamps = timestamps - timestamps[0]
                    
                    # a_1, a_2, _, NETO = piecewise_pointwise_linear_regression(timestamps, overall_growth)
                    
                    # if 0<a_1<=a_2:
                    a_1, a_2, _, NETO = piecewise_pointwise_linear_regression(timestamps, new_pole_growth)
                    if 0<a_1<=a_2:
                        
                                
                            for key in pnt_ROI:
                                if len(pnt_ROI[key]) >= 2:
                                    root = pnt_ROI[key][0]
                                    frame_data = cell[0]
                                    orientation = Orientation(frame_data['orientation'])
                                    time = pnt_list[root][-2]
                                    if time >0 :
                                        generation = int(pnt_list[root][1])
                                        if generation >2 and  cell[generation]['timestamp']>700:
                                            crea_time = cell[generation]['timestamp'] - frame_data['timestamp']
                                            total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                                            x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])
                                            match orientation:
                                                case Orientation.UNKNOWN:
                                                    continue
                                                        
                                                case Orientation.NEW_POLE_OLD_POLE:
                                                    if x_coord > total_length-1.5 :
                                                        x_coord = total_length - x_coord
                                                        stat_list_op.append([x_coord, crea_time, NETO])
                                                        print('op',[roi_id, crea_time, NETO])
                                                    elif x_coord < 1.5:
                                                        stat_list_np.append([x_coord, crea_time, NETO])
                                                        print('np',[roi_id, crea_time, NETO])
                                                        
                                                case Orientation.OLD_POLE_NEW_POLE:
                                                    if x_coord > total_length-1.5:
                                                        x_coord = total_length - x_coord
                                                        stat_list_np.append([x_coord, crea_time, NETO])
                                                        print('np',[roi_id, crea_time, NETO])
                                                    elif x_coord < 1.5:
                                                        stat_list_op.append([x_coord, crea_time, NETO])
                                                        print('op',[roi_id, crea_time, NETO])
                                            
    stat_list_op = np.array(stat_list_op)
    stat_list_np = np.array(stat_list_np)
    
    mask_op = stat_list_op[:,1]<stat_list_op[:,2]
    mask_np = stat_list_np[:,1]<stat_list_np[:,2]

    print( f'Feature numbers : {len(stat_list_op)+ len(stat_list_np)},  Creation : \n' 
          f'OP before Neto {len(stat_list_op[mask_op])}, '
          f'NP before Neto {len(stat_list_np[mask_np])}, '
          f'NP after Neto {len(stat_list_np[np.logical_not(mask_np)])}, ' 
          f'OP after Neto {len(stat_list_op[np.logical_not(mask_op)])}')
    
    # title =f"Creation pole vs NETO\'{dataset_names}\',\n and {len(stat_list_op)+len(stat_list_np)} features"
    # _, ax = plt.subplots()
    # ax.boxplot([stat_list_op[mask_op], stat_list_op[np.logical_not(mask_op)], stat_list_np[np.logical_not(mask_np)]]
    #            ,  showfliers=False, medianprops=dict(color='k')) 
    # print(title)
    # print_stats([stat_list_op[mask_op], stat_list_op[np.logical_not(mask_op)], stat_list_np[np.logical_not(mask_np)]])
    # ax.set_xticklabels(["Peaks new pole", "Peaks old pole", "Troughs new pole"])
    # ax.set_ylabel(r"Distance ($\mu m$)")
    # ax.set_title(title)
    # pvalue1 = stats.ttest_ind(whole_stat['lenght']['np']['peaks'], whole_stat['lenght']['op']['peaks']).pvalue
    # x1 = 1
    # x2 = 2 
    # y = 2.6
    # h=0.02
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    # pvalue2 = stats.ttest_ind(whole_stat['lenght']['np']['troughs'], whole_stat['lenght']['op']['troughs']).pvalue
    # x1 = 3
    # x2 = 4 
    # y =  2.8
    # h=0.02
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    # pvalue1 = stats.ttest_ind(whole_stat['lenght']['np']['peaks'], whole_stat['lenght']['np']['troughs']).pvalue
    # x1 = 1
    # x2 = 3 
    # y = 3.1
    # h=0.02
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    # pvalue2 = stats.ttest_ind(whole_stat['lenght']['op']['troughs'], whole_stat['lenght']['op']['peaks']).pvalue
    # x1 = 2
    # x2 = 4 
    # y = 3.4
    # h=0.02
    # ax.plot([x1, x2], [y, y], color = 'k')
    # ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    # ax.set_ylim(0,3.6)
                                        
                                
                                

#comparision of old pole / new pole distributions
def feature_properties_pole_feature(dataset_names):
    params = get_scaled_parameters(data_set=True, stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    pole_size = params["pole_region_size"] 
    whole_stat = {}
    for prop in ['lenght', 'height']:
        whole_stat[prop] = {}
        for pole in ['np', 'op']:
            whole_stat[prop][pole]={}
            for feat in ['peaks', 'troughs']:
                whole_stat[prop][pole][feat] = []
    
    
    for dataset in datasets:
        
        for _, cell in load_dataset(dataset, False):
            stat = {}
            for prop in ['lenght', 'height']:
                stat[prop] = {}
                for pole in ['np', 'op']:
                    stat[prop][pole]={}
                    for feat in ['peaks', 'troughs']:
                        stat[prop][pole][feat] = []
            if len(cell) > 1:
                orientation = Orientation(cell[0]['orientation'])
                if orientation == Orientation.UNKNOWN:
                    continue
                for frame_data in cell:
                    if len(frame_data['peaks']) >=1 and len(frame_data['troughs'])>=1:
                        height = {}
                        length = {}
                        height['peaks'], height['troughs'] = compute_dist(frame_data['peaks'], frame_data['troughs'], frame_data['ys'],array=True)
                        length['peaks'], length['troughs']= compute_dist(frame_data['peaks'], frame_data['troughs'], frame_data['xs'],array=True)
                        for feature in ['peaks', 'troughs']: 
                            mask = frame_data['xs'][frame_data[feature]]- frame_data['xs'][0] <= pole_size
                            match orientation:
                                case Orientation.NEW_POLE_OLD_POLE:
                                    pole = 'np'
                                case Orientation.OLD_POLE_NEW_POLE:
                                    pole = 'op'
                            if np.any(mask):
                                        stat['lenght'][pole][feature].extend(list(length[feature][mask])) 
                                        stat['height'][pole][feature].extend(list(height[feature][mask]))
                            mask = frame_data['xs'][frame_data[feature]] >= frame_data['xs'][-1]-pole_size
                            match orientation:
                                case Orientation.NEW_POLE_OLD_POLE:
                                    pole = 'op'
                                case Orientation.OLD_POLE_NEW_POLE:
                                    pole = 'np'
                            if np.any(mask):
                                        stat['lenght'][pole][feature].extend(list(length[feature][mask])) 
                                        stat['height'][pole][feature].extend(list(height[feature][mask]))
            for prop in ['lenght', 'height']:
                    for pole in ['np', 'op']:
                        for feat in ['peaks', 'troughs']:
                            l = stat[prop][pole][feat]
                            if len(l)>0:
                                whole_stat[prop][pole][feat].append(np.average(np.array(l)))

                    
    numbers = [len(whole_stat['height'][pole][feat])for pole in ['np', 'op']for feat in ['peaks', 'troughs']]                        
    title =f"Inter-feature distance with dataset \'{dataset_names}\',\n and {numbers} features"
    
    _, ax = plt.subplots()
    ax.boxplot([whole_stat['lenght']['np']['peaks'],
                whole_stat['lenght']['op']['peaks'],
                whole_stat['lenght']['np']['troughs'],
                whole_stat['lenght']['op']['troughs']]
               ,  showfliers=False, medianprops=dict(color='k')) 
    print(title)
    print_stats([whole_stat['lenght']['np']['peaks'],
                whole_stat['lenght']['op']['peaks'],
                whole_stat['lenght']['np']['troughs'],
                whole_stat['lenght']['op']['troughs']])
    ax.set_xticklabels(["Peaks new pole", "Peaks old pole", "Troughs new pole", "Troughs old pole"])
    ax.set_ylabel(r"Distance ($\mu m$)")
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(whole_stat['lenght']['np']['peaks'], whole_stat['lenght']['op']['peaks']).pvalue
    x1 = 1
    x2 = 2 
    y = 2.6
    h=0.02
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['lenght']['np']['troughs'], whole_stat['lenght']['op']['troughs']).pvalue
    x1 = 3
    x2 = 4 
    y =  2.8
    h=0.02
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(whole_stat['lenght']['np']['peaks'], whole_stat['lenght']['np']['troughs']).pvalue
    x1 = 1
    x2 = 3 
    y = 3.1
    h=0.02
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['lenght']['op']['troughs'], whole_stat['lenght']['op']['peaks']).pvalue
    x1 = 2
    x2 = 4 
    y = 3.4
    h=0.02
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    ax.set_ylim(0,3.6)
    
    
    title = f"Inter-feature amplitude with dataset \'{dataset_names}\',\n and {numbers} features"
    _, ax = plt.subplots()
    ax.boxplot([whole_stat['height']['np']['peaks'],
                whole_stat['height']['op']['peaks'],
                whole_stat['height']['np']['troughs'],
                whole_stat['height']['op']['troughs']]
               ,  showfliers=False, medianprops=dict(color='k')) 
    print(title)
    print_stats([whole_stat['height']['np']['peaks'],
                whole_stat['height']['op']['peaks'],
                whole_stat['height']['np']['troughs'],
                whole_stat['height']['op']['troughs']])
    ax.set_xticklabels(["Peaks new pole", "Peaks old pole", "Troughs new pole", "Troughs old pole"])
    ax.set_ylabel(r"Amplitude ($n m$)")
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(whole_stat['height']['np']['peaks'], whole_stat['height']['op']['peaks']).pvalue
    x1 = 1
    x2 = 2 
    y = 300
    h = 2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['height']['np']['troughs'], whole_stat['height']['op']['troughs']).pvalue
    x1 = 3
    x2 = 4 
    y = 410
    h = 2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    pvalue1 = stats.ttest_ind(whole_stat['height']['np']['peaks'], whole_stat['height']['np']['troughs']).pvalue
    x1 = 1
    x2 = 3 
    y = 350
    h=2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    pvalue2 = stats.ttest_ind(whole_stat['height']['op']['troughs'], whole_stat['height']['op']['peaks']).pvalue
    x1 = 2
    x2 = 4 
    y = 450
    h=2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    ax.set_ylim(0,490)
    plt.show()
                             

def feature_properties_pole(dataset_names):
    params = get_scaled_parameters(data_set=True, stats=True)
    if isinstance(dataset_names, str):
        if dataset_names in params.keys():
            datasets = params[dataset_names]
        else: 
            raise NameError('This directory does not exist')
    else :
        datasets = dataset_names 
    pole_size = params["pole_region_size"] 
    whole_stat = {}
    for prop in ['lenght', 'height']:
        whole_stat[prop] = {}
        for pole in ['np', 'op']:
            whole_stat[prop][pole]=[]
    
    
    for dataset in datasets:
        
        for _, cell in load_dataset(dataset, False):
            stat = {}
            for prop in ['lenght', 'height']:
                stat[prop] = {}
                for pole in ['np', 'op']:
                    stat[prop][pole] = []
            if len(cell) > 1:
                orientation = Orientation(cell[0]['orientation'])
                if orientation == Orientation.UNKNOWN:
                    continue
                for frame_data in cell:
                    if len(frame_data['peaks']) >=1 and len(frame_data['troughs'])>=1:
                        feature = np.zeros(len(frame_data['peaks']) + len(frame_data['troughs']), dtype=int)
                        if frame_data['peaks'][0] < frame_data['troughs'][0]:
                            feature[::2] = frame_data['peaks']
                            feature[1::2] = frame_data['troughs']
                        else :
                            feature[1::2] = frame_data['peaks']
                            feature[::2] = frame_data['troughs']
                        height = np.absolute (frame_data['ys'][feature][:-1] - frame_data['ys'][feature][1:])
                        length = np.absolute (frame_data['xs'][feature][:-1] - frame_data['xs'][feature][1:])
                        
                        mask = frame_data['xs'][feature] - frame_data['xs'][0] <= pole_size
                        mask = mask [:-1]
                        match orientation:
                            case Orientation.NEW_POLE_OLD_POLE:
                                pole = 'np'
                            case Orientation.OLD_POLE_NEW_POLE:
                                pole = 'op'
                        if np.any(mask):
                                    stat['lenght'][pole].extend(list(length[mask])) 
                                    stat['height'][pole].extend(list(height[mask]))
                        
                        mask = frame_data['xs'][-1] - frame_data['xs'][feature] <= pole_size
                        mask = mask [1:]
                        match orientation:
                            case Orientation.NEW_POLE_OLD_POLE:
                                pole = 'np'
                            case Orientation.OLD_POLE_NEW_POLE:
                                pole = 'op'
                        if np.any(mask):
                                    stat['lenght'][pole].extend(list(length[mask])) 
                                    stat['height'][pole].extend(list(height[mask]))
                                    
                                    
                        
            for prop in ['lenght', 'height']:
                    for pole in ['np', 'op']:
                            l = stat[prop][pole]
                            if len(l)>0:
                                whole_stat[prop][pole].append(np.average(np.array(l)))

    
    
    plt.show()            
    numbers = [len(whole_stat['height'][pole])for pole in ['np', 'op']]                        
    title =f"Inter-feature distance, amplitude \n with dataset \'{dataset_names}\',\n and {numbers} features"
    
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title)
    ax[0].boxplot([whole_stat['lenght']['np'],
                whole_stat['lenght']['op'],]
               ,  showfliers=False, medianprops=dict(color='k'), widths = 0.5) 
    print(title)
    print_stats([whole_stat['lenght']['np'],
                whole_stat['lenght']['op'], 
                whole_stat['height']['np'],
                whole_stat['height']['op']])
    ax[0].set_xticklabels(["New pole", "Old pole"])
    ax[0].set_ylabel(r"Distance ($\mu m$)")
    pvalue1 = stats.ttest_ind(whole_stat['lenght']['np'], whole_stat['lenght']['op']).pvalue
    x1 = 1
    x2 = 2 
    y = 1.45 # 1.05
    h=0.02
    ax[0].plot([x1, x2], [y, y], color = 'k')
    ax[0].text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    ax[0].set_ylim(0,2.2)
    # ax[0].set_ylim(0,1.25)
    
    ax[1].boxplot([whole_stat['height']['np'],
                whole_stat['height']['op']]
               ,  showfliers=False, medianprops=dict(color='k'), widths = 0.5) 
    ax[1].set_xticklabels(["New pole", "Old pole"])
    ax[1].set_ylabel(r"Amplitude ($n m$)")
    pvalue1 = stats.ttest_ind(whole_stat['height']['np'], whole_stat['height']['op']).pvalue
    x1 = 1
    x2 = 2 
    y = 140
    h = 2
    ax[1].plot([x1, x2], [y, y], color = 'k')
    ax[1].text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    ax[1].set_ylim(0,220)
    # ax[1].set_ylim(0,160)
    plt.tight_layout()
    plt.show()
       

def feature_number(dataset_names):
    params = get_scaled_parameters(data_set=True, paths_and_names=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    
    peak_root = []
    peak_leaf = []
    peak_var = []
    trough_root = []
    trough_leaf = []
    trough_var = []
    count = 0
    
    for dataset in datasets:
        data_direc = params["main_data_direc"]
        roi_dic_name = params["roi_dict_name"]
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()
        
        for roi_id, cell in load_dataset(dataset, False):
            
            if len(cell) > 5:
                if roi_dic[roi_id]['Parent'] != '' and len(roi_dic[roi_id]['Children'])>0:
                    peak_root.append(len(cell[0]['peaks']))
                    trough_root.append(len(cell[0]['troughs']))
                    peak_leaf.append(len(cell[-1]['peaks']))
                    trough_leaf.append(len(cell[-1]['troughs']))
                    peak_var.append(len(cell[-1]['peaks'])-len(cell[0]['peaks']))
                    trough_var.append(len(cell[-1]['troughs'])-len(cell[0]['troughs']))
                    count += 1
                elif  roi_dic[roi_id]['Parent'] != '':
                    peak_root.append(len(cell[0]['peaks']))
                    trough_root.append(len(cell[0]['troughs']))
                    count += 1
                    
                elif len(roi_dic[roi_id]['Children'])>0 :
                    peak_leaf.append(len(cell[-1]['peaks']))
                    trough_leaf.append(len(cell[-1]['troughs']))
                    count += 1
                    
                
    _, ax = plt.subplots()
    print(trough_leaf, trough_var)
    title = f"Number of features with dataset \'{dataset_names}\',\n and {count} cells"
    ax.boxplot([peak_root, peak_leaf, peak_var, trough_root, trough_leaf, trough_var], showfliers=False, medianprops=dict(linewidth= 1.5,color='k'))
    print(title)
    print_stats([peak_root, peak_leaf, peak_var, trough_root, trough_leaf, trough_var])
    ax.set_xticklabels([f"Peaks  after \n division", f"Peaks \n before division", f"Peaks \n variation", f"Troughs \n after division", f"Troughs  before \n division", f"Troughs \n variation"])
    ax.set_ylabel("Feature number")
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(peak_root, peak_leaf).pvalue
    pvalue2 = stats.ttest_ind( trough_root, trough_leaf).pvalue
    x1 = 1
    x2 = 2 
    y = 6.2
    h=0.2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    
    x1 = 4
    x2 = 5 
    y = 6.2
    h=0.2
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue2), ha='center', va='bottom')
    ax.set_ylim(-0.5,7)
    plt.tight_layout()
    plt.show()
    


def feature_general_properties(dataset_names, plot=True):
    
    params = get_scaled_parameters(data_set=True,stats=True)
    if isinstance(dataset_names, str):
        if dataset_names in params.keys():
            datasets = params[dataset_names]
        else: 
            raise NameError('This directory does not exist')
    else :
        datasets = dataset_names 
    
    stat_list = []
    position_list = []
    diff_list = []
    dist_list = {'peaks':[], 'troughs':[]}
    amp_list = {'peaks':[], 'troughs':[]}
    
    for dataset in datasets:
        for _, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                for frame_data in cell:
                    ys = frame_data["ys"]
                    xs = frame_data["xs"]-frame_data["xs"][0]
                    peaks = frame_data["peaks"]
                    troughs = frame_data["troughs"]
                    if len(peaks)>=2:
                        dist_list['peaks'].append(xs[peaks][1:]-xs[peaks][:-1])
                    if len(troughs)>=2:
                        dist_list['troughs'].append(xs[troughs][1:]-xs[troughs][:-1])
                    
                    totlen = len(peaks)+len(troughs)
                    if totlen>=2:
                        feature = np.zeros(totlen)
                        x_coord = np.zeros(totlen)
                        if peaks[0]<troughs[0]:
                            feature[::2] = ys[peaks]
                            feature[1::2] = ys[troughs]
                            x_coord[::2] = xs[peaks]
                            x_coord[1::2] = xs[troughs]
                            
                            D = np.absolute(feature[1:] - feature[:-1])
                            if len(D) >= 2:
                                D = [np.absolute(feature[1]-feature[0])] + list(0.5* D[1:] + 0.5* D[:-1]) + [np.absolute(feature[-1]-feature[-2])]
                            else: 
                                D = 2*list(D)
                            D = np.array(D)
                            amp_list['peaks'].append(D[::2])
                            amp_list['troughs'].append(D[1::2])
                            
                        else:
                            feature[1::2] = ys[peaks]
                            feature[::2] = ys[troughs]
                            x_coord[1::2] = xs[peaks]
                            x_coord[::2] = xs[troughs]
                            
                            D = np.absolute(feature[1:] - feature[:-1])
                            if len(D) >= 2:
                                D = [np.absolute(feature[1]-feature[0])] + list(0.5* D[1:] + 0.5* D[:-1]) + [np.absolute(feature[-1]-feature[-2])]
                            else: 
                                D = 2*list(D)
                            D = np.array(D)
                            amp_list['peaks'].append(D[1::2])
                            amp_list['troughs'].append(D[::2])
                            
                        stat_list.append(np.absolute(feature[:-1]-feature[1:]))
                        diff_list.append(x_coord[1:]-x_coord[:-1])
                        for elem in x_coord[:-1]:
                            if elem < xs[-1]/2:
                                position_list.append(elem/xs[-1])
                            else :
                                position_list.append((xs[-1]-elem)/xs[-1])

    
    stat_list = np.concatenate(stat_list)
    position_list = np.array(position_list)
    diff_list = np.concatenate(diff_list)
         
    good_ind = (stat_list<500)#& (diff_list<4) & (position_list<10)
    stat_list = stat_list[good_ind]
    position_list = position_list[good_ind]
    diff_list = diff_list[good_ind]
    
    if plot:
        fig, ax = plt.subplots(1,2)
        title = (
            f" Feature properties dataset \'{dataset_names}\',\n and {len(stat_list)} individual features"
        )
        fig.suptitle(title)
        ax[0].boxplot(stat_list, showfliers=False, showmeans=True, meanline=True, medianprops=dict(color='k'),meanprops=dict(color='k')) 
        print(title)
        print_stats([stat_list])
        ax[0].set_xticklabels(['Inter-feature amplitude'])
        ax[0].set_ylabel(r'($n m$) ')

        
        ax[1].boxplot(diff_list, showfliers=False, showmeans=True, meanline=True, medianprops=dict(color='k'),meanprops=dict(color='k')) 
        print(title)
        print_stats([diff_list])
        ax[1].set_xticklabels(['Inter-feature distance'])
        ax[1].set_ylabel(r'($\mu m$) ')
        
        plt.tight_layout()
        
        dist_list['peaks'] = np.concatenate(dist_list['peaks'] )
        dist_list['troughs'] = np.concatenate(dist_list['troughs'])
        amp_list['peaks'] = np.concatenate(amp_list['peaks'])
        amp_list['troughs'] = np.concatenate(amp_list['troughs'])
        
        _, ax = plt.subplots()
        title = (
            f"Feature distance with \'{dataset_names}\',\n and {[len(dist_list['peaks']), len(dist_list['troughs'])]} features tracked"
        )
        ax.boxplot([dist_list['peaks'], dist_list['troughs']],  showfliers=False, medianprops=dict(color='k'))
        print(title)
        print_stats([dist_list['peaks'], dist_list['troughs']])
        ax.set_xticklabels(['Peaks distance', 'Troughs distance'])
        ax.set_ylabel(r"($\mu m$)")
        ax.set_title(title)
        pvalue1 = stats.ttest_ind(dist_list['peaks'], dist_list['troughs']).pvalue
        x1 = 1
        x2 = 2 
        y = 2.5
        h = 0.05
        ax.plot([x1, x2], [y, y], color = 'k')
        ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
        plt.tight_layout()
        
        _, ax = plt.subplots()
        title = (
            f"Feature amplitude with \'{dataset_names}\',\n and {[len(amp_list['peaks']), len(amp_list['troughs'])]} features tracked"
        )
        ax.boxplot([ amp_list['peaks'], amp_list['troughs']],  showfliers=False, medianprops=dict(color='k'))
        print(title)
        print_stats([ amp_list['peaks'], amp_list['troughs']])
        ax.set_xticklabels([ 'Peaks amplitude', 'Troughs amplitude'])
        ax.set_ylabel(r"($n m$)")
        ax.set_title(title)
        pvalue1 = stats.ttest_ind(amp_list['peaks'], amp_list['troughs']).pvalue
        x1 = 1
        x2 = 2 
        y = 120
        h = 1
        ax.plot([x1, x2], [y, y], color = 'k')
        ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
        plt.tight_layout()
        
        
    

        # plt.figure()
        # title = (
        #     f"Feature height with dataset \'{dataset_names}\',\n and {len(stat_list)} individual features, normalized total lenght"
        # )
        # # plt.scatter(position_list, stat_list, marker='.')
        # bins = 10
        # data_list = []
        # for elem in range(bins):
        #     data_list.append([])
        # for i, elem in enumerate(position_list):
        #     index = int(elem *2*bins)
        #     if index == 10:
        #         index = 9
        #     data_list[index].append(stat_list[i])

        # plt.boxplot(data_list,labels= np.linspace(0.5,5,10)/10, showfliers=False)       #, showmeans=True, meanline=True
        # # slope, intercept = statistics.linear_regression(position_list, stat_list)
        # # min_x= np.min(position_list)
        # # max_x= np.max(position_list)
        # # plt.plot([min_x,max_x], [slope*min_x+intercept, slope*max_x+intercept], color='r', label='linear interpolation')
        # plt.xlabel(r' $\leftarrow \;\text{pole}\;|\;  \text{center} \;\rightarrow$')
        # plt.ylabel(r'Height ($n m$)')
        # # plt.legend()
        # plt.title(title)
        

        plt.show()
    
    return stat_list, position_list, diff_list
    
                    
                    

def feature_creation_comparison(dataset_names1, dataset_names2):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names1 in params.keys():
        datasets1 = params[dataset_names1]
    else : 
        datasets1 = dataset_names1
    
    if dataset_names2 in params.keys():
        datasets2 = params[dataset_names2]
    else : 
        datasets2 = dataset_names2
    
    bin_num = params["bin_number_hist_feat_crea"]
    
    
    stat_list1 = []
    for dataset in datasets1:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                            x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])/total_length
                            if generation >2 :
                                if 0 <= x_coord <= 0.5:
                                    stat_list1.append(x_coord)
                                else :
                                    stat_list1.append(1-x_coord)
    
    stat_list1 = np.array(stat_list1)
    
    stat_list2 = []
    for dataset in datasets2:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 4:
                        root = pnt_ROI[key][0]
                        time = pnt_list[root][-2]
                        if time >0 :
                            generation = int(pnt_list[root][1])
                            total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                            x_coord = abs(pnt_list[root][3]-cell[generation]['xs'][0])/total_length
                            if generation >2 :
                                if 0 <= x_coord <= 0.5:
                                    stat_list2.append(x_coord)
                                else :
                                    stat_list2.append(1-x_coord)
    stat_list2 = np.array(stat_list2)
    pvalue = stats.ttest_ind(stat_list1,stat_list2).pvalue
    
    plt.figure()
    plt.hist(stat_list1, bin_num, alpha=0.5, label=f' dataset {dataset_names1}, {len(stat_list1)} features', density=True)
    plt.hist(stat_list2, bin_num, alpha=0.5, label=f' dataset {dataset_names2}, {len(stat_list2)} features', density=True)
    plt.annotate(p_value_to_str(pvalue),(0.4,3))
    plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
    plt.legend()
    plt.title("Comparision of feature creation distributions")

    plt.show()

def get_feature_ind(feature_id, cell, pnt_list):
    generation = int(pnt_list[feature_id][1])
    if pnt_list[feature_id][2] == 1:
        ind = np.argmax(cell[generation]['peaks_index']==feature_id)
        res = cell[generation]['peaks'][ind]
    else :
        ind = np.argmax(cell[generation]['troughs_index']==feature_id)
        res = cell[generation]['troughs'][ind]
    return res, generation


def feature_displacement(dataset_names, plot=True, return_smooth_approx=False):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    pole_size = params["pole_region_size"]
    
    stat_list_p = []
    stat_list_c = []
    height_list_p = []
    height_list_c = []
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 6:
                        pos_list = []
                        height_list = []
                        time_list = []
                        len_list = []
                        for elem in pnt_ROI[key]:
                            ind, generation = get_feature_ind(elem, cell, pnt_list)
                            pos_list.append(cell[generation]['xs'][ind])
                            height_list.append(cell[generation]['ys'][ind])
                            time_list.append(cell[generation]['timestamp'])
                            len_list.append([cell[generation]['xs'][0], cell[generation]['xs'][-1]])
                            
                        pos_list = np.array(pos_list)
                        height_list = np.array(height_list )
                        time_list = np.array(time_list)
                        len_list = np.array(len_list)
                        
                        mask = (pos_list - len_list[:,0]  <= pole_size) | (pos_list >= len_list[:,1]-pole_size)
                        if np.sum(mask)>=2:
                            last_in_pole = np.argwhere(mask)[-1][0]
                            if time_list[last_in_pole]- time_list[0]>10:
                                stat_list_p.append(abs(pos_list[last_in_pole]-pos_list[0])/(time_list[last_in_pole]- time_list[0]))
                                height_list_p.append(abs(height_list[last_in_pole]-height_list[0])/(time_list[last_in_pole]- time_list[0]))
                            if last_in_pole < len(mask) -1 :
                                if  time_list[-1] - time_list[last_in_pole]>10:
                                    stat_list_c.append(abs(pos_list[-1] - pos_list[last_in_pole])/(time_list[-1] - time_list[last_in_pole]))
                                    height_list_c.append(abs(height_list[-1] - height_list[last_in_pole])/(time_list[-1] - time_list[last_in_pole]))
                        else :
                            if  time_list[-1] - time_list[0]>10:
                                stat_list_c.append(abs(pos_list[-1] - pos_list[0])/(time_list[-1] - time_list[0]))
                                height_list_c.append(abs(height_list[-1] - height_list[0])/(time_list[-1] - time_list[0]))
                            
                            
                            
                            
                            
                            
                        # root = pnt_ROI[key][0]
                        # leaf = pnt_ROI[key][-1]
                        # time_diff = pnt_list[leaf][-2] - pnt_list[root][-2]
                        # pos_diff = abs(pnt_list[leaf][3]-pnt_list[root][3])
                        
                        # generation = int(pnt_list[leaf][1]) #root
                        # total_length =  abs(cell[generation]['xs'][0]-cell[generation]['xs'][-1])
                        # x_coord = abs(pnt_list[leaf][3]-cell[generation]['xs'][0]) #root
                        
                        # if time_diff >= 10:
                        #     if (x_coord <= pole_size) | (x_coord >= total_length-pole_size):
                        #         stat_list_p.append(pos_diff/time_diff)
                        #     else :
                        #         stat_list_c.append(pos_diff/time_diff)
                                

    

    
    stat_list_p = np.array(stat_list_p)
    stat_list_c = np.array(stat_list_c)
    
    height_list_p = np.array(height_list_p)
    height_list_c = np.array(height_list_c)
    

    title = (
    f"Variation of feature position and height \n with dataset \'{dataset_names}\',\n and {len(stat_list_p),len(stat_list_c) } features tracked"
)
    
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title)
    ax[0].boxplot([stat_list_p*1000,stat_list_c*1000], widths = 0.5,  showfliers=False, medianprops=dict(color='k')) 
    print(title)
    print_stats([stat_list_p*1000,stat_list_c*1000])
    ax[0].set_xticklabels(['Pole', 'Center'])
    ax[0].set_ylabel(f'Variation of feature : \n position '+r'$(n m (min)^{-1})$')
    # ax[0].set_title(title)
    pvalue1 = stats.ttest_ind(stat_list_p,stat_list_c).pvalue
    x1 = 1
    x2 = 2 
    y = 11.5
    h = 0.1
    ax[0].plot([x1, x2], [y, y], color = 'k')
    ax[0].text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    ax[0].set_ylim(-0.5,13)
    

    # _, ax = plt.subplots()
    ax[1].boxplot([height_list_p,height_list_c], widths = 0.5,  showfliers=False, medianprops=dict(color='k')) 
    print_stats([height_list_p,height_list_c])
    ax[1].set_xticklabels(['Pole', 'Center'])
    ax[1].set_ylabel(r'height $(nm (min)^{-1})$')
    # ax.set_title(title)
    pvalue1 = stats.ttest_ind(height_list_p,height_list_c).pvalue
    x1 = 1
    x2 = 2 
    y = 1.1
    h = 0.01
    ax[1].plot([x1, x2], [y, y], color = 'k')
    ax[1].text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    ax[1].set_ylim(-0.05,1.25)

    
    plt.tight_layout()

    
    plt.show()            




            
def feature_len_height_variation(dataset_names):
    params = get_scaled_parameters(data_set=True,stats=True)
    if dataset_names in params.keys():
        datasets = params[dataset_names]
    else : 
        datasets = dataset_names
    
    pole_size = params["pole_region_size"]
    stat_list_len = []
    stat_list_height = []
    
    
    for dataset in datasets:
        for roi_id, cell in load_dataset(dataset, False):
            if len(cell) > 1:
                pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
                for key in pnt_ROI:
                    if len(pnt_ROI[key]) >= 5:
                        
                        dist_list = []
                        for elem in pnt_ROI[key]:
                            ind, generation = get_feature_ind(elem, cell, pnt_list)
                            total_length = abs(cell[generation]['xs'][-1]-cell[generation]['xs'][0])
                            x_coord = abs(cell[generation]['xs'][ind] - cell[generation]['xs'][0])
                            if x_coord > 0.5 * total_length:
                                x_coord = total_length - x_coord
                            dist_list.append(x_coord)
                        dist_list = np.array(dist_list)
                        if (dist_list[0] <= pole_size) == (dist_list[-1] <= pole_size):
                            root = pnt_ROI[key][0]
                            leaf = pnt_ROI[key][-1]
                            _update_len_height_variation(leaf, root, pnt_list, cell, stat_list_len, stat_list_height)
                        elif dist_list[0] <= pole_size and dist_list[-1] > pole_size :
                            swi_ind = np.argmax(dist_list> pole_size)
                            root = pnt_ROI[key][0]
                            leaf = pnt_ROI[key][swi_ind]
                            _update_len_height_variation(leaf, root, pnt_list, cell, stat_list_len, stat_list_height)
                            root = pnt_ROI[key][swi_ind]
                            leaf = pnt_ROI[key][-1]
                            _update_len_height_variation(leaf, root, pnt_list, cell, stat_list_len, stat_list_height)
                            
                        
                        
                      
    
    stat_list_len = np.array(stat_list_len)
    stat_list_height = np.array(stat_list_height)

    title = (
        f"Variation of inter-feature distance \n with dataset \'{dataset_names}\',\n and {np.sum(stat_list_height[:,0] <= pole_size) } + {np.sum(stat_list_height[:,0] > pole_size) } features tracked"
    )

    _, ax = plt.subplots()
    mask1 = stat_list_len[:,0] <= pole_size
    mask2 = stat_list_len[:,0] > pole_size
    val1 = stat_list_len[mask1]
    val2 = stat_list_len[mask2]
    bp=ax.boxplot([1000*val1[:,1],1000*val2[:,1]],  showfliers=False, medianprops=dict(color='k')) 
    print(title)
    print_stats([1000*val1[:,1],1000*val2[:,1]])
    ax.set_xticklabels(['Sub-polar region', 'Central region'])
    ax.set_ylabel(f'Variation of inter-feature \n distance '+ r'$(n m (min)^{-1})$')
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(val1[:,1],val2[:,1]).pvalue
    x1 = 1
    x2 = 2 
    y = 6
    h = 0.02
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    plt.tight_layout()
    
    title = (
        f"Variation of inter-feature amplitude \n with dataset \'{dataset_names}\',\n and {np.sum(stat_list_height[:,0] <= pole_size) } + {np.sum(stat_list_height[:,0] > pole_size) } features tracked"
    )
    _, ax = plt.subplots()
    mask1 = stat_list_height[:,0] <= pole_size
    mask2 = stat_list_height[:,0] > pole_size
    val1 = stat_list_height[mask1]
    val2 = stat_list_height[mask2]
    ax.boxplot([val1[:,1],val2[:,1]],  showfliers=False, medianprops=dict(color='k')) 
    
    print(title)
    print_stats([val1[:,1],val2[:,1]])
    
    ax.set_xticklabels(['Sub-polar region', 'Central region'])
    ax.set_ylabel(f'Variation of inter-feature \n amplitude '+r'$(n m (min)^{-1})$')
    ax.set_title(title)
    pvalue1 = stats.ttest_ind(val1[:,1],val2[:,1]).pvalue
    x1 = 1
    x2 = 2 
    y = 0.4
    h = 0.002
    ax.plot([x1, x2], [y, y], color = 'k')
    ax.text((x1+x2)*.5, y+h, p_value_to_str(pvalue1), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()        
    
    
def _update_len_height_variation(leaf, root, pnt_list, cell, stat_list_len, stat_list_height):
    time_diff = pnt_list[leaf][-2] - pnt_list[root][-2]
    if time_diff >= 100:
        feature_type = bool(pnt_list[root][2])
        feature = feature_type * 'peaks' + (1-feature_type) * 'troughs'
        non_feature = (1-feature_type) * 'peaks' + feature_type * 'troughs'
        generation_root = int(pnt_list[root][1]) 
        generation_leaf = int(pnt_list[leaf][1]) 
        
        total_length =  abs(cell[generation_root]['xs'][0]-cell[generation_root]['xs'][-1])
        x_coord = abs(pnt_list[root][3]-cell[generation_root]['xs'][0]) 
        if x_coord > 0.5 * total_length:
            x_coord = total_length - x_coord
        
        ind_root = np.nonzero(cell[generation_root][feature + '_index'] == root)[0][0]
        root_pos = cell[generation_root][feature][ind_root]
        
        ind_leaf = np.nonzero(cell[generation_leaf][feature + '_index'] == leaf)[0][0]
        leaf_pos = cell[generation_leaf][feature][ind_leaf]
        
        pos_diff_1l = 0
        pos_diff_2l = 0
        pos_diff_1r = 0
        pos_diff_2r = 0
        
        mask_r = cell[generation_root][non_feature]<root_pos
        mask_l = cell[generation_leaf][non_feature]<leaf_pos
        
        if np.any(mask_r) and np.any(mask_l):
                left_root = cell[generation_root][non_feature][np.nonzero(mask_r)[0][-1]]
                left_leaf = cell[generation_leaf][non_feature][np.nonzero(mask_l)[0][-1]]
                pos_diff_1l = cell[generation_leaf]['xs'][leaf_pos] - cell[generation_leaf]['xs'][left_leaf]\
                                - cell[generation_root]['xs'][root_pos] + cell[generation_root]['xs'][left_root] 
                # stat_list_len.append([x_coord, abs(pos_diff_1l)/time_diff]) 
                
                pos_diff_2l = cell[generation_leaf]['ys'][leaf_pos] - cell[generation_leaf]['ys'][left_leaf]\
                                - cell[generation_root]['ys'][root_pos] + cell[generation_root]['ys'][left_root] 
                # stat_list_height.append([x_coord, abs(pos_diff_2l)/time_diff])
                
        mask_r = cell[generation_root][non_feature]>root_pos
        mask_l = cell[generation_leaf][non_feature]>leaf_pos
        
        if np.any(mask_r) and np.any(mask_l):
                right_root = cell[generation_root][non_feature][np.nonzero(mask_r)[0][0]]
                right_leaf = cell[generation_leaf][non_feature][np.nonzero(mask_l)[0][0]]
                pos_diff_1r =  cell[generation_leaf]['xs'][leaf_pos] - cell[generation_leaf]['xs'][right_leaf]\
                                - cell[generation_root]['xs'][root_pos] + cell[generation_root]['xs'][right_root] 
                # stat_list_len.append([x_coord, abs(pos_diff_1r)/time_diff])
                
                pos_diff_2r = cell[generation_leaf]['ys'][leaf_pos] - cell[generation_leaf]['ys'][right_leaf]\
                                - cell[generation_root]['ys'][root_pos] + cell[generation_root]['ys'][right_root] 
                # stat_list_height.append([x_coord, abs(pos_diff_2r)/time_diff])
                
        if pos_diff_1l>0 + pos_diff_1r>0 == 2:

            stat_list_len.append([x_coord,abs(pos_diff_1l-pos_diff_1r)/2 /time_diff]) 
            stat_list_height.append([x_coord,abs(pos_diff_2l-pos_diff_2r)/2 /time_diff])
                
        else :
            if pos_diff_1l>0:
                stat_list_len.append([x_coord,abs(pos_diff_1l) /time_diff]) 
            else:
                stat_list_len.append([x_coord,abs(pos_diff_1r) /time_diff])
            if pos_diff_2l>0:
                stat_list_height.append([x_coord,abs(pos_diff_2l) /time_diff]) 
            else:
                stat_list_height.append([x_coord,abs(pos_diff_2r) /time_diff])

# def feature_displacement_comparison(*dataset_name_list):
#     plt.figure()
#     for dataset_names in dataset_name_list:
#         x2, y2 = feature_displacement(dataset_names, plot=False, return_smooth_approx=True)
#         plt.plot(x2, y2, label=f'{dataset_names}')
#     title = (
#         f"Drift speed of features comparison between datasets \n {dataset_name_list}"
#     )
#     plt.xlabel(r'$\leftarrow$ pole | center $\rightarrow$ ')
#     plt.legend()
#     plt.title(title)
#     plt.show()    
    
 
         

def datasets_statistics():
    params = get_scaled_parameters(data_set=True, paths_and_names=True)
    mask_number = 0
    ROI_number = 0
    frame_num = 0
    
    dicname = params["main_dict_name"]
    data_direc = params["main_data_direc"]
    
    set = "WT_mc2_55/30-03-2015"
    for _, cell in load_dataset(set, False):
        if len(cell) > 5:
            ROI_number+=1
    dic=np.load(os.path.join(data_direc, set, dicname), allow_pickle=True)['arr_0'].item()
    frame_num += len(dic)
    for fichier in dic:
        mask_number += len(dic[fichier]['outlines'])
    print(f'WT good mask number : {mask_number}')
    print(f'WT good ROI number : {ROI_number}')
    print(f'WT good frame number : {frame_num}')
    
    
    mask_number = 0
    ROI_number = 0
    frame_num = 0
    
    set = "INH_after_700"
    for _, cell in load_dataset(set, False):
        if len(cell) > 5:
            ROI_number+=1
    dic=np.load(os.path.join(data_direc, set, dicname), allow_pickle=True)['arr_0'].item()
    # frame_num += len(dic)
    for fichier in dic:
        if dic[fichier]['time']>=700:
            frame_num +=1
            mask_number += len(dic[fichier]['outlines'])
    print(f'INH mask number : {mask_number}')
    print(f'INH good ROI number : {ROI_number}')
    print(f'INH good frame number : {frame_num}')
    
    
    
    
    
    datasets = params['WT_no_drug']
    
    for set in datasets:
        for _, cell in load_dataset(set, False):
            if len(cell) > 5:
                ROI_number+=1
        dic=np.load(os.path.join(data_direc, set, dicname), allow_pickle=True)['arr_0'].item()
        frame_num += len(dic)
        for fichier in dic:
            mask_number += len(dic[fichier]['outlines'])
    print(f'WT no drug mask number : {mask_number}')
    print(f'WT no drug ROI number : {ROI_number}')
    print(f'WT no drug frame number : {frame_num}')
    
    mask_number = 0
    ROI_number = 0
    frame_num = 0
    
    datasets = params['WT_no_drug']
    
    for set in datasets:
        for _, cell in load_dataset(set, False):
            if len(cell) > 5:
                ROI_number+=1
        dic=np.load(os.path.join(data_direc, set, dicname), allow_pickle=True)['arr_0'].item()
        frame_num += len(dic)
        for fichier in dic:
            mask_number += len(dic[fichier]['outlines'])
    
    datasets = params['WT_drug']
    
    for set in datasets:
        for _, cell in load_dataset(set, False):
            if len(cell) > 5:
                ROI_number+=1
        dic=np.load(os.path.join(data_direc, set, dicname), allow_pickle=True)['arr_0'].item()
        frame_num += len(dic)
        for fichier in dic:
            mask_number += len(dic[fichier]['outlines'])
    print(f'WT mask number : {mask_number}')
    print(f'WT ROI number : {ROI_number}')
    print(f'WT frame number : {frame_num}')
    
    datasets = params['no_WT']
    for set in datasets:
        for _, cell in load_dataset(set, False):
            if len(cell) > 5:
                ROI_number+=1
        dic=np.load(os.path.join(data_direc, set, dicname), allow_pickle=True)['arr_0'].item()
        frame_num += len(dic)
        for fichier in dic:
            mask_number += len(dic[fichier]['outlines'])
    
    print(f'Total mask number : {mask_number}')
    print(f'Total good ROI number : {ROI_number}')
    print(f'Total frame number : {frame_num}')
    



def main():
    datasets = None
    dataset = None

    if dataset is None:
        if datasets is None:
            cells_dir = os.path.join("data", "cells")
            pattern = os.path.join(cells_dir, "**", "ROI *", "")
            datasets = glob.glob(pattern, recursive=True)
            datasets = {
                os.path.dirname(os.path.relpath(path, cells_dir)) for path in datasets
            }
    else:
        datasets = [dataset]

    for dataset in datasets:
        stats = compute_stats(dataset)
        plot_stats(dataset, **stats)


if __name__ == "__main__":
    # main()
    plt.rcParams.update({'font.size': 13})
    # feature_number("good")
    # feature_creation("good")     
    feature_general_properties("good")
    feature_creation_time_vs_NETO("good")
    feature_displacement("good") 
    
    feature_len_height_variation ("good")
    feature_properties_pole('WT_no_drug') 
    datasets_statistics()
    plt.rcParams.update({'font.size': 10})
