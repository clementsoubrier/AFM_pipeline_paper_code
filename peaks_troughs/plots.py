import os
import sys

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from scaled_parameters import get_scaled_parameters
from peaks_troughs.group_by_cell import load_dataset, get_peak_troughs_lineage_lists, load_cell
from peaks_troughs.preprocess import evenly_spaced_resample
from peaks_troughs.stiffness_stats import extract_feature
from peaks_troughs.division_detection import detect_division

# Functions to visualize kymographs

def plot_single_centerline(xs, ys, peaks, troughs):
    plt.figure()
    plt.plot(xs, ys, color="black")
    for x_1 in peaks:
        plt.plot(xs[x_1],ys[x_1], marker='v', color="red")
    for x_1 in troughs:
        plt.plot(xs[x_1],ys[x_1], marker='o', color="green")
    plt.xlabel("Curvilign abscissa (µm)")
    plt.ylabel("Height")
    plt.show()


def plot_cell_centerlines(*cells_and_id, dataset=''):   #first cell is the mother, each argument is a tuple (cell, id)
    
    plt.figure()
    cell_centerlines=[]
    cell_peaks=[]
    cell_troughs=[]
    cell_timestamps=[]
    title = []
    base_time=cells_and_id[0][0][0]["timestamp"]
    for cell_id in cells_and_id:
        cell = cell_id[0]
        roi_id = cell_id[1]
        title.append(roi_id)

        for frame_data in cell:
            cell_centerlines.append(np.vstack((frame_data["xs"],frame_data["ys"])))
            cell_peaks.append(frame_data["peaks"])
            cell_troughs.append(frame_data["troughs"])
            cell_timestamps.append(frame_data["timestamp"]-base_time)

        peaks_x = []
        peaks_y = []
        troughs_x = []
        troughs_y = []
        for centerline, peaks, troughs,timestamp in zip(cell_centerlines, cell_peaks,
                                            cell_troughs,cell_timestamps):
            xs = centerline[0, :]
            ys = centerline[1, :] + 2*timestamp 
            if peaks.size:
                peaks_x.extend(xs[peaks])
                peaks_y.extend(ys[peaks])
            if troughs.size:
                troughs_x.extend(xs[troughs])
                troughs_y.extend(ys[troughs])
            plt.plot(xs, ys, color='k')
        plt.scatter(peaks_x, peaks_y, c="red")
        plt.scatter(troughs_x, troughs_y, c="green")
        
        pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
        for key in pnt_ROI :
            coord_x = []
            coord_y = []
            for elem in pnt_ROI[key]:
                coord_x.append(pnt_list[elem,3])
                coord_y.append(pnt_list[elem,4] + 2*(pnt_list[elem,5]-base_time))
            plt.plot(coord_x, coord_y, color = 'b')
        plt.xlabel(r'centerline length ($\mu m$)')
        plt.ylabel(r'height ($nm$)')
        
        

    plt.title(title)
    plt.show()

def kymograph(*cells_and_id,  dataset='', division_point = None, saving=False, saving_name='',  dir_im=''):   #first cell is the mother, each argument is a tuple (cell, id)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    

    base_time=cells_and_id[0][0][0]["timestamp"]
    pixelsize=[]
    title = []
    v_min = None
    v_max = None
    ys_l=[]
    first = True
    for cell_id in cells_and_id:
        cell = cell_id[0]
        if first:
            xs,ys = cell[0]["xs"],cell[0]["ys"]
            x_init = xs[0]
            base_time = cell[0]["timestamp"]
            y_init = ys[0]
            first = False
        title.append(cell_id[1])
        for frame_data in cell:
            xs,ys = frame_data["xs"],frame_data["ys"]
            pixelsize.append((xs[-1]-xs[0])/(len(xs)-1))
            ys_l.append(ys)
            base_time = min (base_time, frame_data["timestamp"])
            y_init = min(y_init,np.min(ys))
    ys_l = np.concatenate(ys_l)
    v_max = np.quantile(ys_l,0.95)
    v_min = np.quantile(ys_l,0.05)
    step=min(pixelsize)
    
    for cell_id in cells_and_id:
        cell_centerlines=[]
        cell_centerlines_renorm=[]
        cell_peaks=[]
        cell_troughs=[]
        cell_timestamps=[]
        cell = cell_id[0]
        roi_id = cell_id[1]
        for frame_data in cell:
            xs,ys=frame_data["xs"],frame_data["ys"]
            cell_centerlines.append(np.vstack((xs,ys)))
            xs,ys=evenly_spaced_resample(xs,ys,step)
            cell_centerlines_renorm.append(np.vstack((xs,ys)))
            cell_peaks.append(frame_data["peaks"])
            cell_troughs.append(frame_data["troughs"])
            cell_timestamps.append(frame_data["timestamp"])

        peaks_x = []
        peaks_y = []
        peaks_z = []
        troughs_x = []
        troughs_y = []
        troughs_z = []
        

        xs_data=np.ravel(np.concatenate([cent[0] for cent in cell_centerlines]))
        xs_min=np.min(xs_data)
        xs_max=np.max(xs_data)
        width=round((xs_max-xs_min)/step+1)
        shape = (len(cell_centerlines), width)
        
        xs_3d = np.zeros(shape, dtype=np.float64)
        ys_3d = np.zeros(shape, dtype=np.float64)
        zs_3d = np.zeros(shape, dtype=np.float64)


        for i, centerline in enumerate(cell_centerlines_renorm):
            xs = centerline[0,:]
            ys = centerline[1,:]
            preval=round((xs[0]-xs_min)/step)
            postval=preval+len(xs)
            xs_3d[i, :preval] = xs[0]
            xs_3d[i, postval:] = xs[-1]
            xs_3d[i, preval:postval]=xs
            xs_3d[i, :]-= x_init
            ys_3d[i, :] = cell_timestamps[i] - base_time
            zs_3d[i, :preval] = ys[0]
            zs_3d[i, postval:] = ys[-1]
            zs_3d[i, preval:postval]=ys
            zs_3d[i, :] -= y_init
        ax.plot_surface(xs_3d, ys_3d, zs_3d, cmap="viridis", lw=0.5, rstride=1,
                        cstride=1, alpha=0.6, edgecolor='none',
                        norm=mplc.PowerNorm(gamma=2.3,vmin=v_min-y_init, vmax=v_max-y_init))
        divx = []
        divy = []
        divz = []
        for centerline, peaks, troughs,timestamp in zip(cell_centerlines, cell_peaks,
                                            cell_troughs,cell_timestamps):
            xs = centerline[0, :] - x_init
            ys = centerline[1, :] - y_init
            zs = timestamp * np.ones(len(xs)) - base_time
            if peaks.size:
                peaks_x.extend(xs[peaks])
                peaks_y.extend(ys[peaks])
                peaks_z.extend([timestamp - base_time]*len(peaks))
            if troughs.size:
                troughs_x.extend(xs[troughs])
                troughs_y.extend(ys[troughs])
                troughs_z.extend([timestamp - base_time]*len(troughs))
            ax.plot3D(xs, zs, ys,c="k")
            if division_point is not None and division_point[1] >= timestamp - base_time:
                div_ind = np.argmin((xs + x_init-division_point[0])**2)
                divx.append(xs[div_ind])
                divy.append(timestamp - base_time)
                divz.append(ys[div_ind] + 10)
        if divx!=[]:
            ax.scatter(divx, divy, divz, alpha=1,  color="magenta")          #marker='|',markersize = 10,, linestyle='dashed'
        ax.scatter(peaks_x,  peaks_z, peaks_y, c="r", s=10, alpha=1)
        ax.scatter(troughs_x, troughs_z, troughs_y, c="k", s=10,  alpha=1)
        
        
        pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
        for key in pnt_ROI :
            coord_x = []
            coord_y = []
            coord_z = []
            for elem in pnt_ROI[key]:
                coord_x.append(pnt_list[elem,3] - x_init)
                coord_y.append(pnt_list[elem,4] - y_init)
                coord_z.append(pnt_list[elem,5] - base_time) 
            ax.plot(coord_x, coord_z, coord_y, color = 'b')
    
    ax.set_zlabel(r'height ($n m$)')
    ax.set_ylabel(r'time ($min$)')
    ax.set_xlabel(r' centerline length ($\mu m$)')
    ax.zaxis.labelpad=10
    # ax.set_xlim([-9, 4])
    # ax.set_zlim([0, 1000])
    # ax.set_ylim([0, 600])
    ax.view_init(elev=60, azim=-88, roll=0)
    plt.title(title)
    plt.tight_layout()
    if saving:

        plt.savefig(os.path.join(dir_im, saving_name + '.svg'), format='svg')
        plt.close()
            

    





def kymograph_feature(*cells_and_id,  dataset='', feature='DMTModulus_fwd', averaged = True):   #first cell is the mother, each argument is a tuple (cell, id)

    params = get_scaled_parameters(paths_and_names=True)
    
    dicname = params["main_dict_name"]
    listname = params["masks_list_name"]
    data_direc = params["main_data_direc"]
    
    masks_list = np.load(os.path.join(data_direc, dataset, listname), allow_pickle=True)['arr_0']
    main_dict = np.load(os.path.join(data_direc, dataset, dicname), allow_pickle=True)['arr_0'].item()
    
    plt.figure()
    ax = plt.axes(projection='3d')
    base_time=cells_and_id[0][0][0]["timestamp"]
    pixelsize=[]
    for cell_id in cells_and_id:
        cell = cell_id[0]
        for frame_data in cell:
            pixelsize.append((frame_data["xs"][-1]-frame_data["xs"][0])/(len(frame_data["xs"])-1))
    
    step=min(pixelsize)
    for cell_id in cells_and_id:
        cell_centerlines=[]
        cell_centerlines_renorm=[]
        cell_peaks=[]
        cell_troughs=[]
        cell_timestamps=[]
        cell = cell_id[0]
        roi_id = cell_id[1]
        for frame_data in cell:
            xs = frame_data["xs"]
            ys = extract_feature(frame_data, main_dict, masks_list, feature, averaged)
            
            cell_centerlines.append(np.vstack((xs,ys)))
            xs,ys=evenly_spaced_resample(xs,ys,step)
            cell_centerlines_renorm.append(np.vstack((xs,ys)))
            cell_peaks.append(frame_data["peaks"])
            cell_troughs.append(frame_data["troughs"])
            cell_timestamps.append(frame_data["timestamp"]-base_time)

        peaks_x = []
        peaks_y = []
        peaks_z = []
        troughs_x = []
        troughs_y = []
        troughs_z = []
        

        xs_data=np.ravel(np.concatenate([cent[0] for cent in cell_centerlines]))
        xs_min=np.min(xs_data)
        xs_max=np.max(xs_data)
        width=round((xs_max-xs_min)/step+1)
        shape = (len(cell_centerlines), width)
        
        xs_3d = np.zeros(shape, dtype=np.float64)
        ys_3d = np.zeros(shape, dtype=np.float64)
        zs_3d = np.zeros(shape, dtype=np.float64)


        for i, centerline in enumerate(cell_centerlines_renorm):
            xs = centerline[0,:]
            ys = centerline[1,:]
            preval=round((xs[0]-xs_min)/step)
            postval=preval+len(xs)
            xs_3d[i, :preval] = xs[0]
            xs_3d[i, postval:] = xs[-1]
            xs_3d[i, preval:postval]=xs
            ys_3d[i, :] = cell_timestamps[i]
            zs_3d[i, :preval] = ys[0]
            zs_3d[i, postval:] = ys[-1]
            zs_3d[i, preval:postval]=ys
        ax.plot_surface(xs_3d, ys_3d, zs_3d, cmap="viridis", lw=0.5, rstride=1,
                        cstride=1, alpha=0.5, edgecolor='none',
                        norm=mplc.PowerNorm(gamma=1.5))
        

        for centerline, peaks, troughs,timestamp in zip(cell_centerlines, cell_peaks,
                                            cell_troughs,cell_timestamps):
            xs = centerline[0, :]
            ys = centerline[1, :] 
            zs = timestamp * np.ones(len(xs))
            if peaks.size:
                peaks_x.extend(xs[peaks])
                peaks_y.extend(ys[peaks])
                peaks_z.extend([timestamp]*len(peaks))
            if troughs.size:
                troughs_x.extend(xs[troughs])
                troughs_y.extend(ys[troughs])
                troughs_z.extend([timestamp]*len(troughs))
            ax.plot3D(xs, zs, ys,c="k")
        ax.scatter(peaks_x,  peaks_z, peaks_y,c="r")
        ax.scatter(troughs_x, troughs_z, troughs_y, c="k")
        
        
        # pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
        # for key in pnt_ROI :
        #     coord_x = []
        #     coord_y = []
        #     coord_z = []
        #     for elem in pnt_ROI[key]:
        #         coord_x.append(pnt_list[elem,3])
        #         coord_y.append(pnt_list[elem,4] )
        #         coord_z.append(pnt_list[elem,5]-base_time)
        #     ax.plot(coord_x, coord_z, coord_y, color = 'b')
        
    unit = main_dict[next(iter(main_dict))]['units'][feature]
    
    ax.set_zlabel(f'{feature} ({unit})')
    ax.set_ylabel(r'time ($min$)')
    ax.set_xlabel(r' centerline length ($\mu m$)')

    plt.title(roi_id + f', {feature}')





def main(Directory='all', saving=False):
    params = get_scaled_parameters(data_set=True)
    if Directory in params.keys():
        datasets = params[Directory]
    elif isinstance(Directory, list)  : 
        datasets = Directory
    elif isinstance(Directory, str)  : 
        raise NameError('This directory does not exist')
    
    
    for dataset in datasets:
        params = get_scaled_parameters(paths_and_names=True)
        data_direc = params["main_data_direc"]
        roi_dic_name = params["roi_dict_name"]
        final_img_dir = params["final_img_dir"]
        
        roi_dic = np.load(os.path.join(data_direc, dataset, roi_dic_name), allow_pickle=True)['arr_0'].item()
        dir_im = ''
        if saving:
            dir_im = os.path.join(final_img_dir,'kymographs',dataset) 
            
            if os.path.exists(dir_im):
                for file in os.listdir(dir_im):
                    os.remove(os.path.join(dir_im, file))
            else:
                os.makedirs(dir_im)
        
        
        
        
        for roi_id, cell in load_dataset(dataset, False):
            
            ## indent or un-indent to have different visualizations
            if len(cell)>5:
                
                
                if len(roi_dic[roi_id]['Children']) >= 1:
                    division_point = detect_division(cell[-1], roi_id, roi_dic, dataset, use_one_daughter = True)
                    if division_point is not None:
                        division_point = [cell[-1]['xs'][division_point], cell[-1]['timestamp']-cell[0]['timestamp'],cell[-1]['ys'][division_point]]
                    for daughter_cell in roi_dic[roi_id]['Children']:
                        d_cell = load_cell(daughter_cell, dataset=dataset)
                        if len(d_cell)>5:
                            lineage = [(cell, roi_id),(d_cell, daughter_cell)]
                            
                            kymograph(*lineage, dataset=dataset, division_point=division_point, saving=saving, saving_name=roi_id.replace(" ","")+'+'+daughter_cell.replace(" ",""), dir_im=dir_im)
                            # kymograph_feature(*lineage, dataset=dataset)
                
                
                # kymograph_feature((cell, roi_id), dataset=dataset)
                # kymograph((cell, roi_id), dataset=dataset)
                
                # if len(roi_dic[roi_id]['Children']) >= 1:
                #     for daughter_cell in roi_dic[roi_id]['Children']:
                #         d_cell = load_cell(daughter_cell, dataset=dataset)
                #         kymograph((d_cell, daughter_cell), dataset=dataset)
                
                # plot_cell_centerlines((cell, roi_id), dataset=dataset)
    plt.show()
    
    

    



if __name__ == "__main__":
    main(Directory="all", saving = True)
    
