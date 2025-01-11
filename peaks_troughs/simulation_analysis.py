import numpy as np

# Computing statistics from the reaction-diffusion system simulations

def load_data():
    return np.load('/data/simulations/res.npz',allow_pickle=True)['arr_0'].item()


def detect_zeros(arr):
    is_pos = ( arr[1:]- arr[:-1]) >= 0
    sign_change = np.logical_xor(is_pos[1:], is_pos[:-1])
    return (np.argwhere(sign_change)+1)[:,0]
    
    
def track_simulation():
    dic = np.load('data/simulations/res.npz',allow_pickle=True)['arr_0'].item()
    for elem in dic :
        time_range = dic[elem]['t']
        x_array = dic[elem]['x']
        uv_array = dic[elem]['val']
        
        
        old_features = detect_zeros(uv_array[-1,:,0])
        lin = -10*np.ones((len(time_range), len(old_features)))
        old_feat_pos = x_array[-1,old_features]
        old_feat_ind = np.linspace(0,len(old_features)-1,len(old_features),dtype=int)
        old_feat_ind = {}
        count = 0
        for feat in old_features:
            old_feat_ind[feat] = count
            count += 1
        
        lin[-1,:] = old_features
        for i,vals in enumerate(uv_array[-2::-1,:,0]):
            new_features = detect_zeros(vals)
            new_feat_pos = x_array[-2-i,new_features]
            dist_arg_list = np.array([[old_features[j],new_features[np.argmin((new_feat_pos-old_feat_pos[j])**2)]] for j in range(len(old_feat_pos))])
            dist_arg_list2 = np.array([[old_features[np.argmin((old_feat_pos-new_feat_pos[j])**2)],new_features[j]] for j in range(len(new_feat_pos))])
            
            if np.all(dist_arg_list[1:,1]!=dist_arg_list[:-1,1]):
                new_feat_ind = {}
                for j,arg in enumerate(dist_arg_list[:,0]):
                    if arg in old_feat_ind:
                        new_feat_ind[dist_arg_list[j,1]] = old_feat_ind[arg]
                        lin[-2-i,old_feat_ind[arg]] = dist_arg_list[j,1]
                old_feat_ind = new_feat_ind  
                old_features = new_features
                old_feat_pos = new_feat_pos
            elif np.all(dist_arg_list2[1:,0]!=dist_arg_list2[:-1,0]):
                new_feat_ind = {}
                for j,arg in enumerate(dist_arg_list2[:,0]):
                    if arg in old_feat_ind:
                        new_feat_ind[dist_arg_list2[j,1]] = old_feat_ind[arg]
                        lin[-2-i,old_feat_ind[arg]] = dist_arg_list2[j,1]
                old_feat_ind = new_feat_ind  
                old_features = new_features
                old_feat_pos = new_feat_pos
            else:
                raise ValueError('problem in tracking')
                
        print(lin)
                
            
            

        
if __name__ == "__main__":
    track_simulation()