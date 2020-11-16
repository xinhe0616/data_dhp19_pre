import os
import h5py
path_ = '/media/xinhe/hub/DHP19/h5_dataset_7500_events/346x260/'

path_list = os.listdir(path_)
path_list.sort(key=lambda x:(int(x[x.index('S')+1:x.index('_')]),x[x.index('session')+7:x.index('_',4)],
                             x[x.index('mov')+3:x.index('_',14)]))
print(path_list)
print(len(path_list))
def load_file_(filepath):
    if filepath.endswith('.h5'):
        with h5py.File(filepath, 'r') as f_:
            data = (f_[list(f_.keys())[0]])[()]
    else:
        raise ValueError('.h5 required format.')
    return data
load_file_('/media/xinhe/hub/DHP19/h5_dataset_7500_events/346x260/S1_session2_mov1_7500events.h5')
# S1_session2_mov1_7500events.h5