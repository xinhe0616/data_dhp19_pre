import json_tricks as json
import os
import numpy as np
import cv2
import h5py
from os.path import join

path_ = '../data_test/346x260/'
# path_ = '../h5_dataset_7500_events/346x260/'

# path_ = 'data_small/'
P_mat_dir = 'P_matrices'
image_h, image_w, num_joints = 260, 346, 13
P_mat_cam = np.load(join(P_mat_dir, 'P2.npy'))
path_list = os.listdir(path_)
path_list.sort(key=lambda x:(int(x[x.index('S')+1:x.index('_')]),x[x.index('session')+7:x.index('_',4)],
                             x[x.index('mov')+3:x.index('_',14)]))
ch_idx = 3
def makefile(file):
    if os.path.exists(file):
        pass
    else:
        open(file, 'w')
def mkdir(file):
    if os.path.exists(file):
        pass
    else:
        os.mkdir(file)
def load_file_(filepath):
    if filepath.endswith('.h5'):
        with h5py.File(filepath, 'r') as f_:
            data = (f_[list(f_.keys())[0]])[()]
    else:
        raise ValueError('.h5 required format.')
    return data

mkdir(path_+'../'+'dhp_lstm')
gt_db = []

for i in range(len(path_list)):

    file = path_list[i]
    # print(item)
    if "label" in file:
        pass
        # print(file)
    else:
        # print(file)
        # print(i / len(path_list) * 100 / 2)
        subj = file[file.index('S') + 1:file.index('_')]
        sess = file[file.index('session') + 7:file.index('_', 8)]
        mov = file[file.index('mov') + 3:file.index('_', 16)]
        file_name = (2-len(subj))*'0'+subj+'_'+sess+'_'+mov
        images_all = load_file_(path_+file)
        # print(images_all.shape)    # (k, 260, 346, 4)
        file = file[:file.index('.h5')] + "_label" + file[file.index('.h5'):]
        vicon_xyz_all = load_file_(path_+file)
        # print(vicon_xyz_all.shape)  # (k, 3, 13)

        mkdir(path_ + '../dhp_lstm/' + file_name)
        mkdir(path_+'../dhp_lstm/'+file_name+'/annot/')
        mkdir(path_+'../dhp_lstm/'+file_name+'/images/')
        res = dict()
        num_img = 0
        for t in range(vicon_xyz_all.shape[0]):
            vicon_xyz = vicon_xyz_all[t]
            image = images_all[t, :, :, ch_idx]*50
            image = image[2:258, 46:302]
            # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
            vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1, 13])], axis=0)
            coord_pix_all_cam2_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
            coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]

            u = coord_pix_all_cam2_homog_norm[0]-44
            v = image_h - coord_pix_all_cam2_homog_norm[1]  # flip v coordinate to match the image direction
            if u[np.isnan(u)].any():
                continue
                # print(u[np.isnan(u)])
                # print("error"+"!"*30)

            num_img +=1
            if v[v<=0].any():
                v[v <= 0] = 1
            if u[u<=0].any():
                u[u <= 0] = 1
            # if v[v <= 0].any():
            #     print(v[v<=0])
            #     print("error"+"!"*30)

            # u[np.isnan(u)] = 1
            # v[np.isnan(v)] = 1
            # u[u>=260] = 259
            u = u.astype(np.int32)
            v = v.astype(np.int32)

            image_name = '_'+str(num_img)+".jpg"
            joints_3d = np.zeros((13, 3), dtype=np.float)
            joints_3d_vis = np.ones(13, dtype=np.int)
            joints_3d[:, 0] = u
            joints_3d[:, 1] = v
            joints_3d[:, 0:2] = joints_3d[:, 0:2] - 1
            res[t] = joints_3d.tolist()

            sava_path = path_+'../dhp_lstm/'+file_name+'/images/'+file_name+image_name
            cv2.imwrite(sava_path, image)
        makefile(path_+'../dhp_lstm/'+file_name+'/annot/' +file_name+ '.json')
        with open(path_+'../dhp_lstm/'+file_name+'/annot/' +file_name+ '.json', "w") as f:
            json.dump(res, f)
        print("finish")
