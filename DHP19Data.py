from torch.utils.data import Dataset
import os
import json_tricks as json
import numpy as np
import cv2
import matplotlib.image as mpimg
import torch

import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')
import scipy.misc

class Dhp19PoseDataset(Dataset):
    def __init__(self, data_dir, label_dir, train, temporal=5, joints=13, transform=None, sigma=1):
        self.seqs = os.listdir(data_dir)
        self.data_dir = data_dir
        self.temporal_dir = []
        self.labels = dict()
        self.label_dir = label_dir
        self.image_size=[192, 256]
        self.heatmap_size =[48, 64]
        self.size = 0
        self.gen_temporal_dir()


    def plot_2d(self, dvs_frame, joint):
        " To plot image and 2D ground truth and prediction "
        # plt.figure()
        plt.cla()
        plt.imshow(dvs_frame)

        temp = np.array(joint)
        plt.plot(temp[:, 0], temp[:, 1], '.', c='red', label='gt')
        plt.show()
        plt.pause(0.0002)

    plt.ion()

    def show(self):
        print(len(self.temporal_dir))
        for idx in range(len(self.temporal_dir)):
            img_path = self.temporal_dir[idx]
            print(img_path)
            joint = self.labels[idx]
            data_numpy = mpimg.imread(img_path)
            joint = np.array(joint)
            self.plot_2d(data_numpy,joint)

    def gen_temporal_dir(self):
        self.seqs.sort(key=lambda x: (int(x[:2]), x[3:4], x[5:6]))
        for seq in self.seqs:
            if seq == 'annot':
                continue
            image_path = os.path.join(self.data_dir, seq)
            imgs = os.listdir(image_path+'/images/')
            imgs.sort(
                key=lambda x: (int(x[x.index('_', 6) + 1:x.index('.')])))
            img_num = len(imgs)
            label_path = os.path.join(self.label_dir, seq)
            lables = json.load(open(label_path + '/annot/' + seq + '.json'))
            for i in range(img_num):
                self.temporal_dir.append(os.path.join(image_path + '/images', imgs[i]))
                self.labels[self.size] = lables[str(i)]
                self.size+=1
        self.show()
        print('total numbers of image sequence is ' + str(len(self.temporal_dir)))







