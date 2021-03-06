import json_tricks as json
import os
import numpy as np


path_ = '../data_test/dhp_lstm/'
seqs = os.listdir(path_)
temporal_dir = []
seqs.sort(key=lambda x: (int(x[:2]), x[3:4], x[5:6]))
# seqs = seqs[-2:]
size=0
labels=dict()
for seq in seqs:
    if seq == 'annot':
        continue
    image_path = os.path.join(path_, seq)  #
    imgs = os.listdir(image_path + '/images/')

    imgs.sort(
        key=lambda x: (int(x[x.index('_', 6) + 1:x.index('.')])))

    img_num = len(imgs)
    label_path = os.path.join(path_, seq)
    lable = json.load(open(label_path + '/annot/' + seq + '.json'))
    # print(len(lable))
    for i in range(img_num):
        temporal_dir.append(os.path.join(image_path + '/images', imgs[i]))
        labels[size] = lable[str(i)]
        size += 1


print("finish")

# temporal_dir, labels保存了图片路径和所有annot
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
import scipy.misc
def plot_2d(dvs_frame, joint):
    " To plot image and 2D ground truth and prediction "
    # plt.figure()
    plt.cla()
    plt.imshow(np.zeros((260,260)))
    # plt.imshow(dvs_frame)



    temp = np.array(joint)
    if temp[temp <= 0].any():
        print(temp)
    # print(temp)
    plt.plot(temp[:, 0], temp[:, 1], '.', c='red', label='gt')
    plt.show()
    plt.pause(0.0002)

plt.ion()
save_dir="out/"


for i in range(len(temporal_dir)):
    # print(len(labels[i]))
    # print(i)
    im = Image.open(temporal_dir[i])
    label = labels[i]
    plot_2d(im, label)