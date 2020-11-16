import matplotlib;matplotlib.use("TKAgg")
image_h, image_w, num_joints = 260, 346, 13
image_size = [192, 256]
heatmap_size = [48, 64]

def loss_history_init(temporal=5):
    loss_history = {}
    for t in range(temporal):
        loss_history['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history
def save_loss(predict_heatmaps, label_map, epoch, step, criterion, train, temporal=5, save_dir='ckpt/'):
    loss_save = loss_history_init(temporal=temporal)

    predict = predict_heatmaps[0]
    target = label_map[:, 0, :, :, :]
    initial_loss = criterion(predict, target)  # loss initial
    total_loss = initial_loss

    for t in range(temporal):
        predict = predict_heatmaps[t + 1]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(predict, target)  # loss in each stage
        total_loss += tmp_loss
        loss_save['temporal' + str(t)] = float('%.8f' % tmp_loss)

    total_loss = total_loss
    loss_save['total'] = float(total_loss)

    # save loss to file
    # if train is True:
    #     if not os.path.exists(save_dir + 'loss_epoch' + str(epoch)):
    #         os.mkdir(save_dir + 'loss_epoch' + str(epoch))
    #     json.dump(loss_save, open(save_dir + 'loss_epoch' + str(epoch) + '/s' + str(step).zfill(4) + '.json', 'w', encoding="utf8"))
    #
    # else:
    #     if not os.path.exists(save_dir + 'loss_test/'):
    #         os.mkdir(save_dir + 'loss_test/')
    #     json.dump(loss_save, open(save_dir + 'loss_test/' + str(step).zfill(4) + '.json', 'w', encoding="utf8"))

    return total_loss
