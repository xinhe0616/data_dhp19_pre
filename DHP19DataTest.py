from DHP19Data import Dhp19PoseDataset
train_data_dir = 'dhp_lstm/'
train_label_dir = 'dhp_lstm/'
temporal = 5
train_data = Dhp19PoseDataset(data_dir=train_data_dir, label_dir=train_label_dir, temporal=temporal, train=True)
