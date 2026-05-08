import math
import numpy as np
import torch
from tqdm import tqdm
from dotmap import DotMap
from mne.decoding import CSP
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from utils.functions import *
import utils.hypergraph_utils as hgut


def get_DTU_data(model, name="S1", timelen=1, data_document_path="/data1/xyhe/Data/DTU_mat"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, label):
            self.data = torch.Tensor(data)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.label[index]
        
    class CustomDatasets_DHGCN(Dataset):
        def __init__(self, data, labels, G, G_sp):
            self.data = data
            self.labels = labels
            self.G = G
            self.G_sp = G_sp

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            G = self.G[idx]
            sp = self.G_sp[idx]
            return sample, G, sp, label
        
    class CustomDatasets_DBPNet(Dataset):
        # initialization: data and label
        def __init__(self, seq_data, fre_data, event_data):
            self.seq_data = seq_data
            self.fre_data = fre_data
            self.label = event_data

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            seq_data = torch.Tensor(self.seq_data[index])
            fre_data = torch.Tensor(self.fre_data[index])
            label = torch.Tensor(self.label[index])

            return seq_data, fre_data, label

    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_wavA_data = []
        mat_wavB_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']
        mat_eeg = matstruct_contents[0, 0]['eeg']
        mat_wavA = matstruct_contents[0, 0]['wavA']
        mat_wavB = matstruct_contents[0, 0]['wavB']
        for i in range(mat_eeg.shape[1]):
            mat_eeg_data.append(mat_eeg[0, i])
            mat_wavA_data.append(mat_wavA[0, i])
            mat_wavB_data.append(mat_wavB[0, i])
            mat_event_data.append(mat_event_value[i][0][0])

        return mat_eeg_data, mat_event_data

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
            test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
            train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
            test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        test_label = np.stack(test_label, axis=0).reshape(-1, 1)

        return train_eeg, test_eeg, train_label, test_label
    
    def cal_graph(data, K_neigs, is_probH=True, m_prob=1):
        G = []
        for i in tqdm(range(data.shape[0]), disable=True):
            tmp = hgut.construct_H_with_KNN(data[i], K_neigs=K_neigs, is_probH=is_probH, m_prob=m_prob)
            g = hgut.generate_G_from_H(tmp)
            g = torch.tensor(g)
            G.append(g)
        return G

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    time_len = timelen
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 64
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.people_number = 18
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60
    args.cell_number = 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.csp_comp = 64
    args.label_col = 0
    args.log_path = "Results/2s"
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)
    eeg_data = eeg_data[:, :, 0:64]

    event_data = np.array(event_data)
    print(eeg_data.shape)
    eeg_data = np.vstack(eeg_data)
    eeg_data = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = np.array(eeg_data)
    print(eeg_data.shape)
    event_data = np.squeeze(event_data - 1)

    train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)

    if model != "FAConformer" and model != "DHGCN" and model != "DBPNet":
        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        del eeg_data
        del event_data

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data, train_label = train_data[indices], train_label[indices]

        valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
        train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

        if model in ["DARNet", "TMSANet", "IFNet"]:
            train_data = np.squeeze(train_data)
            valid_data = np.squeeze(valid_data)
            test_data = np.squeeze(test_data)
        else:
            train_data = np.expand_dims(train_data, axis=1)
            valid_data = np.expand_dims(valid_data, axis=1)
            test_data = np.expand_dims(test_data, axis=1)

        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)

    elif model == "FAConformer":

        args.delta_low = 1
        args.delta_high = 4
        args.theta_low = 4
        args.theta_high = 8
        args.alpha1_low = 8
        args.alpha1_high = 10
        args.alpha2_low = 10
        args.alpha2_high = 13
        args.beta1_low = 13
        args.beta1_high = 16
        args.beta2_low = 16
        args.beta2_high = 20
        args.beta3_low = 20
        args.beta3_high = 26
        args.gamma1_low = 26
        args.gamma1_high = 32

        args.frequency_resolution = args.fs / args.window_length

        args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
        args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
        args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
        args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
        args.point2_low = math.ceil(args.alpha1_low / args.frequency_resolution)
        args.point2_high = math.ceil(args.alpha1_high / args.frequency_resolution) + 1
        args.point3_low = math.ceil(args.alpha2_low / args.frequency_resolution)
        args.point3_high = math.ceil(args.alpha2_high / args.frequency_resolution) + 1
        args.point4_low = math.ceil(args.beta1_low / args.frequency_resolution)
        args.point4_high = math.ceil(args.beta1_high / args.frequency_resolution) + 1
        args.point5_low = math.ceil(args.beta2_low / args.frequency_resolution)
        args.point5_high = math.ceil(args.beta2_high / args.frequency_resolution) + 1
        args.point6_low = math.ceil(args.beta3_low / args.frequency_resolution)
        args.point6_high = math.ceil(args.beta3_high / args.frequency_resolution) + 1
        args.point7_low = math.ceil(args.gamma1_low / args.frequency_resolution)
        args.point7_high = math.ceil(args.gamma1_high / args.frequency_resolution) + 1

        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        del eeg_data
        del event_data

        # filter different frequency band
        train_data0 = filter_signal_by_fft(train_data, args.point0_low, args.point0_high, args.window_length)
        test_data0 = filter_signal_by_fft(test_data, args.point0_low, args.point0_high, args.window_length)
        train_data1 = filter_signal_by_fft(train_data, args.point1_low, args.point1_high, args.window_length)
        test_data1 = filter_signal_by_fft(test_data, args.point1_low, args.point1_high, args.window_length)
        train_data2 = filter_signal_by_fft(train_data, args.point2_low, args.point2_high, args.window_length)
        test_data2 = filter_signal_by_fft(test_data, args.point2_low, args.point2_high, args.window_length)
        train_data3 = filter_signal_by_fft(train_data, args.point3_low, args.point3_high, args.window_length)
        test_data3 = filter_signal_by_fft(test_data, args.point3_low, args.point3_high, args.window_length)
        train_data4 = filter_signal_by_fft(train_data, args.point4_low, args.point4_high, args.window_length)
        test_data4 = filter_signal_by_fft(test_data, args.point4_low, args.point4_high, args.window_length)
        train_data5 = filter_signal_by_fft(train_data, args.point5_low, args.point5_high, args.window_length)
        test_data5 = filter_signal_by_fft(test_data, args.point5_low, args.point5_high, args.window_length)
        train_data6 = filter_signal_by_fft(train_data, args.point6_low, args.point6_high, args.window_length)
        test_data6 = filter_signal_by_fft(test_data, args.point6_low, args.point6_high, args.window_length)
        train_data7 = filter_signal_by_fft(train_data, args.point7_low, args.point7_high, args.window_length)
        test_data7 = filter_signal_by_fft(test_data, args.point7_low, args.point7_high, args.window_length)

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        
        train_label = train_label[indices]
        train_data0 = train_data0[indices]
        train_data1 = train_data1[indices]
        train_data2 = train_data2[indices]
        train_data3 = train_data3[indices]
        train_data4 = train_data4[indices]
        train_data5 = train_data5[indices]
        train_data6 = train_data6[indices]
        train_data7 = train_data7[indices]

        valid_label = train_label[args.n_train:]
        train_label = train_label[:args.n_train]
        valid_data0 = train_data0[args.n_train:]
        train_data0 = train_data0[:args.n_train]
        valid_data1 = train_data1[args.n_train:]
        train_data1 = train_data1[:args.n_train]
        valid_data2 = train_data2[args.n_train:]
        train_data2 = train_data2[:args.n_train]
        valid_data3 = train_data3[args.n_train:]
        train_data3 = train_data3[:args.n_train]
        valid_data4 = train_data4[args.n_train:]
        train_data4 = train_data4[:args.n_train]
        valid_data5 = train_data5[args.n_train:]
        train_data5 = train_data5[:args.n_train]
        valid_data6 = train_data6[args.n_train:]
        train_data6 = train_data6[:args.n_train]
        valid_data7 = train_data7[args.n_train:]
        train_data7 = train_data7[:args.n_train]

        train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7], axis=1)
        valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4, valid_data5, valid_data6, valid_data7], axis=1)
        test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4, test_data5, test_data6, test_data7], axis=1)

        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        
    elif model == "DHGCN":
        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        del eeg_data
        del event_data

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data, train_label = train_data[indices], train_label[indices]

        valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
        train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

        train_data = train_data.transpose(0, 2, 1)
        valid_data = valid_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        train_data = torch.tensor(train_data, dtype=torch.float)
        valid_data = torch.tensor(valid_data, dtype=torch.float)
        test_data = torch.tensor(test_data, dtype=torch.float)
        train_label = torch.tensor(train_label, dtype=torch.long)
        valid_label = torch.tensor(valid_label, dtype=torch.long)
        test_label = torch.tensor(test_label, dtype=torch.long)

        K_neigs = [1]
        K_neigs_sp = [3]
        is_probH = True
        m_prob = 1.0

        # construct temporal hypergraph G
        G_train = cal_graph(train_data, K_neigs, is_probH=is_probH, m_prob=m_prob)
        G_valid = cal_graph(valid_data, K_neigs, is_probH=is_probH, m_prob=m_prob)
        G_test = cal_graph(test_data, K_neigs, is_probH=is_probH, m_prob=m_prob)

        train_data = train_data.permute(0, 2, 1)
        valid_data = valid_data.permute(0, 2, 1)
        test_data = test_data.permute(0, 2, 1)

        # construct spatial hypergraph G_sp
        G_train_sp = cal_graph(train_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)
        G_valid_sp = cal_graph(valid_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)
        G_test_sp = cal_graph(test_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)

        train_loader = DataLoader(dataset=CustomDatasets_DHGCN(train_data, train_label, G_train, G_train_sp), batch_size=args.batch_size,
                                  drop_last=True)
        valid_loader = DataLoader(dataset=CustomDatasets_DHGCN(valid_data, valid_label, G_valid, G_valid_sp), batch_size=args.batch_size,
                                  drop_last=True)
        test_loader = DataLoader(dataset=CustomDatasets_DHGCN(test_data, test_label, G_test, G_test_sp), batch_size=args.batch_size,
                                 drop_last=True)
        
    elif model == "DBPNet":
        args.image_size = 32
        args.delta_low = 1
        args.delta_high = 3
        args.theta_low = 4
        args.theta_high = 7
        args.alpha_low = 8
        args.alpha_high = 13
        args.beta_low = 14
        args.beta_high = 30
        args.gamma_low = 31
        args.gamma_high = 50
        args.frequency_resolution = args.fs / args.window_length

        args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
        args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
        args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
        args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
        args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
        args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
        args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
        args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
        args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
        args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

        # fft
        train_data0 = to_alpha0(train_data, args)
        test_data0 = to_alpha0(test_data, args)
        train_data1 = to_alpha1(train_data, args)
        test_data1 = to_alpha1(test_data, args)
        train_data2 = to_alpha2(train_data, args)
        test_data2 = to_alpha2(test_data, args)
        train_data3 = to_alpha3(train_data, args)
        test_data3 = to_alpha3(test_data, args)
        train_data4 = to_alpha4(train_data, args)
        test_data4 = to_alpha4(test_data, args)

        train_data0 = gen_images(train_data0, args)
        test_data0 = gen_images(test_data0, args)
        train_data1 = gen_images(train_data1, args)
        test_data1 = gen_images(test_data1, args)
        train_data2 = gen_images(train_data2, args)
        test_data2 = gen_images(test_data2, args)
        train_data3 = gen_images(train_data3, args)
        test_data3 = gen_images(test_data3, args)
        train_data4 = gen_images(train_data4, args)
        test_data4 = gen_images(test_data4, args)

        fre_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)
        fre_test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)
        fre_train_data = np.expand_dims(fre_train_data, axis=-1)
        fre_test_data = np.expand_dims(fre_test_data, axis=-1)

        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)
        seq_train_data = np.expand_dims(train_data, axis=-1)
        seq_test_data = np.expand_dims(test_data, axis=-1)

        del eeg_data
        del event_data

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(fre_train_data.shape[0])
        np.random.shuffle(indices)
        fre_train_data, seq_train_data, train_label = fre_train_data[indices], seq_train_data[indices], train_label[indices]

        fre_valid_data, seq_valid_data, valid_label = fre_train_data[args.n_train:], seq_train_data[args.n_train:], train_label[args.n_train:]
        fre_train_data, seq_train_data, train_label = fre_train_data[:args.n_train], seq_train_data[:args.n_train], train_label[:args.n_train]

        fre_train_data = fre_train_data.transpose(0, 4, 1, 2, 3)
        fre_valid_data = fre_valid_data.transpose(0, 4, 1, 2, 3)
        fre_test_data = fre_test_data.transpose(0, 4, 1, 2, 3)

        seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
        seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
        seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

        train_loader = DataLoader(dataset=CustomDatasets_DBPNet(seq_train_data, fre_train_data, train_label),
                                    batch_size=args.batch_size, drop_last=True)
        valid_loader = DataLoader(dataset=CustomDatasets_DBPNet(seq_valid_data, fre_valid_data, valid_label),
                                    batch_size=args.batch_size, drop_last=True)
        test_loader = DataLoader(dataset=CustomDatasets_DBPNet(seq_test_data, fre_test_data, test_label),
                                    batch_size=args.batch_size, drop_last=True)

    return train_loader, valid_loader, test_loader


def get_KUL_data(model, name="S1", timelen=1, data_document_path="/data1/xyhe/Data/KUL_mat"):
    class CustomDatasets(Dataset):
        # initialization: data and label
        def __init__(self, data, label):
            self.data = torch.Tensor(data)
            self.label = torch.tensor(label, dtype=torch.uint8)

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            return self.data[index], self.label[index]
        
    class CustomDatasets_DHGCN(Dataset):
        def __init__(self, data, labels, G, G_sp):
            self.data = data
            self.labels = labels
            self.G = G
            self.G_sp = G_sp

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            G = self.G[idx]
            sp = self.G_sp[idx]
            return sample, G, sp, label
        
    class CustomDatasets_DBPNet(Dataset):
        # initialization: data and label
        def __init__(self, seq_data, fre_data, event_data):
            self.seq_data = seq_data
            self.fre_data = fre_data
            self.label = event_data

        # get the size of data
        def __len__(self):
            return len(self.label)

        # get the data and label
        def __getitem__(self, index):
            seq_data = torch.Tensor(self.seq_data[index])
            fre_data = torch.Tensor(self.fre_data[index])
            label = torch.Tensor(self.label[index])

            return seq_data, fre_data, label

    def get_data_from_mat(mat_path):
        '''
        discription:load data from mat path and reshape
        param{type}:mat_path: Str
        return{type}: onesub_data
        '''
        mat_eeg_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['trials']
        for i in range(8):
            session_contents = matstruct_contents[0, i]
            direction_contents = session_contents[0, 0]['attended_ear']
            if direction_contents == 'L':
                direction_label = [0]
            else:
                direction_label = [1]
            eeg_contents = session_contents[0, 0]['RawData']
            eeg_contents = eeg_contents[0, 0]['EegData']
            eeg_contents = eeg_contents[1920:48000, :]
            mat_eeg_data.append(eeg_contents)
            mat_event_data.append(direction_label)

        return mat_eeg_data, mat_event_data

    def sliding_window(eeg_datas, labels, args, eeg_channel):
        window_size = args.window_length
        stride = int(window_size * (1 - args.overlap))

        train_eeg = []
        test_eeg = []
        train_label = []
        test_label = []

        for m in range(len(labels)):
            eeg = eeg_datas[m]
            label = labels[m]
            windows = []
            new_label = []
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                window = eeg[i:i + window_size, :]
                windows.append(window)
                new_label.append(label)
            train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
            test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
            train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
            test_label.append(np.array(new_label)[int(len(windows) * 0.9):])
        train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, eeg_channel)
        train_label = np.stack(train_label, axis=0).reshape(-1, 1)
        test_label = np.stack(test_label, axis=0).reshape(-1, 1)

        return train_eeg, test_eeg, train_label, test_label

    def cal_graph(data, K_neigs, is_probH=True, m_prob=1):
        G = []
        for i in tqdm(range(data.shape[0]), disable=True):
            tmp = hgut.construct_H_with_KNN(data[i], K_neigs=K_neigs, is_probH=is_probH, m_prob=m_prob)
            g = hgut.generate_G_from_H(tmp)
            g = torch.tensor(g)
            G.append(g)
        return G

    print("Num GPUs Available: ", torch.cuda.is_available())
    print(name)
    time_len = timelen
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.log_interval = 20
    args.csp_comp = 64
    args.label_col = 0
    args.log_path = "Results/2s"
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    subpath = args.data_document_path + '/' + str(args.name) + '.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)

    event_data = np.array(event_data)
    print(eeg_data.shape)
    eeg_data = np.vstack(eeg_data)
    eeg_data = eeg_data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = np.array(eeg_data)
    print(eeg_data.shape)
    event_data = np.squeeze(event_data)

    train_data, test_data, train_label, test_label = sliding_window(eeg_data, event_data, args, args.csp_comp)

    if model != "FAConformer" and model != "DHGCN" and model != "DBPNet":
        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        del eeg_data
        del event_data

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data, train_label = train_data[indices], train_label[indices]

        valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
        train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

        if model in ["DARNet", "TMSANet", "IFNet"]:
            train_data = np.squeeze(train_data)
            valid_data = np.squeeze(valid_data)
            test_data = np.squeeze(test_data)
        else:
            train_data = np.expand_dims(train_data, axis=1)
            valid_data = np.expand_dims(valid_data, axis=1)
            test_data = np.expand_dims(test_data, axis=1)

        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)

    elif model == "FAConformer":

        args.delta_low = 1
        args.delta_high = 4
        args.theta_low = 4
        args.theta_high = 8
        args.alpha1_low = 8
        args.alpha1_high = 10
        args.alpha2_low = 10
        args.alpha2_high = 13
        args.beta1_low = 13
        args.beta1_high = 20
        args.beta2_low = 20
        args.beta2_high = 30
        args.gamma1_low = 30
        args.gamma1_high = 50
        args.gamma2_low = 50
        args.gamma2_high = 64

        args.frequency_resolution = args.fs / args.window_length

        args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
        args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
        args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
        args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
        args.point2_low = math.ceil(args.alpha1_low / args.frequency_resolution)
        args.point2_high = math.ceil(args.alpha1_high / args.frequency_resolution) + 1
        args.point3_low = math.ceil(args.alpha2_low / args.frequency_resolution)
        args.point3_high = math.ceil(args.alpha2_high / args.frequency_resolution) + 1
        args.point4_low = math.ceil(args.beta1_low / args.frequency_resolution)
        args.point4_high = math.ceil(args.beta1_high / args.frequency_resolution) + 1
        args.point5_low = math.ceil(args.beta2_low / args.frequency_resolution)
        args.point5_high = math.ceil(args.beta2_high / args.frequency_resolution) + 1
        args.point6_low = math.ceil(args.gamma1_low / args.frequency_resolution)
        args.point6_high = math.ceil(args.gamma1_high / args.frequency_resolution) + 1
        args.point7_low = math.ceil(args.gamma2_low / args.frequency_resolution)
        args.point7_high = math.ceil(args.gamma2_high / args.frequency_resolution) + 1

        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)        

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        del eeg_data
        del event_data

        # filter different frequency band
        train_data0 = filter_signal_by_fft(train_data, args.point0_low, args.point0_high, args.window_length)
        test_data0 = filter_signal_by_fft(test_data, args.point0_low, args.point0_high, args.window_length)
        train_data1 = filter_signal_by_fft(train_data, args.point1_low, args.point1_high, args.window_length)
        test_data1 = filter_signal_by_fft(test_data, args.point1_low, args.point1_high, args.window_length)
        train_data2 = filter_signal_by_fft(train_data, args.point2_low, args.point2_high, args.window_length)
        test_data2 = filter_signal_by_fft(test_data, args.point2_low, args.point2_high, args.window_length)
        train_data3 = filter_signal_by_fft(train_data, args.point3_low, args.point3_high, args.window_length)
        test_data3 = filter_signal_by_fft(test_data, args.point3_low, args.point3_high, args.window_length)
        train_data4 = filter_signal_by_fft(train_data, args.point4_low, args.point4_high, args.window_length)
        test_data4 = filter_signal_by_fft(test_data, args.point4_low, args.point4_high, args.window_length)
        train_data5 = filter_signal_by_fft(train_data, args.point5_low, args.point5_high, args.window_length)
        test_data5 = filter_signal_by_fft(test_data, args.point5_low, args.point5_high, args.window_length)
        train_data6 = filter_signal_by_fft(train_data, args.point6_low, args.point6_high, args.window_length)
        test_data6 = filter_signal_by_fft(test_data, args.point6_low, args.point6_high, args.window_length)
        train_data7 = filter_signal_by_fft(train_data, args.point7_low, args.point7_high, args.window_length)
        test_data7 = filter_signal_by_fft(test_data, args.point7_low, args.point7_high, args.window_length)

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        
        train_label = train_label[indices]
        train_data0 = train_data0[indices]
        train_data1 = train_data1[indices]
        train_data2 = train_data2[indices]
        train_data3 = train_data3[indices]
        train_data4 = train_data4[indices]
        train_data5 = train_data5[indices]
        train_data6 = train_data6[indices]
        train_data7 = train_data7[indices]

        valid_label = train_label[args.n_train:]
        train_label = train_label[:args.n_train]
        valid_data0 = train_data0[args.n_train:]
        train_data0 = train_data0[:args.n_train]
        valid_data1 = train_data1[args.n_train:]
        train_data1 = train_data1[:args.n_train]
        valid_data2 = train_data2[args.n_train:]
        train_data2 = train_data2[:args.n_train]
        valid_data3 = train_data3[args.n_train:]
        train_data3 = train_data3[:args.n_train]
        valid_data4 = train_data4[args.n_train:]
        train_data4 = train_data4[:args.n_train]
        valid_data5 = train_data5[args.n_train:]
        train_data5 = train_data5[:args.n_train]
        valid_data6 = train_data6[args.n_train:]
        train_data6 = train_data6[:args.n_train]
        valid_data7 = train_data7[args.n_train:]
        train_data7 = train_data7[:args.n_train]

        train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7], axis=1)
        valid_data = np.stack([valid_data0, valid_data1, valid_data2, valid_data3, valid_data4, valid_data5, valid_data6, valid_data7], axis=1)
        test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4, test_data5, test_data6, test_data7], axis=1)

        train_loader = DataLoader(dataset=CustomDatasets(train_data, train_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        valid_loader = DataLoader(dataset=CustomDatasets(valid_data, valid_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset=CustomDatasets(test_data, test_label),
                                batch_size=args.batch_size, drop_last=True, pin_memory=True)
        
    elif model == "DHGCN":
        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        del eeg_data
        del event_data

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data, train_label = train_data[indices], train_label[indices]

        valid_data, valid_label = train_data[args.n_train:], train_label[args.n_train:]
        train_data, train_label = train_data[:args.n_train], train_label[:args.n_train]

        train_data = train_data.transpose(0, 2, 1)
        valid_data = valid_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        train_data = torch.tensor(train_data, dtype=torch.float)
        valid_data = torch.tensor(valid_data, dtype=torch.float)
        test_data = torch.tensor(test_data, dtype=torch.float)
        train_label = torch.tensor(train_label, dtype=torch.long)
        valid_label = torch.tensor(valid_label, dtype=torch.long)
        test_label = torch.tensor(test_label, dtype=torch.long)

        K_neigs = [1]
        K_neigs_sp = [3]
        is_probH = True
        m_prob = 1.0

        # construct temporal hypergraph G
        G_train = cal_graph(train_data, K_neigs, is_probH=is_probH, m_prob=m_prob)
        G_valid = cal_graph(valid_data, K_neigs, is_probH=is_probH, m_prob=m_prob)
        G_test = cal_graph(test_data, K_neigs, is_probH=is_probH, m_prob=m_prob)

        train_data = train_data.permute(0, 2, 1)
        valid_data = valid_data.permute(0, 2, 1)
        test_data = test_data.permute(0, 2, 1)

        # construct spatial hypergraph G_sp
        G_train_sp = cal_graph(train_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)
        G_valid_sp = cal_graph(valid_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)
        G_test_sp = cal_graph(test_data, K_neigs_sp, is_probH=is_probH, m_prob=m_prob)

        train_loader = DataLoader(dataset=CustomDatasets_DHGCN(train_data, train_label, G_train, G_train_sp), batch_size=args.batch_size,
                                  drop_last=True)
        valid_loader = DataLoader(dataset=CustomDatasets_DHGCN(valid_data, valid_label, G_valid, G_valid_sp), batch_size=args.batch_size,
                                  drop_last=True)
        test_loader = DataLoader(dataset=CustomDatasets_DHGCN(test_data, test_label, G_test, G_test_sp), batch_size=args.batch_size,
                                 drop_last=True)
        
    elif model == "DBPNet":
        args.image_size = 32
        args.delta_low = 1
        args.delta_high = 3
        args.theta_low = 4
        args.theta_high = 7
        args.alpha_low = 8
        args.alpha_high = 13
        args.beta_low = 14
        args.beta_high = 30
        args.gamma_low = 31
        args.gamma_high = 50
        args.frequency_resolution = args.fs / args.window_length

        args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
        args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
        args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
        args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
        args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
        args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
        args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
        args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
        args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
        args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1

        # fft
        train_data0 = to_alpha0(train_data, args)
        test_data0 = to_alpha0(test_data, args)
        train_data1 = to_alpha1(train_data, args)
        test_data1 = to_alpha1(test_data, args)
        train_data2 = to_alpha2(train_data, args)
        test_data2 = to_alpha2(test_data, args)
        train_data3 = to_alpha3(train_data, args)
        test_data3 = to_alpha3(test_data, args)
        train_data4 = to_alpha4(train_data, args)
        test_data4 = to_alpha4(test_data, args)

        train_data0 = gen_images(train_data0, args)
        test_data0 = gen_images(test_data0, args)
        train_data1 = gen_images(train_data1, args)
        test_data1 = gen_images(test_data1, args)
        train_data2 = gen_images(train_data2, args)
        test_data2 = gen_images(test_data2, args)
        train_data3 = gen_images(train_data3, args)
        test_data3 = gen_images(test_data3, args)
        train_data4 = gen_images(train_data4, args)
        test_data4 = gen_images(test_data4, args)

        fre_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)
        fre_test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)
        fre_train_data = np.expand_dims(fre_train_data, axis=-1)
        fre_test_data = np.expand_dims(fre_test_data, axis=-1)

        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)

        csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
                norm_trace=True)
        train_label = np.squeeze(train_label)
        train_data = csp.fit_transform(train_data, train_label)
        test_data = csp.transform(test_data)
        train_label = train_label.reshape(-1,1)

        train_data = train_data.transpose(0, 2, 1)
        test_data = test_data.transpose(0, 2, 1)
        seq_train_data = np.expand_dims(train_data, axis=-1)
        seq_test_data = np.expand_dims(test_data, axis=-1)

        del eeg_data
        del event_data

        # set the number of training, testing and validating data
        args.n_test = len(test_label)
        args.n_valid = args.n_test
        args.n_train = len(train_label) - args.n_test

        indices = np.arange(fre_train_data.shape[0])
        np.random.shuffle(indices)
        fre_train_data, seq_train_data, train_label = fre_train_data[indices], seq_train_data[indices], train_label[indices]

        fre_valid_data, seq_valid_data, valid_label = fre_train_data[args.n_train:], seq_train_data[args.n_train:], train_label[args.n_train:]
        fre_train_data, seq_train_data, train_label = fre_train_data[:args.n_train], seq_train_data[:args.n_train], train_label[:args.n_train]

        fre_train_data = fre_train_data.transpose(0, 4, 1, 2, 3)
        fre_valid_data = fre_valid_data.transpose(0, 4, 1, 2, 3)
        fre_test_data = fre_test_data.transpose(0, 4, 1, 2, 3)

        seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
        seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
        seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

        train_loader = DataLoader(dataset=CustomDatasets_DBPNet(seq_train_data, fre_train_data, train_label),
                                    batch_size=args.batch_size, drop_last=True)
        valid_loader = DataLoader(dataset=CustomDatasets_DBPNet(seq_valid_data, fre_valid_data, valid_label),
                                    batch_size=args.batch_size, drop_last=True)
        test_loader = DataLoader(dataset=CustomDatasets_DBPNet(seq_test_data, fre_test_data, test_label),
                                    batch_size=args.batch_size, drop_last=True)

    return train_loader, valid_loader, test_loader
