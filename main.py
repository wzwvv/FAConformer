import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotmap import DotMap

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from models.DARNet import DARNet
from models.DBPNet import DBPNet
from models.DHGCN import DHGCN
from models.EEGNet import EEGNet
from models.ShallowConvNet import ShallowConvNet
from models.IFNetV2 import IFNet
from models.CTNet import CTNet
from models.TMSANet import TMSANet
from models.Conformer import Conformer
from models.MSCFormer import MSCFormer, Parameters
from models.MSVTNet import MSVTNet
from models.DBConformer import DBConformer
from models.FAConformer import FAConformer

from utils.data_loader import *
from utils.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

config = dict()

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initiate(args, train_loader, valid_loader, test_loader, subject, is_first_seed=False):
    if args.model == "DARNet":
        model = DARNet(chn=args.channel)
    elif args.model == "DBPNet":
        model = DBPNet(in_channels=args.channel)
    elif args.model == "DHGCN":
        model = DHGCN(in_ch=args.channel,
                      n_class=args.num_class,
                      time_point=args.time,
                      dropout=0.4)
    elif args.model == "EEGNet":
        model = EEGNet(n_classes=args.num_class,
                       Chans=args.channel,
                       Samples=args.time,
                       kernLenght=int(args.time // 2),
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25,
                       norm_rate=0.5)
    elif args.model == "SCNN":
        model = ShallowConvNet(n_classes=args.num_class,
                               input_ch=args.channel,
                               input_time=args.time,
                               batch_norm=True,
                               batch_norm_alpha=0.1)
    elif args.model == "IFNet":
        model = IFNet(data_name=args.dataset,
                      in_planes=args.channel,
                      out_planes=64,
                      kernel_size=63,
                      radix=1,
                      patch_size=64,
                      time_points=args.time,
                      num_classes=args.num_class)
    elif args.model == "CTNet":
        model = CTNet(heads=2,
                      emb_size=16,
                      depth=6,
                      eeg1_f1=8,
                      eeg1_D=2,
                      eeg1_kernel_size=64,
                      eeg1_pooling_size1=8,
                      eeg1_pooling_size2=8,
                      eeg1_dropout_rate=args.dropout_rate,
                      eeg1_number_channel=args.channel,
                      number_class=args.num_class,
                      flatten_eeg1=16)
    elif args.model == "TMSANet":
        model = TMSANet(in_planes=args.channel,
                        radix=1,
                        time_points=args.time,
                        num_classes=args.num_class)
    elif args.model == "EEGConformer":
        model = Conformer(args,
                          emb_size=40,
                          depth=6,
                          chn=args.channel,
                          n_classes=args.num_class)
    elif args.model == "MSCFormer":
        model = MSCFormer(parameters=args.MSCFormer_params,
                          class_num=args.num_class,
                          chn=args.channel)
    elif args.model == "MSVTNet":
        model = MSVTNet(chn=args.channel,
                        time_sample_num=args.time,
                        class_num=args.num_class,
                        F=[9, 9, 9, 9],
                        C1=[15, 31, 63, 125],
                        C2=15,
                        D=2,
                        P1=8,
                        P2=7,
                        Pc=0.3,
                        nhead=8,
                        ff_ratio=1,
                        Pt=0.5,
                        layers=2,
                        b_preds=False)
    elif args.model == "DBConformer":
        model = DBConformer(data_name=args.dataset,
                            patch_size=128,
                            time_sample_num=args.time,
                            spa_dim=16,
                            emb_size=40,
                            tem_depth=5,
                            chn_depth=5,
                            chn=args.channel,
                            n_classes=2)
    elif args.model == "FAConformer":
        model = FAConformer(data_name=args.dataset,
                            in_planes=args.channel,
                            out_planes_branch=args.output_size_branch,
                            out_planes_total=args.output_size_total,
                            kernel_size=63,
                            radix=1,
                            patch_size=args.patch_size,
                            time_points=args.time,
                            num_classes=2,
                            depth_branch=args.depth_branch,
                            num_heads_branch=args.num_heads_branch,
                            depth_total=args.depth_total,
                            num_heads_total=args.num_heads_total,
                            num_bands=args.num_bands,
                            dim_feedforward=args.dim_feedforward)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    if args.model == "DHGCN":
        optimizer = AdamW(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=0.003 / 10)
        criterion = LabelSmoothing(smoothing=0.1)

        model = model.cuda()
        criterion = criterion.cuda()

        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler}
        
        return train_model_DHGCN(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed)
    elif args.model == "DBPNet":
        optimizer = AdamW(params=model.parameters(), lr=0.003, weight_decay=3e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.003 / 10)
        criterion = nn.CrossEntropyLoss()

        model = model.cuda()
        criterion = criterion.cuda()

        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion,
                    'scheduler': scheduler}

        return train_model_DBPNet(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed)
    elif args.model == "FAConformer":
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)

        model = model.cuda()
        criterion = criterion.cuda()

        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion}

        return train_model_FAConformer(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)

        model = model.cuda()
        criterion = criterion.cuda()

        settings = {'model': model,
                    'optimizer': optimizer,
                    'criterion': criterion}

        return train_model(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed)


def train_model_DBPNet(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    test_batch_times = [] if is_first_seed else None

    def train(model, optimizer, criterion, scheduler):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for i_batch, batch_data in enumerate(train_loader):
            train_seq_data, train_fre_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_seq_data, train_fre_data, train_label = train_seq_data.cuda(), train_fre_data.cuda(), train_label.cuda()
            preds = model(train_seq_data, train_fre_data)
            
            loss = criterion(preds, train_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        scheduler.step()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        batch_size = loader.batch_size
        
        start_time = time.time() if is_first_seed else None
        
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_seq_data, test_fre_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_seq_data, test_fre_data, test_label = test_seq_data.cuda(), test_fre_data.cuda(), test_label.cuda()
                preds = model(test_seq_data, test_fre_data)

                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()
        
        if is_first_seed:
            total_time = time.time() - start_time
            avg_batch_time = total_time / num_batches if num_batches > 0 else 0
            test_batch_times.append(avg_batch_time)

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)

        return avg_loss, avg_acc

    train_start_time = time.time()
    
    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')
    
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 15:
                break

    total_train_time = time.time() - train_start_time
    
    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    avg_test_batch_time = None
    if is_first_seed:
        avg_test_batch_time = np.mean(test_batch_times) if test_batch_times else 0

    return test_loss, test_acc, total_train_time, avg_test_batch_time

def train_model_DHGCN(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    test_batch_times = [] if is_first_seed else None

    def train(model, optimizer, criterion, scheduler):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for i_batch, batch_data in enumerate(train_loader):
            train_data, train_G, train_G_sp, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_G, train_G_sp, train_label = train_data.cuda(), train_G.cuda(), train_G_sp.cuda(), train_label.cuda()
            preds = model(train_data, train_G, train_G_sp)
            
            loss = criterion(preds, train_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        scheduler.step()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        batch_size = loader.batch_size

        start_time = time.time() if is_first_seed else None
        
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_data, test_G, test_G_sp, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_G, test_G_sp, test_label = test_data.cuda(), test_G.cuda(), test_G_sp.cuda(), test_label.cuda()
                preds = model(test_data, test_G, test_G_sp)

                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        if is_first_seed:
            total_time = time.time() - start_time
            avg_batch_time = total_time / num_batches if num_batches > 0 else 0
            test_batch_times.append(avg_batch_time)

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)

        return avg_loss, avg_acc

    train_start_time = time.time()
    
    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')
    
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 50:
                break

    total_train_time = time.time() - train_start_time
    
    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    avg_test_batch_time = None
    if is_first_seed:
        avg_test_batch_time = np.mean(test_batch_times) if test_batch_times else 0

    return test_loss, test_acc, total_train_time, avg_test_batch_time

def train_model(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    test_batch_times = [] if is_first_seed else None

    def train(model, optimizer, criterion):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for i_batch, batch_data in enumerate(train_loader):
            train_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_label = train_data.cuda(), train_label.cuda()
            preds = model(train_data)
            
            loss = criterion(preds, train_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        batch_size = loader.batch_size

        start_time = time.time() if is_first_seed else None
        
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_label = test_data.cuda(), test_label.cuda()
                preds = model(test_data)

                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        if is_first_seed:
            total_time = time.time() - start_time
            avg_batch_time = total_time / num_batches if num_batches > 0 else 0
            test_batch_times.append(avg_batch_time)

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)

        return avg_loss, avg_acc

    train_start_time = time.time()
    
    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')
    
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                break

    total_train_time = time.time() - train_start_time
    
    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    avg_test_batch_time = None
    if is_first_seed:
        avg_test_batch_time = np.mean(test_batch_times) if test_batch_times else 0

    return test_loss, test_acc, total_train_time, avg_test_batch_time


def train_model_FAConformer(settings, args, train_loader, valid_loader, test_loader, subject, is_first_seed):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    test_batch_times = [] if is_first_seed else None

    def train(model, optimizer, criterion):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for i_batch, batch_data in enumerate(train_loader):
            train_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_label = train_data.cuda(), train_label.cuda()

            preds, preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8 = model(train_data)
            loss = criterion(preds, train_label.long())
            loss_b1 = criterion(preds1, train_label.long())
            loss_b2 = criterion(preds2, train_label.long())
            loss_b3 = criterion(preds3, train_label.long())
            loss_b4 = criterion(preds4, train_label.long())
            loss_b5 = criterion(preds5, train_label.long())
            loss_b6 = criterion(preds6, train_label.long())
            loss_b7 = criterion(preds7, train_label.long())
            loss_b8 = criterion(preds8, train_label.long())
            loss_all = loss + args.lamda * (loss_b1 + loss_b2 + loss_b3 + loss_b4 + loss_b5 + loss_b6 + loss_b7 + loss_b8) / 8

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += loss_all.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        batch_size = loader.batch_size

        start_time = time.time() if is_first_seed else None
        
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_label = test_data.cuda(), test_label.cuda()

                preds, preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8 = model(test_data)
                loss = criterion(preds, test_label.long()).item() * batch_size
                loss_b1 = criterion(preds1, test_label.long()).item() * batch_size
                loss_b2 = criterion(preds2, test_label.long()).item() * batch_size
                loss_b3 = criterion(preds3, test_label.long()).item() * batch_size
                loss_b4 = criterion(preds4, test_label.long()).item() * batch_size
                loss_b5 = criterion(preds5, test_label.long()).item() * batch_size
                loss_b6 = criterion(preds6, test_label.long()).item() * batch_size
                loss_b7 = criterion(preds7, test_label.long()).item() * batch_size
                loss_b8 = criterion(preds8, test_label.long()).item() * batch_size
                loss_all = loss + args.lamda * (loss_b1 + loss_b2 + loss_b3 + loss_b4 + loss_b5 + loss_b6 + loss_b7 + loss_b8) / 8
                total_loss += loss_all

                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        if is_first_seed:
            total_time = time.time() - start_time
            avg_batch_time = total_time / num_batches if num_batches > 0 else 0
            test_batch_times.append(avg_batch_time)

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)

        return avg_loss, avg_acc

    train_start_time = time.time()
    
    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')
    
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                break

    total_train_time = time.time() - train_start_time
    
    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    avg_test_batch_time = None
    if is_first_seed:
        avg_test_batch_time = np.mean(test_batch_times) if test_batch_times else 0

    return test_loss, test_acc, total_train_time, avg_test_batch_time


def set_global_seed(seed):
    # Set global random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_experiment(seed, time_len=2, dataset="DTU", is_first_seed=False):
    set_global_seed(seed)
    
    # Define subject list
    if dataset == "DTU":
        subject_list = [f"S{i}" for i in range(1, 19)]
    elif dataset == "KUL":
        subject_list = [f"S{i}" for i in range(1, 17)]
    
    results = []
    train_times = []
    test_batch_times = []

    args = DotMap()
    args.name = None
    args.seed = seed
    args.model = "FAConformer"
    args.max_epoch = 200
    args.num_class = 2
    args.time_len = time_len
    args.channel = 64
    args.hidden = 0.6
    args.num_bands = 8
    args.lamda = 1.0
    args.dim_feedforward = 16
    args.dataset = dataset
    if args.dataset == "DTU":
        args.path = "/data1/xyhe/Data/DTU_mat"
        args.fs = 64
        args.feature_deep_dim = 80
        args.output_size_branch = 64
        args.output_size_total = 32
        args.patch_size = 32
        args.num_heads_branch = 2
        args.depth_branch = 2
        args.num_heads_total = 2
        args.depth_total = 2
    elif args.dataset == "KUL":
        args.path = "/data1/xyhe/Data/KUL_mat"
        args.fs = 128
        args.feature_deep_dim = 440
        args.output_size_branch = 64
        args.output_size_total = 32
        args.patch_size = 32
        args.num_heads_branch = 2
        args.depth_branch = 2
        args.num_heads_total = 2
        args.depth_total = 2
    args.time = math.ceil(args.time_len * args.fs)
    args.dropout_rate = 0.5
    args.MSCFormer_params = Parameters(dropout_rate=args.dropout_rate)
    
    for name in subject_list:
        print(f"\n========== Processing subject {name} (seed {seed}) ==========")
        args.name = name
        if args.dataset == "DTU":
            train_loader, valid_loader, test_loader = get_DTU_data(args.model, name, time_len, args.path)
        elif args.dataset == "KUL":
            train_loader, valid_loader, test_loader = get_KUL_data(args.model, name, time_len, args.path)
        
        # Model Training and Inference
        loss, acc, train_time, test_batch_time = initiate(args, train_loader, valid_loader, test_loader, args.name, is_first_seed)
        acc = acc * 100
        
        train_times.append(train_time)
        if is_first_seed and test_batch_time is not None:
            test_batch_times.append(test_batch_time)
        
        print(f"Sunject {name} (Seed{seed}) Valid loss: {loss}, Accuracy: {acc.item()}, Training Time: {train_time:.2f}s")
        
        results.append({
            "subject": name,
            "acc": acc.item()
        })
    
    result_dir = f"result_{args.model}_{args.hidden}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    results_df = pd.DataFrame(results)
    
    csv_filename = f"{args.dataset}_{args.model}_{time_len}s_seed{seed}_results.csv"
    csv_path = os.path.join(result_dir, csv_filename)
    
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nThe results of the {seed} experiment have been saved to: {csv_path}")
    
    time_stats = {
        "train_times": train_times,
        "avg_train_time": np.mean(train_times),
        "test_batch_times": test_batch_times if is_first_seed else None,
        "avg_test_batch_time": np.mean(test_batch_times) if (is_first_seed and test_batch_times) else None
    }
    
    return args.model, result_dir, time_stats

def merge_results(seed_list, time_len=2, dataset="DTU", model="FAConformer", result_dir="result"):
    merged_df = None
    avg_acc_list = []
    
    for idx, seed in enumerate(seed_list):
        csv_filename = f"{dataset}_{model}_{time_len}s_seed{seed}_results.csv"
        csv_path = os.path.join(result_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The output file for seed {seed} does not exist: {csv_path}")
        
        df = pd.read_csv(csv_path)
        df.rename(columns={"acc": f"acc_seed{seed}"}, inplace=True)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="subject", how="inner")

    acc_columns = [f"acc_seed{seed}" for seed in seed_list]
    merged_df["acc_average"] = merged_df[acc_columns].mean(axis=1)

    merged_df["acc_std"] = merged_df[acc_columns].apply(lambda row: np.std(row), axis=1)

    merged_df["acc_average"] = merged_df["acc_average"].round(4)
    merged_df["acc_std"] = merged_df["acc_std"].round(4)

    avg_row = {"subject": "average"}
    for col in acc_columns:
        avg_row[col] = merged_df[col].mean().round(4)
        avg_acc_list.append(avg_row[col])

    avg_row["acc_average"] = np.mean(avg_acc_list).round(4)
    avg_row["acc_std"] = np.std(avg_acc_list).round(4)
    
    merged_df = pd.concat([merged_df, pd.DataFrame([avg_row])], ignore_index=True)

    total_csv_filename = f"{dataset}_{model}_{time_len}s_total_results.csv"
    total_csv_path = os.path.join(result_dir, total_csv_filename)

    merged_df.to_csv(total_csv_path, index=False, encoding='utf-8')
    print(f"\nThe merging of all seed results is complete. The final file has been saved to: {total_csv_path}")
    print(f"\nFile names after merging: {merged_df.columns.tolist()}")
    print(f"\nPreview of some results: \n{merged_df.head()}")
    
    return total_csv_path

def main(time_len=2, dataset="DTU"):
    cpu_num = 1
    torch.set_num_threads(cpu_num)

    seed_list = list(range(41, 46))
    merge_list = list(range(41, 46))
    all_train_times = []
    first_seed_test_batch_time = None
    
    for idx, seed in enumerate(seed_list):
        is_first_seed = (idx == 0)
        model, result_dir, time_stats = run_experiment(seed, time_len, dataset, is_first_seed)
        
        all_train_times.extend(time_stats["train_times"])
        
        if is_first_seed:
            first_seed_test_batch_time = time_stats["avg_test_batch_time"]
    
    merge_results(merge_list, time_len, dataset, model, result_dir)
    
    total_runs = len(all_train_times)
    avg_train_time = np.mean(all_train_times)
    
    print("\n" + "="*50)
    print("Time statistics:")
    print(f"Total number of training sessions: {total_runs}")
    print(f"Average training time: {avg_train_time:.4f}s")
    if first_seed_test_batch_time is not None:
        print(f"Average time per batch during testing: {first_seed_test_batch_time:.6f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main(time_len=2, dataset="DTU")