import argparse
import configparser
import os
import sys
import time
import pickle

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn as nn
import configparser
from model.main_model import Model as Network_Predict
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, MSE_torch, huber_loss
from lib.Params_pretrain import parse_args
from lib.Params_predictor import get_predictor_params
from trainer import Trainer
from new_trainer import New_Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss


def scaler_huber_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae, mae_loss = huber_loss(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss


def get_data_and_model(args, dow=False, weather=False, single=False):
    if args.model == 'PIGWN':
        if args.dataset == 'usa':
            print("2222")
            train_loader = ['../data/usa/train_dataloader_1.pth', '../data/usa/train_dataloader_2.pth', '../data/usa/train_dataloader_3.pth']
            val_loader = torch.load('../data/usa/val_dataloader.pth')
            test_loader = torch.load('../data/usa/test_dataloader.pth')
            with open('../data/usa/scaler_data.pkl', 'rb') as f:
                scaler_data = pickle.load(f)
            with open('../data/usa/scaler_data_label.pkl', 'rb') as f:
                scaler_data_label = pickle.load(f)
            with open('../data/usa/scaler_phy_label.pkl', 'rb') as f:
                scaler_phy = pickle.load(f) 
        else:
            train_loader, val_loader, test_loader, scaler_data, scaler_data_label, scaler_day, scaler_week, scaler_holiday, scaler_phy = get_dataloader(args,
                                                                                                                     normalizer=args.normalizer,
                                                                                                                     tod=args.tod,
                                                                                                                     dow=False,
                                                                                                                     weather=False,
                                                                                                                     single=False)
    else:
        train_loader, val_loader, test_loader, scaler_data, scaler_data_label, scaler_day, scaler_week, scaler_holiday = get_dataloader(args,
                                                                                                                 normalizer=args.normalizer,
                                                                                                                 tod=args.tod,
                                                                                                                 dow=False,
                                                                                                                 weather=False,
                                                                                                                 single=False)
    # model
    model = Network_Predict(args, args_predictor)
    model = model.to(args.device)


    if args.xavier:
        for p in model.parameters():
            if p.requires_grad == True:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                 weight_decay=0, amsgrad=False)

    # loss
    if args.loss_func == 'mask_mae':
        loss = scaler_mae_loss(scaler_data_label, mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
    elif args.loss_func == 'mask_huber':
        loss = scaler_mae_loss(scaler_data_label, mask_value=args.mape_thresh)
        print('============================scaler_mae_loss')
        # print(args.model, Mode)
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError
    
    if args.model == 'PIGWN':
        loss_phy = scaler_mae_loss(scaler_phy, mask_value=args.mape_thresh)
        print('============================scaler_mae_loss_loss_phy')
    loss_kl = nn.KLDivLoss(reduction='sum').to(args.device)

    # lr decay
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    if args.model == 'PIGWN':
        return model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data, scaler_data_label, lr_scheduler, scaler_phy, loss_phy
    else:
        return model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data, scaler_data_label, lr_scheduler


if __name__ == "__main__":
    args = parse_args(device)
    args_predictor = get_predictor_params(args)
    init_seed(args.seed, args.seed_mode)

    # TODO: args需要修改
    # config log path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(current_dir, args.model, args.dataset+'_1')
    Mkdir(log_dir)
    args.log_dir = log_dir
    print("111")

    if args.model == 'PIGWN':
        if args.dataset == 'usa':
            model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data, scaler_data_label, lr_scheduler, scaler_phy, loss_phy = get_data_and_model(args)
        else:
            model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data, scaler_data_label, lr_scheduler, scaler_phy, loss_phy = get_data_and_model(args)
    else:
        model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data, scaler_data_label, lr_scheduler = get_data_and_model(args)
    if args.model == 'new':
        trainer = New_Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data_label,
                      args, lr_scheduler=lr_scheduler)
    elif args.model == 'PIGWN':
        trainer = New_Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data_label,
                      args, scaler_phy, loss_phy,lr_scheduler=lr_scheduler)
    else:
        trainer = Trainer(model, loss, loss_kl, optimizer, train_loader, val_loader, test_loader, scaler_data_label,
                      args, lr_scheduler=lr_scheduler)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        model_path = './{}/china.pth'.format(args.model)  # 修改为模型的路径
        model.load_state_dict(torch.load(model_path))
        print("Load saved model")
        trainer.test(model, trainer.args, test_loader, trainer.scaler, trainer.logger)

    else:
        raise ValueError


