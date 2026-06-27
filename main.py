# -*- coding: utf-8 -*-
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import yaml
from models import MInterface
from data import DInterface
from data.utils import load_labels
from utils import str2bool, TBLogger
import sys
import numpy as np
import torch
import pandas as pd
import warnings
import shutil
warnings.filterwarnings("ignore")
import copy
import torch.nn as nn



dir1 = 'synse_resources'
dir2 = 'zsl_splits'

def load_callbacks(args):
    callbacks = []
    # early stop callback
    callbacks.append(plc.EarlyStopping(
        monitor='train-loss',
        mode='min',
        patience=50,
        min_delta=0.001
    ))  
    # checkpoint callback
    callbacks.append(plc.ModelCheckpoint(
        dirpath=args.work_dir,
        monitor='train-loss',  # use valid set's acc
        filename='best',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs = args.save_interval,
    ))



    return callbacks

def get_trainer(args):
    logger = TBLogger(
        save_dir=args.tb_folder,
        name=args.work_dir_name,
        default_hp_metric=False
    )

    args.logger = logger

    return Trainer(
        devices=args.device,
        accelerator='gpu',
        strategy='auto',
        logger=logger,
        max_epochs=args.num_epoch,
        check_val_every_n_epoch=args.eval_interval,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        enable_model_summary=False
    )

def main(args):
    hparams = copy.deepcopy(args)
    """
    pre-process arguments
    """
    hparams.tb_folder = '/'.join((hparams.work_dir).split('/')[:-1]) + '/tensorboard'
    hparams.work_dir_name = (hparams.work_dir).split('/')[-1]
    
    print("Current Experiment Configs: {}".format(hparams))
    """
    init environment
    """
    if hparams.activate_train: # start a new training
        if hparams.resume: # resume if work_dir exists, otherwise create
            if not os.path.exists(hparams.work_dir):
                print("Create working dir {}".format(hparams.work_dir))
                os.mkdir(hparams.work_dir)
                hparams.resume = False
            else:
                # check checkpoint file
                if not os.path.exists('{}/best.ckpt'.format(hparams.work_dir)):
                    hparams.resume = False
        else: # if work_dir exists, create new; otherwise directly create
            if os.path.exists(hparams.work_dir):
                shutil.rmtree(hparams.work_dir)
            print("Create working dir {}".format(hparams.work_dir))
            os.makedirs(hparams.work_dir, exist_ok=True)
    # constant seed
    pl.seed_everything(hparams.seed)
    

    
    """
    Pre-known dataset configs
    """
    hparams.num_class, hparams.emb_dim, \
    hparams.cls_labels, hparams.bj_names,\
    hparams.bp_names, hparams.t_names = load_labels(hparams.root, hparams.dataloader)
    hparams.dataloader_type = hparams.dataloader.split('_')[0]
    """
    data module
    """
    print("Load data module.")
    print(hparams.backbone)
    data_module = DInterface(**vars(hparams))
    data_module.setup()
   
   
    """
    model module
    """
    print("Load model module.")
    model = MInterface(**vars(hparams))
    
    """
    Processor
    """
    trainer = get_trainer(hparams)
    train_loader = data_module.train_dataloader()
    print(f"数据加载器信息:")
    print(f"  数据集大小: {len(train_loader.dataset)}")
    print(f"  批次数量: {len(train_loader)}")
    print(f"  批次大小: {train_loader.batch_size}")


    if hparams.activate_train:
        trainer.fit(
            model,
            data_module.train_dataloader(),
            data_module.test_dataloader(),
            ckpt_path='{}/best.ckpt'.format(hparams.work_dir) if hparams.resume else None
        )

    else:
        
        if hparams.weights is None:
            raise ValueError("Testing requires --weights path/to/checkpoint.ckpt")

        if not os.path.exists(hparams.weights):
            raise FileNotFoundError(f"Checkpoint not found: {hparams.weights}")

        print(f"Load checkpoint for evaluation: {hparams.weights}")

        ckpt = torch.load(
            hparams.weights,
            map_location="cpu",
            weights_only=False
        )

        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False
        )

        print(f"Loaded checkpoint: {hparams.weights}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        trainer.validate(
            model,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=None
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    """
    Basic arguments
    """
    parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
    parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
    parser.add_argument('-bbp', '--backbone_path', default='', help='the work folder for storing results')

    """
    Processor
    """
    parser.add_argument('--gpus', type=str2bool, default=-1, help='use GPUs or not')
    parser.add_argument('--num_epoch', type=int, help='stop training in which epoch', default=50)


    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=50, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--root', help='root repo to load data')
    parser.add_argument('--dataloader', help='class of the dataloader: ntu60')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--few_shot', type=int, default=256, help='training batch size')
    parser.add_argument('-b', '--backbone', default='shift', help='encoder backbone')

    """
    Model
    """
    # configs
    parser.add_argument('--model_name', help='select the training config for a specific model')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_args', default=dict(), help='model configs')
    parser.add_argument('--prompt_args', default=dict(), help='prompt configs')
    parser.add_argument('--adapter_args', default=dict(), help='adapter configs')
    parser.add_argument('--activate_train', type=str2bool, default=False, help='activate training process')
    parser.add_argument('--test_p', default=False, type=str2bool, help='activate hyperparameter testing')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume training')
    # parser.add_argument('--device', default=[5], type=str2bool, help='deviceID')
    parser.add_argument('--device', default=[5], help='deviceID')
    parser.add_argument('--weights', default=None, help='path to checkpoint for evaluation')



    # Reset Some Default Trainer Arguments' Default Values
    p = parser.parse_args(sys.argv[1:])
    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            input_args = yaml.load(f, Loader=yaml.FullLoader)

        # update parser from config file
        key = vars(p).keys()
        for k in input_args.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key

        parser.set_defaults(**input_args) # assign arg values
    args = parser.parse_args() # update args with hand input
    
    
    if not args.test_p:
        """
        run
        """
        main(args)
