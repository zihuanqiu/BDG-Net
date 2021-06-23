import random
import os
import numpy as np
import torch
import argparse
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import dataset
from BDM_Net import BDM_Net

gpu_list = [1]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)


parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--sigma', '-s', type=float, default=5)
parser.add_argument('--encoder_index', '-e', type=int, default=3)

args = parser.parse_args()

def _init_fn(worker_id, seed=42):
    random.seed(seed + worker_id)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def data_loader():
    batch_size = 40
    num_workers = 10

    data_root = './TrainDataset'

    train_list = data_root + '/train.txt'
    val_list = data_root + '/val.txt'

    img_size = 352

    train_trfm = A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.75, 1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    val_trfm = A.Resize(img_size, img_size)

    train_data = dataset.MyDataset(split='train', data_root=data_root, data_list=train_list, transform=train_trfm, sigma=args.sigma)
    val_data = dataset.MyDataset(split='val', data_root=data_root, data_list=val_list, transform=val_trfm, sigma=args.sigma)

    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)

    return train_loader, val_loader


def model_init(encoder_idx=args.encoder_index):
    max_epochs = 100

    model = BDM_Net(nclass=1, max_epoch=max_epochs)
    # path = './logs/default/version_0/checkpoints/BDM-epoch=98-val_mean_dice=0.9050.ckpt'
    path = None
    if path:
        pretrained_dict = torch.load(path, map_location='cpu')['state_dict']

        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

    return model, max_epochs


def train_process(model, train_loader, val_loader, max_epochs):
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_mean_dice',
                                          filename='BDM-{epoch:02d}-{val_mean_dice:.4f}',
                                          save_top_k=5,
                                          mode='max',
                                          save_weights_only=True)

    trainer = Trainer(max_epochs=max_epochs, logger=tb_logger, gpus=[0, ],
                      precision=16, check_val_every_n_epoch=1, benchmark=True,
                      callbacks=[lr_monitor, checkpoint_callback])  # 使用单卡

    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_dataloaders=val_loader)


def main():
    seed_everything(seed=42)
    train_loader, val_loader = data_loader()

    model, max_epochs = model_init()
    train_process(model, train_loader, val_loader, max_epochs)


if __name__ == '__main__':
    main()
