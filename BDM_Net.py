import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from metric.dice import mean_dice
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.heads import SegmentationHead
from utils.util import initialize_weights
import cv2, os
from utils.util import overlay


def seg_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def bdm_loss(pred, target, thresh=0.002, min_ratio=0.1):

    pred = pred.view(-1)
    target = target.view(-1)

    loss = F.mse_loss(pred, target, reduction='none')
    _, index = loss.sort()  # 从小到大排序

    threshold_index = index[-round(min_ratio * len(index))]  # 找到min_kept数量的hardexample的阈值

    if loss[threshold_index] < thresh:  # 为了保证参与loss的比例不少于min_ratio
        thresh = loss[threshold_index].item()

    loss[loss < thresh] = 0

    loss = loss.mean()

    return loss


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Agg(nn.Module):
    def __init__(self, channel=64):
        super(Agg, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.h2h_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.h2l_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )

        # stage 2
        self.h2h_2 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(channel, channel, 3, 1, 1)
        )

    def forward(self, h, l):
        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = h2h + l2h
        l = l2l + h2l

        # stage 2
        h2h = self.h2h_2(h)
        l2h = self.l2h_2(self.l2h_up(l))
        out = h2h + l2h
        return out


class BDMM(nn.Module):
    def __init__(self, inplanes: list, midplanes=32, upsample=8):
        super(BDMM, self).__init__()
        assert len(inplanes) == 3

        self.rfb1 = RFB_modified(inplanes[0], midplanes)
        self.rfb2 = RFB_modified(inplanes[1], midplanes)
        self.rfb3 = RFB_modified(inplanes[2], midplanes)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.agg1 = Agg(midplanes)
        self.agg2 = Agg(midplanes)

        self.conv_out = nn.Sequential(
            Conv2dReLU(midplanes, 1, 3, padding=1),
            nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True),
        )

    def forward(self, x1, x2, x3):
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)

        x2 = self.agg1(x2, x3)
        x1 = self.agg2(x1, x2)

        out = self.conv_out(x1)

        return out


class BDGD_A(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.l2l_0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        # stage 2
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # stage 3
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, dist):
        dist_l = F.interpolate(dist, x.size()[2:], mode='bilinear')

        # stage 1
        l = self.l2l_0(x)

        # stage 2
        l2l = self.l2l_1(l*dist_l)
        l2h = self.l2h_1(self.l2h_up(l+l2l))

        # stage 3
        out = self.l2h_2(self.l2h_up(l)+l2h)
        return out


class BDGD_B(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 1
        self.h2h_0 = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, 1, 1, groups=skip_channels),
            nn.Conv2d(skip_channels, out_channels, 1),
        )

        self.l2l_0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        # stage 2
        self.h2h_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.h2l_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2h_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2l_1 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # stage 3
        self.h2h_2 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.l2h_2 = nn.Sequential(
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, skip, dist):
        dist_h = F.interpolate(dist, skip.size()[2:], mode='bilinear')
        dist_l = F.interpolate(dist, x.size()[2:], mode='bilinear')

        # stage 1
        h_in = self.h2h_0(skip)
        l_in = self.l2l_0(x)

        # stage 2
        h2h = self.h2h_1(h_in * dist_h)
        l2h = self.l2h_1(self.l2h_up(l_in))

        l2l = self.l2l_1(l_in * dist_l)
        h2l = self.h2l_1(self.h2l_pool(h_in))

        h = h2h + l2h
        l = l2l + h2l

        # stage 3
        h2h = self.h2h_2(h)
        l2h = self.l2h_2(self.l2h_up(l)) + l2h
        out = h2h + l2h
        return out


class BDM_Net(pl.LightningModule):
    def __init__(self, nclass=1, max_epoch=None):
        super().__init__()
        self.encoder = get_encoder('timm-efficientnet-b5', weights='noisy-student')
        self.agg = BDMM(self.encoder.out_channels[-3:], 32, upsample=8)

        self.dec1 = BDGD_A(64, 32)
        self.dec2 = BDGD_B(128, self.encoder.out_channels[-4], 64)
        self.dec3 = BDGD_B(256, self.encoder.out_channels[-3], 128)
        self.dec4 = BDGD_B(self.encoder.out_channels[-1], self.encoder.out_channels[-2], 256)

        self.seg_head = SegmentationHead(32, nclass, upsampling=2)

        self.learning_rate = 1e-4
        self.max_epoch = max_epoch

        initialize_weights(self.dec1)
        initialize_weights(self.dec2)
        initialize_weights(self.dec3)
        initialize_weights(self.dec4)

        initialize_weights(self.seg_head)
        initialize_weights(self.agg)

        self.num = 0

    def forward(self, x):
        x = self.encoder(x)
        bdm = self.agg(x[-3], x[-2], x[-1])
        c4 = self.dec4(x[-1], x[-2], bdm)
        c3 = self.dec3(c4, x[-3], bdm)
        c2 = self.dec2(c3, x[-4], bdm)
        c1 = self.dec1(c2, bdm)
        seg = self.seg_head(c1)

        return seg, bdm

    def training_step(self, batch, batch_idx):
        x, y, ibdm = batch
        y_hat, bdm = self(x)
        train_loss_seg = seg_loss(y_hat, y.unsqueeze(1))
        train_loss_bdm = bdm_loss(bdm.squeeze(1), ibdm)

        train_mean_dice = mean_dice(y_hat, y)

        self.log('train_loss_seg', train_loss_seg, on_epoch=True)
        self.log('train_loss_bdm', train_loss_bdm, on_epoch=True)

        self.log('train_mean_dice', train_mean_dice, on_epoch=True)

        return train_loss_seg + train_loss_bdm

    def validation_step(self, batch, batch_idx):
        x, y, ibdm = batch
        y_hat, bdm = self(x)
        val_loss_seg = seg_loss(y_hat, y.unsqueeze(1))
        val_loss_bdm = bdm_loss(bdm.squeeze(1), ibdm)

        val_mean_dice = mean_dice(y_hat, y)

        self.log('val_loss_seg', val_loss_seg)
        self.log('val_loss_bdm', val_loss_bdm)

        self.log('val_mean_dice', val_mean_dice)

        return val_loss_seg + val_loss_bdm

    def test_step(self, batch, batch_idx):
        x, y, ibdm = batch
        y_hat, bdm = self(x)
        test_loss_seg = seg_loss(y_hat, y.unsqueeze(1))
        test_mean_dice = mean_dice(y_hat, y)

        for i in range(y_hat.size()[0]):
            TH = 0.5
            img = x[i, :, :, :]
            seg_gt = y[i, :, :]
            seg = y_hat[i, 0, :, :]
            dh_gt = ibdm[i, :, :]
            dh = bdm[i, 0, :, :]
            seg = seg.sigmoid()

            plt.figure()
            plt.subplot(231)
            plt.title('image')
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).type_as(img)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).type_as(img)
            img *= std.unsqueeze(-1).unsqueeze(-1)
            img += mean.unsqueeze(-1).unsqueeze(-1)

            img = img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            seg = seg.cpu().numpy().astype(np.float32)
            dh = dh.cpu().numpy().astype(np.float32)
            seg_gt = seg_gt.cpu().numpy().astype(np.float32)
            dh_gt = dh_gt.cpu().numpy().astype(np.float32)

            plt.imshow(img)
            plt.xticks([]), plt.yticks([])  # 去除坐标轴

            plt.subplot(232)
            plt.title('seg')
            plt.imshow(seg.astype(np.float32), cmap=plt.cm.gray)
            plt.xticks([]), plt.yticks([])  # 去除坐标轴

            plt.subplot(233)
            plt.title('ground truth')
            plt.imshow(seg_gt, cmap=plt.cm.gray)
            plt.xticks([]), plt.yticks([])  # 去除坐标轴

            plt.subplot(234)
            plt.title('overlay')
            plt.imshow(overlay(img.transpose(2, 0, 1), (seg > TH)))
            plt.xticks([]), plt.yticks([])  # 去除坐标轴

            plt.subplot(235)
            plt.title('bdm')
            plt.imshow(dh, cmap=plt.cm.jet)
            plt.xticks([]), plt.yticks([])  # 去除坐标轴

            plt.subplot(236)
            plt.title('ideal bdm')
            plt.imshow(dh_gt, cmap=plt.cm.jet)
            plt.xticks([]), plt.yticks([])  # 去除坐标轴

            save_path = './save'
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path+'/{}.png'.format(self.num), dpi=400)
            # plt.show()

            self.num += 1
            plt.close()

        self.log('test_mean_dice', test_mean_dice)

        return test_loss_seg

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_learning_rate = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_learning_rate)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    from utils.util import CalParams
    model = BDM_Net(nclass=1)
    CalParams(model, torch.rand(1, 3, 352, 352))
