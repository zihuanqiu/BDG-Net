import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from PIL import Image
import torchvision.transforms as transforms
from metric.dice import mean_dice


gpu_list = [1]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str,
                    default='./logs/default/version_0/checkpoints/BDM-epoch=98-val_mean_dice=0.9050.ckpt')

# ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './TestDataset/{}'.format(_data_name)
    save_path = './results/s=5/{}/'.format(_data_name)

    opt = parser.parse_args()
    from BDM_Net import BDM_Net

    model = BDM_Net(nclass=1)
    model.load_state_dict(torch.load(opt.pth_path, map_location='cpu')['state_dict'])
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    mdice = 0

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt = (gt > 128).astype(np.float32)
        image = image.cuda()

        res = model(image)
        res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        mdice += mean_dice(torch.tensor(res), torch.tensor(gt), if_sigmoid=False)

        cv2.imwrite(save_path+name, res*255)

    mdice /= test_loader.size
    print('mean dice in {0}: {1}'.format(_data_name, mdice))

print('done!')
