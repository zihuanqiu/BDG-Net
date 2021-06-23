import os
import os.path
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from scipy.ndimage.morphology import distance_transform_edt
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from utils.util import overlay


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split()
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class MyDataset(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, sigma=10):
        self.sigma = sigma
        print('sigma is {}'.format(sigma))
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        label = np.float32(label > 128)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        augmented = self.transform(image=image, mask=label)
        image, label = augmented['image'], augmented['mask']

        ibdm = self.distribution_map(label, sigma=self.sigma)

        # if True and self.split=='train':
        #     plt.figure(dpi=400)
        #     plt.subplot(131)
        #     plt.title('image')
        #     plt.imshow(image)
        #     plt.xticks([]), plt.yticks([])  # 去除坐标轴
        #
        #     plt.subplot(132)
        #     plt.title('label')
        #     plt.imshow(label, cmap=plt.cm.gray)
        #     plt.xticks([]), plt.yticks([])  # 去除坐标轴
        #
        #     plt.subplot(133)
        #     plt.title('IBDM')
        #     plt.imshow(ibdm, cmap=plt.cm.jet)
        #     plt.xticks([]), plt.yticks([])  # 去除坐标轴
        #
        #     plt.show()
        #     plt.close()

        return self.as_tensor(image), \
               torch.tensor(label, dtype=torch.float), \
               torch.tensor(ibdm, dtype=torch.float)

    def distribution_map(self, mask, sigma):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 消除标注的问题孤立点

        dist1 = distance_transform_edt(mask)
        dist2 = distance_transform_edt(1-mask)
        dist = dist1 + dist2
        dist = dist - 1

        f = lambda x, sigma: 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-x**2/(2*sigma**2))

        bdm = f(dist, sigma)

        bdm[bdm < 0] = 0

        return bdm * (sigma ** 2)



