import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from tqdm import tqdm
# from torch.utils.data import DataLoader
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ])
    test_trsf = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
         ])
    # common_trsf = []

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ])
    test_trsf = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
         ])

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet100(iData):
    def __init__(self):
        self.train_trsf = None
        self.test_trsf = None
        self.common_trsf = None
        self.class_order = None
        use_path = False
        train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ]
        self.train_trsf = transforms.Compose(train_trsf)
        test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.test_trsf = transforms.Compose(test_trsf)
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dir = "/home/ascll/Dynamic_model/data/ImageNet100/train"
        test_dir = "/home/ascll/Dynamic_model/data/ImageNet100/test"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)


class iFood101(iData):
    def __init__(self):
        self.train_trsf = None
        self.test_trsf = None
        self.common_trsf = None
        self.class_order = None
        train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ]
        self.train_trsf = transforms.Compose(train_trsf)
        test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.test_trsf = transforms.Compose(test_trsf)
        class_order = np.arange(101).tolist()
        self.class_order = class_order

    def download_data(self):

        train_dir = "/home/ascll/Dynamic_model/data/food101/train"
        test_dir = "/home/ascll/Dynamic_model/data/food101/test"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels_imagenet(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels_imagenet(test_dset.imgs)


from PIL import Image
from tqdm import tqdm
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

mode = '1'  # 0 is single process, 1 and 2 are for multi-precess

def get_pil_img(nimg, size):
    with open(nimg[0], "rb") as f:
        img = Image.open(f).convert("RGB")
        if size is not None:
            img = np.array(img.resize((size, size)))
        else:
            img = np.array(img)
    label = nimg[1]
    return img, label


def split_images_labels_imagenet(imgs, size=None):
    if mode == '0':
        t0 = time.time()
        for item in tqdm(imgs):
            images = []
            labels = []
            with open(item[0], "rb") as f:
                img = Image.open(f).convert("RGB")
                if size is not None:
                    img = np.array(img.resize((size, size)))
                else:
                    img = np.array(img)
            images.append(img)
            labels.append(item[1])
        t1 = time.time()
        print("for loop takes %.2f s" % (t1 - t0))
    if mode == '1':
        t2 = time.time()
        pfunc = partial(get_pil_img, size=size)
        pool = Pool(processes=20)
        results = pool.map(pfunc, imgs)
        pool.close()
        pool.join()
        images, labels = zip(*results)
        t3 = time.time()
        print("Pool takes %.2f s" % (t3 - t2))
    if mode == '2':
        t4 = time.time()
        pfunc = partial(get_pil_img, size=size)
        pool = ThreadPool(processes=4)
        results = pool.map(pfunc, imgs)
        pool.close()
        pool.join()
        images, labels = zip(*results)
        t5 = time.time()
        print("ThreadPool takes %.2f s" % (t5 - t4))
    n_labels = np.array(labels)
    print("Finish")
    return images, n_labels
