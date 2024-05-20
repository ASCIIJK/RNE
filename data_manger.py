import logging
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from data import iCIFAR100
from torch.utils.data import DataLoader
import random
import copy
from tqdm import tqdm
from data import iCIFAR100, iImageNet100, iCIFAR10, iFood101


def get_datasets(datasets):
    if datasets == "cifar100":
        return iCIFAR100()
    elif datasets == "cifar10":
        return iCIFAR10()
    elif datasets == "imagenet100":
        return iImageNet100()
    elif datasets == "ifood101":
        return iFood101()

class FeatureSet(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = features
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return idx, feature, label


class Mydataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        # self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(self.images[idx], torch.Tensor):
            image = self.images[idx]
        else:
            image = self.trsf(Image.fromarray(np.uint8(self.images[idx])))
        label = self.labels[idx]
        return idx, image, label


class DataManger(Dataset):
    def __init__(self, dataset, shuffle, init_class, increment, device):
        self.device = device
        self.init_class = init_class
        self.increment = increment
        self.task_size = None
        dataset = get_datasets(dataset)
        if shuffle:
            self.class_order = list(dataset.class_order)
            random.shuffle(self.class_order)
        else:
            self.class_order = dataset.class_order
        dataset.download_data()  # 载入数据
        self.setup_data()
        self._train_data, self._train_targets = dataset.train_data, dataset.train_targets
        self._test_data, self._test_targets = dataset.test_data, dataset.test_targets
        self.train_trsf = dataset.train_trsf
        self.test_trsf = dataset.test_trsf
        self.train_data_memory = []
        self.train_targets_memory = []
        self.memory_list = None

    def setup_data(self):
        self.task_size = (len(self.class_order) - self.init_class) // self.increment

    def get_dataset(self, source, class_list, appendent, num=None):
        # 获取此刻训练数据或者测试数据
        if source == "train":
            x, y = self._train_data, self._train_targets
            trsf = self.train_trsf
        elif source == "test":
            x, y = self._test_data, self._test_targets
            trsf = self.test_trsf
        else:
            raise ValueError("Unknown data source {}.".format(source))
        if num is None:
            data, targets = self._select(x, y, class_list)
        else:
            data, targets = [], []
            for item in class_list:
                data_raw, targets_raw = self._select(x, y, [item], num)
                data.extend(data_raw), targets.extend(targets_raw)
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            for i in range(len(self.memory_list)):
                data.extend(appendent_data[i])
                targets.extend(appendent_targets[i])
            print('memory_list is {}'.format(self.memory_list))
        return Mydataset(data, targets, trsf)

    def get_dataset_pseudo(self, source, class_list, appendent, num=None):
        # 获取此刻训练数据或者测试数据
        if source == "train":
            x, y = self._train_data, self._train_targets
            trsf = self.train_trsf
        elif source == "test":
            x, y = self._test_data, self._test_targets
            trsf = self.test_trsf
        else:
            raise ValueError("Unknown data source {}.".format(source))
        if num is None:
            data, targets = self._select(x, y, class_list)
        else:
            data, targets = self._select(x, y, class_list, num)
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.extend(appendent_data)
            targets.extend(appendent_targets)
            print('memory_list is {}'.format(self.memory_list))
        return Mydataset(data, targets, trsf)

    def _select(self, x, y, class_list, num=None):
        # 选择当前时刻的样本
        data, targets = [], []
        cur_num = 0
        index_num = list(range(len(y)))
        random.shuffle(index_num)
        if len(class_list) == 0:
            return data, targets
        for i in range(len(y)):
            if y[index_num[i]] in class_list:
                data.append(x[index_num[i]])
                targets.append(y[index_num[i]])
                cur_num += 1
                if num is not None and cur_num >= num:
                    break
        return data, targets

    def rebuild_memory(self, model, new_class_list, num):
        # num为单个类别旧类保存的数量
        if self.memory_list is None:
            for i in new_class_list:
                data, targets = self._select_memory(i, num, model, mode='herding')
                self.train_data_memory.append(data)
                self.train_targets_memory.append(targets)
                torch.cuda.empty_cache()
            self.memory_list = new_class_list  # 更新旧类表
        else:
            # 先删除部分旧类，使得数量满足新的num
            for i in range(len(self.memory_list)):
                data, targets = self._select_memory(self.memory_list[i], num, model, mode='herding')
                self.train_data_memory[i] = data
                self.train_targets_memory[i] = targets
            # 添加新的类
            for i in new_class_list:
                data, targets = self._select_memory(i, num, model, mode='herding')
                self.train_data_memory.append(data)
                self.train_targets_memory.append(targets)
            self.memory_list += new_class_list  # 更新旧类表

    def generate_pseudo_img(self, model, class_list, per_sample_num):
        all_samples_num = len(class_list) * per_sample_num
        trsf = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        num_samples = np.zeros(len(class_list))
        data, targets = [], []
        # model.eval()
        model.train()
        model.to(self.device)
        for i in tqdm(range(all_samples_num // 100)):
            img = torch.rand(100, 3, 32, 32, dtype=torch.float).to(self.device)
            img = trsf(img)
            with torch.no_grad():
                logits = model(img)["logits"]
            _, preds = torch.max(logits, dim=1)
            # img = img.squeeze(0)
            for j in range(100):
                data.append(img[j].cpu())
                targets.append(preds[j].cpu())
                num_samples[preds[j]] += 1
        self.memory_list = class_list
        self.train_data_memory = data
        self.train_targets_memory = targets
        print(num_samples)

    def _select_memory(self, class_name, num, model, mode):
        # 先找出所有新类样本
        data, targets = [], []
        x, y = self._train_data, self._train_targets
        for i in range(len(y)):
            if y[i] == class_name:
                data.append(x[i])
                targets.append(y[i])
        if mode == "herding":
            # 将数据打包成迭代器
            train_loader = DataLoader(Mydataset(data, targets, self.train_trsf), batch_size=128, shuffle=False, num_workers=0)
            model.eval()
            model.to(self.device)
            for i, (_, inputs, _) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    if i == 0:
                        features = model.extract_vector(inputs)
                    else:
                        features = torch.cat((features, model.extract_vector(inputs)))  # 特征向量拼接
            index = self.herding_rule(features, num)
        elif mode == "random":
            index = self.random_rule(data, targets, num)
        else:
            raise ValueError("Unknown rule mode {}.".format(mode))
        data_i, targets_i = [], []
        for i in range(len(index)):
            data_i.append(data[index[i]])
            targets_i.append(targets[index[i]])
        # del features
        return data_i, targets_i

    def get_memory(self):
        if len(self.train_targets_memory) == 0:
            return None
        else:
            return self.train_data_memory, self.train_targets_memory

    @staticmethod
    def random_rule(data, targets, num):
        index = random.sample(range(0, len(targets)), num)
        return index

    @staticmethod
    def herding_rule(features, nb_examplars):
        features = features.cpu()
        D = copy.deepcopy(features.T)
        D = D.numpy()
        D = D / (np.linalg.norm(D, axis=0) + 1e-8)
        mu = np.mean(D, axis=1)
        herding_matrix = np.zeros((features.shape[0],))

        w_t = mu
        iter_herding, iter_herding_eff = 0, 0

        while not (
                np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
        ) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if herding_matrix[ind_max] == 0:
                herding_matrix[ind_max] = 1 + iter_herding
                iter_herding += 1

            w_t = w_t + mu - D[:, ind_max]

        herding_matrix[np.where(herding_matrix == 0)[0]] = 10000
        index = herding_matrix.argsort()[:nb_examplars]
        # print(len(index))

        return index


