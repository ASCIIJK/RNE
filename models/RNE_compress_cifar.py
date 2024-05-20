import math
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
from torch.utils.data import WeightedRandomSampler
from models.BaseNet import RNECompressBaseNet
from data_manger import FeatureSet
from toolkit import tensor2numpy, count_parameters
import os

EPSILON = 1e-8


class RNE(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.num_workers = args["num_workers"]
        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]
        self.batch_size = args["batch_size"]
        self.memory_size = args["memory_size"]
        self.memory_per_class = args["memory_per_class"]
        self.fixed = args["fixed_memory"]
        self.epochs = args["epochs"]
        self.weight_decay = args["weight_decay"]
        self.lr = args["lr"]
        self.lr_fc = args["lr_fc"]
        self.batch_size_fc = args["batch_size_fc"]
        self.epochs_fc = args["epochs_fc"]
        self._network = RNECompressBaseNet(args)

        self.init_class = args["init_cls"]
        self.increment = args["increment"]
        self.class_list = None
        self.DataManger = None

        self._old_network = None
        self.device = args["device"]

        self.known_class = 0
        self.cur_task = -1
        self.task_acc = []
        self.TaskConfusionMatrix = []
        self.ConfusionMatrix = []

        self.test_loader = None
        self.accs = {"all": [], "old": [], "new": []}
        self.logger = loger

        self.feature_mean = None
        self.feature_var = None

        self.normal_net = None

    def after_train(self):
        self._old_network = copy.deepcopy(self._network)
        self._old_network.freeze()  # 冻结旧模型

        # 重构回放数据集
        if self.cur_task == 0:
            class_list = self.class_list[0:self.init_class]
            self.known_class = self.init_class
            if self.fixed is True:
                memory_per_class = self.memory_per_class
            else:
                memory_per_class = self.memory_size // self.known_class
            self.DataManger.rebuild_memory(self._network, class_list, memory_per_class)
        else:
            class_list = self.class_list[self.known_class:self.known_class+self.increment]
            self.known_class += self.increment
            if self.fixed is True:
                memory_per_class = self.memory_per_class
            else:
                memory_per_class = self.memory_size // self.known_class
            self.DataManger.rebuild_memory(self._network, class_list, memory_per_class)

        self.logger.info("save the {} sample per class".format(memory_per_class))
        self.save_check_point()

        acc, _, _ = self.eval_task(self.test_loader)
        self.accs["all"].append(round(acc, 2))
        self.logger.info("total acc: {}".format(self.accs["all"]))

    def save_check_point(self):
        args = self.args
        save_path = "logs/{}/{}/{}/{}/task{}.pkl".format(
            args["model_name"],
            args["dataset"],
            self.init_class,
            self.increment,
            self.cur_task
        )
        torch.save(self._network.state_dict(), save_path)
        self.logger.info('task{}.pkl has been saved in {}'.format(self.cur_task, save_path))

    def init_train(self, datamanger):
        self.DataManger = datamanger
        self.class_list = datamanger.class_order
        self.logger.info("init_class: {}, increment: {}".format(self.init_class, self.increment))

        self.cur_task += 1
        init_class_list = self.class_list[0:self.init_class]

        self.logger.info("training classes is {}".format(init_class_list))

        self._network.update_fc(self.init_class)

        self.logger.info("All params: {}".format(count_parameters(self._network)))
        self.logger.info("Trainable params: {}".format(count_parameters(self._network, True)))

        train_dataset = self.DataManger.get_dataset(
            source="train", class_list=init_class_list, appendent=None
        )
        train_dataset.labels = self.targets_map(train_dataset.labels)

        test_dataset = self.DataManger.get_dataset(
            source="test", class_list=init_class_list, appendent=None
        )
        self.test_dataset = test_dataset
        test_dataset.labels = self.targets_map(test_dataset.labels)
        # 构建迭代器
        train_loader = DataLoader(
            train_dataset, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.test_loader = test_loader

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            momentum=0.9,
            lr=self.init_lr,
            weight_decay=self.init_weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.init_epochs
        )
        self._run(train_loader, test_loader, optimizer, scheduler)

        feature_mean, feature_var, _, _, _ = self.compute_feature_mean(train_loader)
        self.feature_mean = feature_mean
        self.feature_var = feature_var

    def increment_train(self):
        self.cur_task += 1
        increment_class_list = self.class_list[self.known_class:self.known_class+self.increment]    # 增量训练名单
        self.logger.info("training classes is {}".format(increment_class_list))
        self._network.update_fc(self.known_class+self.increment)

        for i in range(self.cur_task):
            for p in self._network.convnets[i].parameters():
                p.requires_grad = False
        for p in self._network.backbone.parameters():
            p.requires_grad = False
        for p in self._network.backbone.conv_layer1.parameters():
            p.requires_grad = True
        for p in self._network.backbone.conv_layer2.parameters():
            p.requires_grad = True
        for p in self._network.backbone.conv_layer3.parameters():
            p.requires_grad = True
        for p in self._network.backbone.conv_layer4.parameters():
            p.requires_grad = True
        for p in self._network.convnets[-1].parameters():
            p.requires_grad = True

        self.logger.info("All params: {}".format(count_parameters(self._network)))
        self.logger.info("Trainable params: {}".format(count_parameters(self._network, True)))
        # 读取训练数据与测试数据(包含旧类样本训练主干)
        train_dataset = self.DataManger.get_dataset(
            source="train", class_list=increment_class_list, appendent=self.DataManger.get_memory()
        )
        train_dataset.labels = self.targets_map(train_dataset.labels)  # 将标签映射为可训练顺序
        test_dataset = self.DataManger.get_dataset(
            source="test", class_list=self.class_list[0:self.known_class+self.increment], appendent=None
        )
        test_dataset.labels = self.targets_map(test_dataset.labels)  # 将标签映射为可训练顺序
        # 构建迭代器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        self.test_loader = test_loader
        # 配置优化器
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            momentum=0.9,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.epochs
        )

        self.logger.info("new model traing...")
        self._run(train_loader, test_loader, optimizer, scheduler)

        self.copy_acc = 0.
        self.copy_net = copy.deepcopy(self._network)

        feature_mean, feature_var, feature_new, labels_new, sum_per_classes = self.compute_feature_mean(train_loader)
        self.feature_mean = np.concatenate(
            (self.feature_mean, feature_mean[:self.known_class, -self._network.convnets[-1].out_dim:]), axis=1)
        self.feature_mean = np.concatenate((self.feature_mean, feature_mean[self.known_class:, :]), axis=0)
        self.feature_var = np.concatenate(
            (self.feature_var, feature_var[:self.known_class, -self._network.convnets[-1].out_dim:]), axis=1)
        self.feature_var = np.concatenate((self.feature_var, feature_var[self.known_class:, :]), axis=0)

        features_dataset = self.generate_fake_feature(feature_new, labels_new, sum_per_classes)

        self.logger.info("features set has {} features".format(len(features_dataset)))

        correct_loader = DataLoader(features_dataset, batch_size=self.batch_size_fc, shuffle=True,
                                    num_workers=self.num_workers, pin_memory=True, drop_last=True)

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            momentum=0.,
            lr=self.lr_fc,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[3, 6], gamma=0.1
        )

        if self.fixed:
            test_dataset_fc = self.DataManger.get_dataset(
                source="train", class_list=increment_class_list, appendent=self.DataManger.get_memory(),
                num=self.memory_per_class
            )
        else:
            memory_per_class = self.memory_size // self.known_class
            test_dataset_fc = self.DataManger.get_dataset(
                source="train", class_list=increment_class_list, appendent=self.DataManger.get_memory(),
                num=memory_per_class
            )
        self.logger.info("Validation set has {} samples".format(len(test_dataset_fc)))
        test_dataset_fc.labels = self.targets_map(test_dataset_fc.labels)
        test_loader_fc = DataLoader(test_dataset_fc, batch_size=16, shuffle=False, num_workers=4)
        self._run_fc(correct_loader, test_loader_fc, optimizer, scheduler)

    def _train_mode(self):
        self._network.train()
        for i in range(self.cur_task):
            self._network.convnets[i].eval()
        self._network.convnets[-1].train()

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        self._network.to(self.device)
        if self.cur_task == 0:
            prog_bar = tqdm(range(self.init_epochs))
        else:
            prog_bar = tqdm(range(self.epochs))

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            losses_old = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            self._train_mode()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self._network(inputs)

                old_classes = np.array(range(self.known_class)).tolist()
                mask_old = [i in old_classes for i in np.array(targets.cpu())]
                mask_new = [i not in old_classes for i in np.array(targets.cpu())]

                if self.cur_task == 0:
                    logits = outputs["logits"]
                    loss = F.cross_entropy(logits, targets.long())
                else:
                    logits, aux_logits = outputs["logits"], outputs["aux_logits"]

                    if mask_old.count(False) > 0:
                        beta = 0.1 + 0.9 * epoch / self.epochs
                        gamma = mask_old.count(True) / mask_old.count(False) / self.known_class * self.increment * beta
                    else:
                        gamma = 0.001
                    loss = gamma * F.cross_entropy(logits[mask_new, ], targets[mask_new].long())
                    if mask_old.count(True) > 0:
                        loss += F.cross_entropy(logits[mask_old, ], targets[mask_old].long())

                    aux_targets = targets.clone()
                    aux_targets = torch.where(
                        aux_targets - self.known_class + 1 > 0,
                        aux_targets - self.known_class + 1,
                        aux_targets - aux_targets,
                    )

                    loss_aux = F.cross_entropy(aux_logits, aux_targets.long())
                    loss += loss_aux
                    losses_aux += loss_aux.item()

                if self._old_network is not None:
                    with torch.no_grad():
                        old_logits = self._old_network(inputs)["logits"][mask_old, :]
                    loss_old = _KD_loss(
                        logits[mask_old, ],
                        torch.cat(
                             [old_logits, torch.zeros(logits[mask_old, :].shape[0], self.increment).to(self.device)]
                             , dim=1),
                        2.
                        )
                    if not math.isnan(loss_old):
                        loss += loss_old
                        losses_old += loss_old.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 10 == 0:
                test_acc = self.compute_test_accuracy(test_loader, self._network)
            else:
                test_acc = None

            if self.cur_task == 0:
                if test_acc is not None:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self.cur_task,
                        epoch + 1,
                        self.init_epochs,
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self.cur_task,
                        epoch + 1,
                        self.init_epochs,
                        losses / len(train_loader),
                        train_acc,
                    )
            else:
                if test_acc is not None:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, old_loss {:.3f}, aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self.cur_task,
                        epoch + 1,
                        self.epochs,
                        losses / len(train_loader),
                        losses_old / len(train_loader),
                        losses_aux / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, old_loss {:.3f}, aux {:.3f}, Train_accy {:.2f}".format(
                        self.cur_task,
                        epoch + 1,
                        self.epochs,
                        losses / len(train_loader),
                        losses_old / len(train_loader),
                        losses_aux / len(train_loader),
                        train_acc,
                    )
            self.logger.info(info)
            prog_bar.set_description(info)

    def _run_fc(self, train_loader, test_loader, optimizer, scheduler):
        self._network.to(self.device)

        prog_bar = tqdm(range(self.epochs_fc))
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = torch.autograd.Variable(inputs)

                outputs = self._network(inputs, mode="fc")
                logits = outputs["logits"]

                loss = F.cross_entropy(logits, targets.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self.compute_test_accuracy(test_loader, self._network)

            if self.copy_acc <= test_acc:
                self.copy_acc = test_acc
                self.copy_net = copy.deepcopy(self._network)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Best_acc {:.2f}".format(
                self.cur_task,
                epoch + 1,
                self.epochs_fc,
                losses / len(train_loader),
                train_acc,
                test_acc,
                self.copy_acc
            )
            self.logger.info(info)
            prog_bar.set_description(info)
        self._network = copy.deepcopy(self.copy_net)

    def generate_fake_feature(self, features_new, labels, sum_per_classes):
        features_need = []
        labels_need = []
        with torch.no_grad():
            features_mean = self.feature_mean
            features_var = self.feature_var
            features_std = np.sqrt(features_var)
            max_distance = 0.
            for i in range(self.increment):
                # 找到距离所有旧类最远的新类
                mean_new = features_mean[i, :]
                distance = 0.
                for j in range(self.known_class):
                    idx = j
                    d = mean_new - features_mean[idx, :]
                    d = d * d
                    d = d.sum()
                    distance += d
                if max_distance < distance:
                    max_distance = distance
                    mean_best = features_mean[i + self.known_class, :]
                    std_best = features_std[i + self.known_class, :]
                    idx_best = i + self.known_class
            print("choose the class {} as the fake generater".format(idx_best))

            for i in range(self.known_class):
                # 生成伪特征
                mean_old = features_mean[i, :]
                std_old = features_std[i, :]
                idx_start = sum_per_classes[:idx_best - self.known_class].sum()
                features_fake = copy.deepcopy(
                    features_new[idx_start:idx_start + sum_per_classes[idx_best - self.known_class]])
                labels_fake = copy.deepcopy(labels[idx_start:idx_start + sum_per_classes[idx_best - self.known_class]])
                for j in range(len(features_fake)):
                    features_fake[j] = features_fake[j] - mean_new + mean_old
                    # features_fake[j] = np.divide(features_fake[j] - mean_best, std_best) * std_old + mean_old
                    features_fake[j] = np.expand_dims(features_fake[j], axis=0)
                    labels_fake[j] = i
                index = herding_rule(np.concatenate(features_fake), 400)
                for item in index:
                    features_need.append(np.squeeze(features_fake[item]))
                    labels_need.append(labels_fake[item])

            for k in range(self.increment):
                idx_start = sum_per_classes[:k].sum()
                features_fake = copy.deepcopy(
                    features_new[idx_start:idx_start + sum_per_classes[k]])
                labels_fake = copy.deepcopy(labels[idx_start:idx_start + sum_per_classes[k]])
                for j in range(len(features_fake)):

                    features_fake[j] = np.expand_dims(features_fake[j], axis=0)
                index = herding_rule(np.concatenate(features_fake), 100)
                for item in index:
                    features_need.append(np.squeeze(features_fake[item]))
                    labels_need.append(labels_fake[item])

        return FeatureSet(features_need, labels_need)

    def compute_feature_mean(self, train_loader):
        self._network.eval()
        if self.cur_task == 0:
            num_classes = self.init_class
        else:
            num_classes = self.known_class + self.increment
        feature_dim = self._network.feature_dim()
        if self.cur_task > 0:
            old_feature_dim = self._old_network.feature_dim()

        feature_mean = torch.zeros((num_classes, feature_dim))
        feature_var = torch.zeros((num_classes, feature_dim))
        feature_mean = feature_mean.to(self.device)
        feature_var = feature_var.to(self.device)
        sum_per_classes = np.zeros((num_classes), dtype=int)
        features_new = []
        labels = []

        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # self._network = self._network.cpu()
            with torch.no_grad():
                features = self._network.extract_vector(inputs)
                for j in range(len(targets)):
                    feature_mean[targets[j]] += features[j, :]
                    feature_var[targets[j]] += torch.mul(features[j, :], features[j, :])
                    sum_per_classes[targets[j]] += 1
                    if targets[j] >= self.known_class:
                        features_new.append(features[j, :].cpu())
                        labels.append(targets[j].cpu())
        for i in range(num_classes):
            with torch.no_grad():
                feature_mean[i] /= sum_per_classes[i]
                feature_var[i] /= sum_per_classes[i]
                feature_var[i] = feature_var[i] - torch.mul(feature_mean[i], feature_mean[i])

        if self.cur_task == 0:
            return np.array(feature_mean.cpu()), np.array(feature_var.cpu()), features_new, labels, sum_per_classes
        else:
            return np.array(feature_mean.cpu()), np.array(feature_var.cpu()), features_new, labels, sum_per_classes[
                                                                                                    -self.increment:]

    def targets_map(self, targets):
        for i in range(len(targets)):
            targets[i] = self.class_list.index(targets[i])
        return targets

    @staticmethod
    def aux_targets_map(targets, class_list):
        for i in range(len(targets)):
            targets[i] = class_list.index(targets[i])
        return targets

    def compute_test_accuracy(self, test_loader, model):
        # model = self._network
        model.eval()
        correct, total = 0, 0
        device = self.device
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def eval_task(self, test_loader):
        model = self._network
        model.eval()
        confusion_matrix = np.zeros((self.known_class, self.known_class), dtype=int)  # 混淆矩阵
        task_confusion_matrix = np.zeros((self.cur_task + 1, self.cur_task + 1), dtype=int)  # 任务混淆矩阵
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            for j in range(len(predicts)):
                if predicts[j] < self.init_class:
                    pred_tsak = 0
                else:
                    pred_tsak = int(predicts[j] - self.init_class) // int(self.increment) + 1
                if targets[j] < self.init_class:
                    true_task = 0
                else:
                    true_task = int(targets[j] - self.init_class) // int(self.increment) + 1
                task_confusion_matrix[true_task, pred_tsak] += 1
                confusion_matrix[targets[j], predicts[j]] += 1
        correct = 0.
        for i in range(self.known_class):
            correct += confusion_matrix[i, i]
        acc = correct / confusion_matrix.sum()  # 识别准确率

        if self.cur_task == 0:
            old_task = []
            new_task = range(self.known_class)
        elif self.cur_task == 1:
            old_task = range(self.init_class)
            new_task = range(self.init_class, self.known_class)
        else:
            old_task = range(self.known_class-self.increment)
            new_task = range(self.known_class-self.increment, self.known_class)
        correct = 0.
        total = 0.
        for i in old_task:
            correct += confusion_matrix[i, i]
            total += confusion_matrix[i, :].sum()
        old_acc = 0. if total == 0. else correct / total
        correct = 0.
        total = 0.
        for i in new_task:
            correct += confusion_matrix[i, i]
            total += confusion_matrix[i, :].sum()
        new_acc = 0. if total == 0. else correct / total
        self.TaskConfusionMatrix.append(task_confusion_matrix)
        self.ConfusionMatrix.append(confusion_matrix)
        model.train()
        return acc * 100., old_acc * 100., new_acc * 100.

    def feature_store(self, train_loader):
        features_set = []
        label_set = []
        self._network.eval()
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                features = self._network.extract_vector(inputs)
                for j in range(len(targets)):
                    features_set.append(features[j, :].cpu())
                    label_set.append(targets[j].cpu())
        return FeatureSet(features_set, label_set)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def get_sampler(target):
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    return sampler


def l2_distance(features1, features2):
    return torch.mean(torch.mul(features1 - features2, features1 - features2))


def targets2task(targets, init_class, increment):
    # task_targets = torch.zeros([targets.shape[0]])
    task_targets = targets.clone()
    with torch.no_grad():
        for i in range(len(task_targets)):
            if targets[i] < init_class:
                task_targets[i] = 0
            else:
                task_targets[i] = (int(targets[i]) - init_class) // increment + 1
    return task_targets


def herding_rule(features, nb_examplars):
    # features = features.cpu()
    D = copy.deepcopy(features.T)
    # D = D.numpy()
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
