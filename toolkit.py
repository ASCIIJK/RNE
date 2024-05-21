import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt


def compute_accuracy(model, loader, device, class_list):
    model.eval()
    correct, total = 0, 0
    for i, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)["logits"]
            # outputs = model(inputs)
        predicts = torch.max(outputs, dim=1)[1]
        for j in range(len(predicts)):
            predicts[j] = class_list[predicts[j]]
        correct += (predicts.cpu() == targets).sum()
        total += len(targets)

    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=1):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    if len(idxes) == 0:
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / 1, decimals=2
        )
    else:
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    print(f'old acc is {all_acc["old"]}, new acc is {all_acc["new"]}')
    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in tqdm(imgs):
        images.append(np.array(pil_loader(item[0])))
        labels.append(item[1])

    return np.array(images), np.array(labels)


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def plot_confusion_matrix(confusion_matrix, savename, title='Confusion Matrix'):
    cm = confusion_matrix
    classes = list(range(confusion_matrix.shape[0]))
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    
    ind_array = np.arange(len(classes) + 1)
    x, y = np.meshgrid(ind_array, ind_array) 
    diags = np.diag(cm)  
    FP = sum(cm.sum(axis=0)) - sum(np.diag(cm))  
    FN = sum(cm.sum(axis=1)) - sum(np.diag(cm)) 
    TP = sum(np.diag(cm))  
    TN = sum(cm.sum().flatten()) - (FP + FN + TP)  
    SUM = TP + FP
    PRECISION = TP / (TP + FP)  
    RECALL = TP / (TP + FN)  
    TP_FNs, TP_FPs = [], []
    for x_val, y_val in zip(x.flatten(), y.flatten()):  
        max_index = len(classes)
        if x_val != max_index and y_val != max_index:  
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, c, color='black', fontsize=15, va='center', ha='center')
        elif x_val == max_index and y_val != max_index:  
            TP = diags[y_val]
            TP_FN = cm.sum(axis=1)[y_val]
            recall = TP / (TP_FN)
            if recall != 0.0 and recall > 0.01:
                recall = str('%.2f' % (recall * 100,)) + '%'
            elif recall == 0.0:
                recall = '0'
            TP_FNs.append(TP_FN)
            plt.text(x_val, y_val, str(TP_FN) + '\n' + str(recall) + '%', color='black', va='center', ha='center')
        elif x_val != max_index and y_val == max_index:  
            TP = diags[x_val]
            TP_FP = cm.sum(axis=0)[x_val]
            precision = TP / (TP_FP)
            if precision != 0.0 and precision > 0.01:
                precision = str('%.2f' % (precision * 100,)) + '%'
            elif precision == 0.0:
                precision = '0'
            TP_FPs.append(TP_FP)
            plt.text(x_val, y_val, str(TP_FP) + '\n' + str(precision), color='black', va='center', ha='center')
    cm = np.insert(cm, max_index, TP_FNs, 1)
    cm = np.insert(cm, max_index, np.append(TP_FPs, SUM), 0)
    plt.text(max_index, max_index, str(SUM) + '\n' + str('%.2f' % (PRECISION * 100,)) + '%', color='red', va='center',
             ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('actual label')
    plt.xlabel('predict label')
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)
    # show confusion matrix
    plt.savefig(savename, format='png')
