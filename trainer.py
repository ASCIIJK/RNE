import sys
import logging
import torch
from utils import factory
from data_manger import DataManger
import os


def train(args):
    _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        init_cls,
        args["increment"],
        args["convnet_type"],
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
        ],
    )

    _set_random(seed=args["seed"])
    _set_device(args)
    print_args(args)

    datamanger = DataManger(
        args["dataset"],
        args["shuffle"],
        args["init_cls"],
        args["increment"],
        args["device"]
    )
    model = factory.get_model(args["model_name"], args, logging)

    model.init_train(datamanger=datamanger)
    model.after_train()

    for i in range(datamanger.task_size):
        model.increment_train()
        model.after_train()


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)

    args["device"] = gpus[0]


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _set_random(seed: int = 1993):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
