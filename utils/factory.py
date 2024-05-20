def get_model(model_name, args, loger):
    name = model_name.lower()
    data_name = args["dataset"]
    if data_name == "cifar100" and name == "rne":
        from models.RNE_cifar100 import RNE
        return RNE(args, loger)
    elif (data_name == "imagenet100" or "ifood101") and name == "rne":
        from models.RNE_imagenet import RNE
        return RNE(args, loger)
    elif data_name == "cifar100" and name == "rne_compress":
        from models.RNE_compress_cifar import RNE
        return RNE(args, loger)
    elif (data_name == "imagenet100" or "ifood101") and name == "rne_compress":
        from models.RNE_compress_cifar import RNE
        return RNE(args, loger)
    else:
        assert 0
