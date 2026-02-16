import torch


def get_params(model, args, nettype: str):
    init_lr = args.lr
    if nettype == "diversify-adv":
        return [
            {"params": model.dbottleneck.parameters(), "lr": args.lr_decay2 * init_lr},
            {"params": model.dclassifier.parameters(), "lr": args.lr_decay2 * init_lr},
            {"params": model.ddiscriminator.parameters(), "lr": args.lr_decay2 * init_lr},
        ]
    if nettype == "diversify-cls":
        return [
            {"params": model.bottleneck.parameters(), "lr": args.lr_decay2 * init_lr},
            {"params": model.regressor.parameters(), "lr": args.lr_decay2 * init_lr},
            {"params": model.discriminator.parameters(), "lr": args.lr_decay2 * init_lr},
        ]
    if nettype == "diversify-all":
        return [
            {"params": model.featurizer.parameters(), "lr": args.lr_decay1 * init_lr},
            {"params": model.abottleneck.parameters(), "lr": args.lr_decay2 * init_lr},
            {"params": model.aregressor.parameters(), "lr": args.lr_decay2 * init_lr},
        ]
    raise ValueError(f"Unsupported nettype: {nettype}")


def get_optimizer(model, args, nettype: str) -> torch.optim.Optimizer:
    params = get_params(model, args, nettype)
    return torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
