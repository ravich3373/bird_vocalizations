import hydra
import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import os
from omegaconf import OmegaConf
from dataset.birdclef import build_dataset
from efficientnet_pytorch import EfficientNet
from model.leaf import PCEN, LogTBN, Leaf, EfficientLeaf
from model.mel import STFT, MelFilter, Squeeze
from model import AudioClassifier
from utils import optimizer_to, scheduler_to
from engine import train


# Default LEAF parameters
n_filters = 40
sample_rate = 16000
window_len = 25.0
window_stride = 10.0
min_freq = 60.0
max_freq = 7800.0


def criterion_and_optimizer(cfg, network):
        ## init criterion, optimizer and set scheduler
        criterion = nn.CrossEntropyLoss(reduction='none')
        if cfg.frontend_lr_factor == 1:
            params = network.parameters()
        else:
            frontend_params = list(network._frontend.parameters())
            frontend_paramids = set(id(p) for p in frontend_params)
            params = [p for p in network.parameters()
                      if id(p) not in frontend_paramids]
        optimizer = torch.optim.Adam(params, lr=cfg.lr, eps=cfg.adam_eps)
        if cfg.frontend_lr_factor != 1:
            optimizer.add_param_group(dict(
                params=frontend_params,
                lr=cfg.lr * cfg.frontend_lr_factor))

        ## lr scheduler
        if cfg.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max' if cfg.scheduler_mode == 'acc' else 'min',
                factor=cfg.scheduler_factor,
                patience=cfg.patience)
        else:
            scheduler = None

        return scheduler,criterion, optimizer


@hydra.main(version_base=None, config_path="cfgs", config_name="default")
def main(cfg):
    OmegaConf.set_struct(cfg, False)    # allows creting new attributes in cfg
    
    device = torch.device(cfg.device)
    if cfg.seed:
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    train_loader, val_loader, test_loader, cfg.nb_classes = build_dataset(args=cfg)

    ## init encoder
    if cfg.compression == 'TBN' and cfg.tbn_median_filter and cfg.tbn_median_filter_append:
        frontend_channels = 2
    else:
        frontend_channels = 1
    encoder = EfficientNet.from_name("efficientnet-b0", num_classes=cfg.nb_classes, include_top=False,
                                     in_channels=frontend_channels)
    encoder._avg_pooling = torch.nn.Identity()

    ## init compression layer
    if cfg.compression == 'PCEN':
        compression_fn = PCEN(num_bands=n_filters,
                              s=0.04,
                              alpha=0.96,
                              delta=2.0,
                              r=0.5,
                              eps=1e-12,
                              learn_logs=cfg.pcen_learn_logs,
                              clamp=1e-5)
    elif cfg.compression == 'TBN':
        compression_fn = LogTBN(num_bands=n_filters,
                                a=cfg.log1p_initial_a,
                                trainable=cfg.log1p_trainable,
                                per_band=cfg.log1p_per_band,
                                median_filter=cfg.tbn_median_filter,
                                append_filtered=cfg.tbn_median_filter_append)

    ## init frontend
    if cfg.frontend == 'Leaf':
        frontend = Leaf(n_filters=n_filters,
                        min_freq=min_freq,
                        max_freq=max_freq,
                        sample_rate=sample_rate,
                        window_len=window_len,
                        window_stride=window_stride,
                        compression=compression_fn)
    elif cfg.frontend == 'EfficientLeaf':
        frontend = EfficientLeaf(n_filters=n_filters,
                                 num_groups=cfg.num_groups,
                                 min_freq=min_freq,
                                 max_freq=max_freq,
                                 sample_rate=sample_rate,
                                 window_len=window_len,
                                 window_stride=window_stride,
                                 conv_win_factor=cfg.conv_win_factor,
                                 stride_factor=cfg.stride_factor,
                                 compression=compression_fn)
    elif cfg.frontend == 'Mel':
        window_size = int(sample_rate * window_len / 1000 + 1)  # to match Leaf
        hop_size = int(sample_rate * window_stride / 1000 + 1)
        frontend = nn.Sequential(OrderedDict([
            ('stft', STFT(window_size, hop_size, complex=False)),
            ('filterbank', MelFilter(sample_rate, window_size, n_filters,
                                     min_freq, max_freq)),
            ('squeeze', Squeeze(dim=1)),  # remove channel dim to match Leaf
            ('compression', compression_fn),
        ]))

    ## init classifier
    network = AudioClassifier(
        num_outputs=cfg.nb_classes,
        frontend=frontend,
        encoder=encoder)

    scheduler = cfg.scheduler
    if not cfg.resume or not os.path.exists(cfg.resume):
        scheduler, criterion, optimizer = criterion_and_optimizer(cfg, network)

    ## load previous run
    if cfg.resume and not os.path.exists(cfg.resume):
        print("resume file %s does not exist; ignoring" % cfg.resume)

    if cfg.resume and os.path.exists(cfg.resume):
        saved_dict = torch.load(cfg.resume, map_location=torch.device('cpu'))
        network.load_state_dict(saved_dict['network'])
        cfg, criterion, optimizer = criterion_and_optimizer(cfg, network)
        cfg.start_epoch = saved_dict['epoch'] + 1
        optimizer.load_state_dict(saved_dict['optimizer'])
        if scheduler is not None and saved_dict['scheduler'] is not None:
            scheduler.load_state_dict(saved_dict['scheduler'])
        del saved_dict
    
    network.to(device)
    torch.cuda.empty_cache()
    optimizer_to(optimizer, device)
    scheduler_to(scheduler, device)

    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    train(network=network, loader_train=train_loader, loader_val=val_loader, loader_test=test_loader,
              path=cfg.output_dir, criterion=criterion, optimizer=optimizer, num_epochs=cfg.epochs, tqdm_on=True,
              overwrite_save=cfg.overwrite_save, save_every=cfg.save_every, starting_epoch=cfg.start_epoch,
              test_every_epoch=cfg.test_every_epoch,
              scheduler=scheduler, scheduler_item=cfg.scheduler_mode, scheduler_min_lr=cfg.min_lr,
              warmup_steps=cfg.warmup_steps,
              save_best_model=cfg.save_best_model, model_name=cfg.model_name)


if __name__ == "__main__":
    main()