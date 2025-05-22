# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import (
    init_data,
)
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from evals.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation,
    FrameAggregation
)

import yaml

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def process_main():
    rank = 0
    fname = "JEPA/new/vitl16_k400_16x8x3.yaml"
    world_size = 1
    devices = ["cuda:0"]

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    # world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    return params


class VJEPAConfig:
    def __init__(self):
        args_eval = process_main()
        # -- PRETRAIN
        args_pretrain = args_eval.get('pretrain')
        self.checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
        self.model_name = args_pretrain.get('model_name', None)
        self.patch_size = args_pretrain.get('patch_size', None)
        self.pretrain_folder = args_pretrain.get('folder', None)
        self.ckp_fname = args_pretrain.get('checkpoint', None)
        self.tag = args_pretrain.get('write_tag', None)
        self.use_sdpa = args_pretrain.get('use_sdpa', True)
        self.use_SiLU = args_pretrain.get('use_silu', False)
        self.tight_SiLU = args_pretrain.get('tight_silu', True)
        self.uniform_power = args_pretrain.get('uniform_power', False)
        self.pretrained_path = os.path.join(self.pretrain_folder, self.ckp_fname)
        # Optional [for Video model]:
        self.tubelet_size = args_pretrain.get('tubelet_size', 2)
        self.pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

        # -- DATA
        args_data = args_eval.get('data')
        self.train_data_path = [args_data.get('dataset_train')]
        self.val_data_path = [args_data.get('dataset_val')]
        self.dataset_type = args_data.get('dataset_type', 'VideoDataset')
        self.num_classes_vid = args_data.get('num_classes_vid')
        self.num_classes_img = args_data.get('num_classes_img')

        self.eval_num_segments = args_data.get('num_segments', 1)
        self.eval_frames_per_clip = args_data.get('frames_per_clip', 16)
        self.eval_frame_step = args_pretrain.get('frame_step', 4)
        self.eval_duration = args_pretrain.get('clip_duration', None)
        self.eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)

        # -- OPTIMIZATION
        args_opt = args_eval.get('optimization')
        self.resolution = args_opt.get('resolution', 224)
        self.batch_size = args_opt.get('batch_size')
        self.attend_across_segments = args_opt.get('attend_across_segments', False)
        self.num_epochs = args_opt.get('num_epochs')
        self.wd = args_opt.get('weight_decay')
        self.start_lr = args_opt.get('start_lr')
        self.lr = args_opt.get('lr')
        self.final_lr = args_opt.get('final_lr')
        self.warmup = args_opt.get('warmup')
        self.use_bfloat16 = args_opt.get('use_bfloat16')

        # -- EXPERIMENT-ID/TAG (optional)
        self.resume_checkpoint = args_eval.get('resume_checkpoint', False)
        self.eval_tag = args_eval.get('tag', None)
        
        # -- PROBE CONFIG
        self.probe_frozen = True
        self.probe_checkpoint = "/app/model_zoo/k400-probe.pth.tar"
        return


def build_vjepa_encoder(config):
    device = "cpu" # caller is responsible for sending the model to the correct device

    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=config.resolution,
        device=device,
        pretrained=config.pretrained_path,
        model_name=config.model_name,
        patch_size=config.patch_size,
        tubelet_size=config.tubelet_size,
        frames_per_clip=config.pretrain_frames_per_clip,
        uniform_power=config.uniform_power,
        checkpoint_key=config.checkpoint_key,
        use_SiLU=config.use_SiLU,
        tight_SiLU=config.tight_SiLU,
        use_sdpa=config.use_sdpa)
    if config.pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=config.tubelet_size,
            attend_across_segments=config.attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def build_vjepa_classifier(config, encoder, video_data=True, checkpoint_path=None, frozen=False, frames_number=128):
    device = "cpu" # caller is responsible for sending the model to the correct device

    if video_data:
        num_classes=config.num_classes_vid
    else:
        num_classes=config.num_classes_img

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
        frames_number=frames_number
    ).to(device)

    if checkpoint_path is not None:
        config.probe_checkpoint=checkpoint_path


    classifier = load_probe_checkpoint(classifier=classifier, path=config.probe_checkpoint)
    
    if frozen:
        freeze_weights(classifier)
    
    return classifier


def load_probe_checkpoint(classifier, path):
    device = "cpu" # caller is responsible for sending the model to the correct device

    # -- load checkpoint
    classifier, _, _, _ = load_checkpoint(
        device=device,
        r_path=path,
        classifier=classifier,
    )
    return classifier


def freeze_weights(model):
    # -- freeze weights
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return


def vjepa_predict(encoder, classifier, clips, clip_indices, training, config):
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config.use_bfloat16):
        # Forward and prediction
        with torch.no_grad():
            outputs = encoder(clips, clip_indices)
            if not training:
                if config.attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os] for os in outputs]
        if training:
            if config.attend_across_segments:
                outputs = [classifier(o) for o in outputs]
            else:
                outputs = [[classifier(ost) for ost in os] for os in outputs]
    return outputs


def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
):
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    if attend_across_segments:
                        outputs = [classifier(o) for o in outputs]
                    else:
                        outputs = [[classifier(ost) for ost in os] for os in outputs]
            if training:
                if attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os] for os in outputs]

        # Compute loss
        if attend_across_segments:
            loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
        else:
            loss = sum([sum([criterion(ost, labels) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
        with torch.no_grad():
            if attend_across_segments:
                outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            else:
                outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
            top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))

    return top1_meter.avg


def load_checkpoint(
    device,
    r_path,
    classifier,
    opt=None,
    scaler=None,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['classifier']
        
        for key in list(pretrained_dict.keys()):
            if "module." in key:
                pretrained_dict[key.replace("module.", "")] = pretrained_dict[key]
                del pretrained_dict[key]

        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        if opt is None or scaler is None:
            return classifier, opt, scaler, epoch

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return classifier, opt, scaler, epoch


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
