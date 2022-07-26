import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .utils import import_attr, load_model_state_dict

import itertools


def sample_swag(args):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            warnings.warn('Not parallelized but given more than 1 GPUs')

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda', 0)

        torch.backends.cudnn.benchmark = True
    else:  # CPU multithreading
        device = torch.device('cpu')

        if args.num_threads is None:
            args.num_threads = int(os.environ['SLURM_CPUS_ON_NODE'])

        torch.set_num_threads(args.num_threads)

    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()


    # Train data will be used to update the batch normalization (Should migrate that into train loop later)
    if args.swag:
        train_dataset = FieldDataset(
            in_patterns=args.train_in_patterns,
            tgt_patterns=args.train_tgt_patterns,
            in_norms=args.in_norms,
            tgt_norms=args.tgt_norms,
            callback_at=args.callback_at,
            augment=args.augment,
            aug_shift=args.aug_shift,
            aug_add=args.aug_add,
            aug_mul=args.aug_mul,
            crop=args.crop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop,
            crop_step=args.crop_step,
            in_pad=args.in_pad,
            tgt_pad=args.tgt_pad,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.loader_workers,
            pin_memory=True,
        )
    

    # Test data set for real testing
    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(sum(in_chan), sum(out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)

    if args.swag:
        args.swag_no_use_conv = not args.swag_use_conv

        SWAG = import_attr('SWAG', models, callback_at=args.callback_at)
        # Initialize model
        swag_model = SWAG(
            base = import_attr(args.model, models, callback_at=args.callback_at),
            no_cov_mat = args.swag_no_use_conv,
            max_num_models=args.swag_K,
            in_chan = sum(in_chan),
            out_chan = sum(out_chan),
            scale_factor=args.scale_factor,
            **args.misc_kwargs
        )
        swag_model.to(device)

    criterion = import_attr(args.criterion, torch.nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)


    
    # Statistics for testing
    if args.stats_callback is not None:
        test_stats = import_attr(args.stats_callback,callback_at=args.callback_at)
        test_stats = test_stats()
        print('Test statistics is enabled!')
    else:
        print('Test statistics is not enabled!')

    make_prefix_dir(args.valid_npy_out_dir) 
    
    state = torch.load(args.load_state, map_location=device)
    if not args.swag:
        load_model_state_dict(model, state['model'], strict=args.load_state_strict)
        print('model state at epoch {} loaded from {}'.format(
            state['epoch'], args.load_state))
    else:
        swag_model.load_state_dict(state['swag_model'])
    del state

    # Prepare batch norm with train data for swag run. Probably should migrate this to train_swag.py later
    if args.swag:
        swag_model.sample(scale=args.swag_sample_scale,cov=args.swag_use_conv)
        bn_update(train_loader,swag_model,device,subset=5) # Disable for quick testing


    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, target = data['input'], data['target']

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)


            if args.swag:
                #swag_model.sample(scale=args.swag_sample_scale,cov=args.swag_use_conv)
                #bn_update(train_loader,swag_model,device,subset=5)
                output = swag_model(input)
            else:
                output = model(input)
           
            if i < 5:
                print('##### sample :', i)
                print('input shape :', input.shape)
                print('output shape :', output.shape)
                print('target shape :', target.shape)

            n_input, n_output, n_target = narrow_cast(input, output, target)
            if i < 5:
                print('narrowed shape :', n_output.shape, flush=True)

            loss = criterion(n_output, n_target)

            print('sample {} loss: {}'.format(i, loss.item()))

            
            # Computing test stats
            #test_stats.compute_stat(input,output,target,relpath=data['target_relpath'])
        

            #if args.in_norms is not None:
            #    start = 0
            #    for norm, stop in zip(test_dataset.in_norms, np.cumsum(in_chan)):
            #        norm = import_attr(norm, norms, callback_at=args.callback_at)
            #        norm(input[:, start:stop], undo=True, **args.misc_kwargs)
            #        start = stop
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm = import_attr(norm, norms, callback_at=args.callback_at)
                    norm(n_output[:, start:stop], undo=True, **args.misc_kwargs)
                    #norm(target[:, start:stop], undo=True, **args.misc_kwargs)
                    start = stop
            #test_stats.compute_stat(input,output,target,relpath=data['target_relpath'])
            #test_dataset.assemble('_in', in_chan, input,
            #                      data['input_relpath'])
            print('Relative path:',data['target_relpath'])
            #test_dataset.assemble(f'_out-{args.swag_num_samples-num_samples+1}', out_chan, n_output,
            #                    data['target_relpath'])
            #test_dataset.assemble('_tgt', out_chan, target,
            #                      data['target_relpath'])
            
            test_dataset.assemble(f'_swag_out', out_chan, n_output,add_prefix(data['target_relpath'],prefix=args.valid_npy_out_dir))

def add_prefix(path_gp,prefix):
    if prefix is not None:
        return [[f'{prefix}/{path}' for path in gp] for gp in path_gp]
    return path_gp

def make_prefix_dir(prefix):
    if prefix is not None:
        from pathlib import Path
        if not Path(prefix).exists():
            os.mkdir(prefix)
# Code modified from Maddox (https://github.com/wjmaddox/swa_gaussian/tree/b172d93278fdb92522c8fccb7c6bfdd6f710e4f0)
def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module,flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model,device,subset=None):
    if not check_bn(model):
        return

    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module,momenta))
    n = 0

    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader,num_batches)

        for i, data in enumerate(loader):
            input, target = data['input'], data['target']

            input = input.to(device, non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b/(n+b)


            for module in momenta.keys():
                module.momentum = momentum
            
            model(input_var)

            n += b
    print('bn_update called with iteration: ',i) 
    model.apply(lambda module: _set_momenta(module,momenta))

