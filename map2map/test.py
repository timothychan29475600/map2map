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


def test(args):
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
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()
    if args.dropout_sample:
        model.train()
    
    with torch.no_grad():
        count = 1
        for i, data in enumerate(test_loader):
            input, target = data['input'], data['target']

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            if i < 5:
                print('##### sample :', i)
                print('input shape :', input.shape)
                print('output shape :', output.shape)
                print('target shape :', target.shape)

            input, output, target = narrow_cast(input, output, target)
            if i < 5:
                print('narrowed shape :', output.shape, flush=True)

            loss = criterion(output, target)

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
                    norm(output[:, start:stop], undo=True, **args.misc_kwargs)
                    #norm(target[:, start:stop], undo=True, **args.misc_kwargs)
                    start = stop
            #test_stats.compute_stat(input,output,target,relpath=data['target_relpath'])
            #test_dataset.assemble('_in', in_chan, input,
            #                      data['input_relpath'])
            test_dataset.assemble(f'_out', out_chan, output,
                                  add_prefix(data['target_relpath'],prefix=args.valid_npy_out_dir))
            print(data['target_relpath']) 
            #test_dataset.assemble('_tgt', out_chan, target,
            #                      data['target_relpath'])
            count += 1
def add_prefix(path_gp,prefix):
    if prefix is not None:
        return [[f'{prefix}/{path}' for path in gp] for gp in path_gp]
    return path_gp

def make_prefix_dir(prefix):
    if prefix is not None:
        from pathlib import Path
        if not Path(prefix).exists():
            os.mkdir(prefix)
