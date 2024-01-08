# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')



def inference(configs, shape, logger: MMLogger) -> dict:
    config_name = Path(configs)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'

    init_default_scope(cfg.get('scope', 'mmseg'))

    if len(shape) == 1:
        input_shape = (3, shape[0], shape[0])
    elif len(shape) == 2:
        input_shape = (3, ) + tuple(shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    result['ori_shape'] = input_shape[-2:]
    result['pad_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)]
    }
    data = model.data_preprocessor(data_batch)
    model.eval()
    if cfg.model.decode_head.type in ['MaskFormerHead', 'Mask2FormerHead']:
        # TODO: Support MaskFormer and Mask2Former
        raise NotImplementedError('MaskFormer and Mask2Former are not '
                                  'supported yet.')
    outputs = get_model_complexity_info(
        model,
        input_shape,
        inputs=data['inputs'],
        show_table=False,
        show_arch=False)
    result['flops'] = _format_size(outputs['flops'])
    result['params'] = _format_size(outputs['params'])
    result['compute_type'] = 'direct: randomly generate a picture'
    return result


def main():
    configs = ''
    shape = [2000, 1500]
    logger = MMLogger.get_instance(name='MMLogger')

    result = inference(configs, shape, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()