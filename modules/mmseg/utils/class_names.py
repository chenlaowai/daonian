# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_str

def CCD7_classes():
    return ['background', 'quex']

def CCD7_palette():
    return [[0, 0, 0], [8, 62, 255]]

def Ccd7_classes():
    return ['background', 'quex']

def Ccd7_palette():
    return [[0, 0, 0], [8, 62, 255]]


def Ccd6_classes():
    return ['background', 'SEALANG']

def Ccd6_palette():
    return [[0, 0, 0], [8, 62, 255]]

def shaboyi_classes():
    return ['background', 'SEALANG']

def shaboyi_palette():
    return [[0, 0, 0], [128, 0, 128]]


def test_image_classes():
    return ['background', 'ear']

def test_image_palette():
    return [[0, 0, 0], [17, 13, 66]]



def get_classes(dataset):
    """Get class names of a dataset."""
    if is_str(dataset):
        labels = eval(dataset + '_classes()')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    if is_str(dataset):
        labels = eval(dataset + '_palette()')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels

def set_dataset_meta(dataset):
    classes_palette: dict = {'classes': get_classes(dataset), 'palette': get_palette(dataset)}
    return classes_palette