# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler
from typing import Dict, Iterator, Optional, Union

import numpy as np
from mmengine.dataset import BaseDataset

from mmengine.dist import get_dist_info, sync_random_seed
from mmseg.registry import DATA_SAMPLERS

from PIL import Image

@DATA_SAMPLERS.register_module()
class ClassAware_InfiniteSampler(Sampler):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 seed: Optional[int] = None,
                 num_sample_class: int = 1,
                 background_balance: bool = False) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.size = len(dataset)
        self.indices = self._indices_of_rank() # 索引产出

        # The number of samples taken from each per-label list
        assert num_sample_class > 0 and isinstance(num_sample_class, int)
        self.num_sample_class = num_sample_class
        # Get per-label image list from dataset
        self.cat_dict = self.get_cat2imgs(background_balance)

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / world_size))
        self.total_size = self.num_samples * self.world_size

        # get number of images containing each category 输出每个类别包含的个数列表
        self.num_cat_imgs = [len(x) for x in self.cat_dict.values()]
        # filter labels without images 输出类别里的个数不为0的类别列表
        self.valid_cat_inds = [
            i for i, length in enumerate(self.num_cat_imgs) if length != 0
        ]
        self.num_classes = len(self.valid_cat_inds)

    def get_cat2imgs(self, background_balance) -> Dict[int, list]:
        """Get a dict with class as key and img_ids as values.

        Returns:
            dict[int, list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        classes = self.dataset.metainfo.get('classes', None)
        if classes is None:
            raise ValueError('dataset metainfo must contain `classes`')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(classes))}
        for i in range(len(self.dataset)):
            # cat_ids = set(self.dataset.get_cat_ids(i))
            data_info = self.dataset.get_data_info(i)
            cat_ids = self.get_unique_labels(data_info, background_balance)
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def get_unique_labels(self, data_info, background_ground):
        try:
            # 获取掩码图像地址
            seg_map_path = data_info['seg_map_path']

            # 打开掩码图像
            seg_map = Image.open(seg_map_path)

            # 获取图像的像素数据
            pixels = seg_map.getdata()

            # 统计像素值对应的标签类型
            unique_labels = set(pixels)

            if background_ground:
                # 判断是否包含背景以及其他标签
                has_background = 0 in unique_labels
                unique_labels.discard(0)  # 去除0标签

                # 如果只有一个背景标签，保留
                if has_background and len(unique_labels) == 0:
                    unique_labels.add(0)

            return list(unique_labels)
        except Exception as e:
            print(f"Error: {e}")
            return []

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            # initialize label list
            label_iter_list = RandomCycleIter(self.valid_cat_inds, generator=g)  # 类别随机
            # initialize each per-label image list
            data_iter_dict = dict()
            for i in self.valid_cat_inds:
                data_iter_dict[i] = RandomCycleIter(self.cat_dict[i], generator=g)  # 每一类中随机取样本

            def gen_cat_img_inds(cls_list, data_dict, num_sample_cls):
                """Traverse the categories and extract `num_sample_cls` image
                indexes of the corresponding categories one by one."""
                # 遍历类别并提取`num_sample_cls`图像对应类别的索引一一对应
                id_indices = []
                for _ in range(len(cls_list)):
                    cls_idx = next(cls_list)
                    for _ in range(num_sample_cls):
                        id = next(data_dict[cls_idx])
                        id_indices.append(id)
                return id_indices

            # deterministically shuffle based on epoch
            num_bins = int(
                math.ceil(self.total_size * 1.0 / self.num_classes /
                          self.num_sample_class))

            indices = []
            for i in range(num_bins):
                indices += gen_cat_img_inds(label_iter_list, data_iter_dict,
                                            self.num_sample_class)
            for indice in indices:
                yield indice

    def _indices_of_rank(self) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        print(
            "shaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyi\n",
            "shaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyi\n",
            "shaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyi\n",
            "shaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyi\n",
            "shaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyishaboyi")
        yield from self.indices

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass

class RandomCycleIter:
    """Shuffle the list and do it again after the list have traversed.

    The implementation logic is referred to
    https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/datasets/loader/sampler.py

    Example:
        >>> label_list = [0, 1, 2, 4, 5]
        >>> g = torch.Generator()
        >>> g.manual_seed(0)
        >>> label_iter_list = RandomCycleIter(label_list, generator=g)
        >>> index = next(label_iter_list)
    Args:
        data (list or ndarray): The data that needs to be shuffled.
        generator: An torch.Generator object, which is used in setting the seed
            for generating random numbers.
    """  # noqa: W605

    def __init__(self,
                 data: Union[list, np.ndarray],
                 generator: torch.Generator = None) -> None:
        self.data = data
        self.length = len(data)
        self.index = torch.randperm(self.length, generator=generator).numpy()
        self.i = 0
        self.generator = generator

    def __iter__(self) -> Iterator:
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __next__(self):
        if self.i == self.length:
            self.index = torch.randperm(
                self.length, generator=self.generator).numpy()
            self.i = 0
        idx = self.data[self.index[self.i]]
        self.i += 1
        return idx
