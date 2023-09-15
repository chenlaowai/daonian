from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS as SEGDATASETS
import os.path as osp
import mmengine.fileio as fileio

@SEGDATASETS.register_module()
class Ccd7_seg(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'quex'),
        palette=[[0, 0, 0], [8, 62, 255]]
    )
    def __init__(self,
             ann_file,
             img_suffix='.jpg',
             seg_map_suffix='.png',
             **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
    