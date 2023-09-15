from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS as DETDATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS as SEGDATASETS

@SEGDATASETS.register_module()
class DefectSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('ear_fold_up', 'ear_fold_horizontal', 'ear_unweld'),
        palette=[[128, 0, 0], [0, 128, 0], [128, 128, 0]]
    )
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)

# @DETDATASETS.register_module()
# class DefectDetDataset(CocoDataset):
#     METAINFO = dict(
#         classes=('ear', 'tab'),
#         palette=[(10, 30, 70), (120, 190, 1)]
#     )
#     COCOAPI = COCO
#     ANN_ID_UNIQUE = True
