from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS as DETDATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS as SEGDATASETS

# @SEGDATASETS.register_module()
# class test_image_seg(BaseSegDataset):
#     METAINFO = dict(
#         classes=('ear'),
#         palette=[[211, 230, 89]]
#     )
#     def __init__(self,
#                  img_suffix='.jpg',
#                  seg_map_suffix='.png',
#                  **kwargs) -> None:
#         super().__init__(
#             img_suffix=img_suffix,
#             seg_map_suffix=seg_map_suffix,
#             **kwargs)

@DETDATASETS.register_module()
class test_image_det(CocoDataset):
    METAINFO = dict(
        classes=('ear'),
        palette=[(211, 230, 89)]
    )
    COCOAPI = COCO
    ANN_ID_UNIQUE = True
        