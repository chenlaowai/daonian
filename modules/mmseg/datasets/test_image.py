from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS as DETDATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS as SEGDATASETS

@SEGDATASETS.register_module()
class test_image_seg(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'ear'),
        palette=[[0, 0, 0], [17, 13, 66]]
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
# class test_imagedet(CocoDataset):
#     METAINFO = dict(
#         classes=('background', 'ear'),
#         palette=[(0, 0, 0), (17, 13, 66)]
#     )
#     COCOAPI = COCO
#     ANN_ID_UNIQUE = True
    