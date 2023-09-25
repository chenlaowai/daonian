from mmdeploy.apis import torch2onnx
from mmdeploy.apis.tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK
import os
import glob
from queue import Queue
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modules
# pytorch模型转onnx模型示例

def pytorch_to_onnx(model_path, model_num, information_queue):
    """
        该函数用于将训练出来的pytorch模型转换为onnx模型
        Args:
            model_path: 目标模型权重文件的地址,同时也是onnx模型的输出地址
            model_num: 模型的id,用于获取数据集的地址,该变量用于获取数据集的一张图片,用于模型转换
            information_queue: 用于向前端传输进度信息的队列
    """
    img = './utils/lidianchi_0057.jpg'
    work_dir = os.path.join(model_path, 'onnx')
    save_file = 'end2end.onnx'
    deploy_cfg = '../configs/mmdeploy_config/mmseg/segmentation_onnxruntime_static-512x512.py'
    model_cfg = 'work_dirs/nextvit_segformer/nextvit_segformer_b2_80k_lidianchi3g.py'
    model_checkpoint = 'work_dirs/nextvit_segformer/iter_80000.pth'
    device = 'cpu'

    # 1. convert model to onnx
    information_queue.put('change_onnx')
    torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
               model_checkpoint, device)

    # 2. extract pipeline info for sdk use (dump-info)
    information_queue.put('change_tensorrt')
    export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)

    information_queue.put('all_end')


if __name__ == '__main__':
    model_path = 'work_dirs/nextvit_segformer'
    model_num = 7
    a = Queue(maxsize=100000)
    pytorch_to_onnx(model_path, model_num, a)