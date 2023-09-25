from mmdeploy_runtime import Segmentor
import cv2
import numpy as np
import os


def onnx_inference(image_path, model_path):
    """
        该函数用于onnx模型的推理
        Args:
            image_path: product文件夹下图片的文件名
            model_path: onnx模型的地址
    """
    img = cv2.imread(os.path.join('utils', image_path))  # 更改为要推理的图片
    output_path = 'utils/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create a classifier
    segmentor = Segmentor(model_path=model_path, device_name='cpu', device_id=0)  # 更改为执行推理的部署模型
    # perform inference
    seg = segmentor(img)

    # visualize inference result
    ## random a palette with size 256x3 生成一个调色板
    palette = [[20, 255, 28], [8, 62, 255], [255, 3, 7], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                    [0, 13, 255], [34, 255, 0], [255, 47, 50], [255, 255, 87], [85, 85, 0]]
    palette = np.array(palette, dtype=np.uint8)
    for label, color in enumerate(palette):
        label += 1  # 此行为了跳过背景类
        mask_bool = seg == label
        img[mask_bool] = color
    img = img.astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, image_path), img)
    return image_path

if __name__ == '__main__':
    model_path = 'work_dirs/nextvit_segformer/onnx'  # 更改为推理模型的路径
    image_path = 'lidianchi_0057.jpg'  # 更改为推理图片的路径
    onnx_inference(image_path, model_path)