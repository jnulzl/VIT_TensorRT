import os
import sys
import cv2
import torch
import time

import pyvit_tensorrt as vit
from utils import read_exp_img_list


class TrtInternVLVIT(object):
    def __init__(self, trt_vit_path, device_id=0):
        # 1. 加载模型
        vit_config = vit.BaseConfig()

        vit_config.input_names = ["pixel_values"]
        vit_config.output_names = ["vit_embeds"]
        vit_config.weights_path = trt_vit_path
        vit_config.deploy_path = trt_vit_path

        vit_config.means[0] = 123.675
        vit_config.means[1] = 116.28
        vit_config.means[2] = 103.53

        vit_config.scales[0] = 0.0171
        vit_config.scales[1] = 0.0175
        vit_config.scales[2] = 0.0174

        vit_config.mean_length = 3
        vit_config.net_inp_channels = 3

        vit_config.net_inp_width = 448
        vit_config.net_inp_height = vit_config.net_inp_width

        vit_config.num_threads = 2
        vit_config.batch_size = 8
        vit_config.device_id = device_id

        self.alg_obj = vit.PyVit()
        self.alg_obj.init(vit_config)
        print("Init VitTrt ...")

    def __del__(self):
        print("Release VitTrt ...")
        self.alg_obj.deinit()

    def infer(self, img_list):
        # 3. 推理
        self.alg_obj.process(img_list)

        output = self.alg_obj.get_result()
        # print("output : ", output, output.shape)
        return output.to(torch.bfloat16)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: \n\tpython trt_vit_inference.py <vit_trt_path> <device_id> <img_dir>"
        )
        sys.exit(1)
    vit_onnx_path = sys.argv[1]
    device_id = int(sys.argv[2])
    img_dir = sys.argv[3]
    model = TrtInternVLVIT(vit_onnx_path, device_id)
    img_list = read_exp_img_list(img_dir)
    vit_embeds = model.infer(img_list)
    print("Vit_embeds:", vit_embeds)
