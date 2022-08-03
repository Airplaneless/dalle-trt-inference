import sys
import json
from re import T
from turtle import forward
import numpy
import torch
from rrdbnet_arch import RRDBNet


class RealESRGAN(torch.nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, img):
        res = self.model(img[0:4])
        for i in range(4, img.shape[0], 4):
            res = torch.cat((res, self.model(img[i:i+4])), 0)
        return res.permute((0,2,3,1)).clamp_(0, 1)


if __name__ == '__main__':
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=8)
    net.load_state_dict(torch.load('models/RealESRGAN_x8.pth'), strict=True)
    model = RealESRGAN(net).eval()
    for p in model.parameters():
        p.requires_grad = False
    inputs = (
        torch.randn((4, 3, 240, 240)),
    )
    torch.onnx.export(model, inputs, f'./onnx/esrgan1x1.onnx', output_names=['z'], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)

