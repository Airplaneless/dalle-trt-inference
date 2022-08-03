import sys
import json
import numpy
import torch
import RRDBNet_arch as arch


import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)

if __name__ == '__main__':
    aspect = int(sys.argv[1])
    assert aspect in [1,2]
    model = arch.RRDBNet(3, 3, 64, 23, gc=32).eval()
    model.load_state_dict(torch.load('models/RRDB_ESRGAN_x4.pth'), strict=True)
    for p in model.parameters():
        p.requires_grad = False
    inputs = (
        torch.rand((1,3,256,256 * aspect))
    )
    torch.onnx.export(model, inputs, f'onnx/srgan1x{aspect}.onnx', output_names=['z'], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)
