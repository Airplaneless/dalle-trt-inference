import sys
import json
from re import T
import numpy
import torch

from dalle import VQGanDetokenizer, VQGanDetokenizerB

import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)

if __name__ == '__main__':
    detokenizer = VQGanDetokenizerB()
    detokenizer.load_state_dict(torch.load('models/detoker.pt'))
    detokenizer = detokenizer.cpu().eval()
    detokenizer.dx = 1
    detokenizer.dy = int(sys.argv[1])
    for p in detokenizer.parameters():
        p.requires_grad = False
    inputs = (
        torch.randint(low=0, high=10000, size=(16, 256), dtype=torch.long),
    )
    torch.onnx.export(detokenizer, inputs, f'onnx/vqgan/vqgan4x4.onnx', output_names=['z'], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)
