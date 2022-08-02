import json
import numpy
import torch

from dalle import VQGanDetokenizer, DalleBartEncoder, DalleBartDecoder, TextTokenizer

import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)

if __name__ == '__main__':
    detokenizer = VQGanDetokenizer()
    detokenizer.load_state_dict(torch.load('models/detoker.pt'))
    detokenizer = detokenizer.cpu().eval()
    for p in detokenizer.parameters():
        p.requires_grad = False
    inputs = (
        torch.randint(low=0, high=10000, size=(1, 256), dtype=torch.long)
    )
    torch.onnx.export(detokenizer, inputs, 'onnx/vqgan/vqgan.onnx', output_names=['z'], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)