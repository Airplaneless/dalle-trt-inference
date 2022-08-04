import sys
import json
from re import T
from turtle import forward
import numpy
import torch

from dalle import VQGanDetokenizer, VQGanDetokenizerB

import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)


class VQGanDetokenizerV(torch.nn.Module):
    
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, z):
        z.clamp_(0, self.model.vocab_count - 1)
        z = self.model.embedding.forward(z)
        z = z.reshape(-1, 2**8, 2**8).reshape(1, -1, 2**8)
        z = z.view((1, -1, 16, 2 ** 8))
        # z = z.repeat(1,1,2,1)
        t1 = z.shape[1]; t2 = z.shape[2]
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.model.post_quant_conv.forward(z)
        z = self.model.decoder.conv_in.forward(z)
        z = self.model.decoder.mid.forward(z, t1, t2)
        for i in range(4, -1, -1):
            z = self.model.decoder.up[i].forward(z, t1, t2)
        z = self.model.decoder.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.model.decoder.conv_out.forward(z)
        z = z.permute(0, 2, 3, 1)
        return (z.clip(0.0, 1.0) * 255).cpu().detach()



if __name__ == '__main__':
    detokenizer = VQGanDetokenizerB()
    detokenizer.load_state_dict(torch.load('models/detoker.pt'))
    model = VQGanDetokenizerV(detokenizer)
    model = model.cpu().eval()
    for p in model.parameters():
        p.requires_grad = False
    inputs = (
        torch.randint(low=0, high=10000, size=(2, 256), dtype=torch.long),
    )
    torch.onnx.export(model, inputs, f'onnx/vqgan/vqgan2x1.onnx', output_names=['z'], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)
