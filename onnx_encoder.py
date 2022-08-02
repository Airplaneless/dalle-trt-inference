import json
import sys
import numpy
import torch

from dalle import VQGanDetokenizer, DalleBartEncoder, DalleBartDecoder, TextTokenizer
from dalle import DalleBartEncoderS0, DalleBartEncoderS1

import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)

if __name__ == '__main__':
    decoder_s_id = int(sys.argv[1])
    encoder = DalleBartEncoder(
        attention_head_count=32,
        embed_count=2048,
        glu_embed_count=4096,
        text_token_count=64,
        text_vocab_count=50272,
        layer_count=24,
        device='cpu'
    ).eval()
    encoder.load_state_dict(torch.load('models/encoder.pt'), strict=False)
    if decoder_s_id == 0:
        encoder_s = DalleBartEncoderS0(encoder)
    else:
        encoder_s = DalleBartEncoderS1(encoder)
    for p in encoder_s.parameters():
        p.requires_grad = False
    if decoder_s_id == 0:
        inputs = (
            torch.randint(low=0, high=100, size=(2, 64)),
        )
    else:
        inputs = (
            torch.randn(2, 64, 2048),
            torch.randint(low=0, high=100, size=(2, 64)),
        )
    torch.onnx.export(encoder_s, inputs, f'onnx/encoder{decoder_s_id}/encoder{decoder_s_id}.onnx', output_names=['encoder_state'], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)
