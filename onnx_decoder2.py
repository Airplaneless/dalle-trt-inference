import json
import sys
import numpy
import torch
from dalle import DalleBartDecoderS0v2, DalleBartDecoderS1v2, DalleBartDecoderS2v2, DalleBartDecoderS3v2, DalleBartDecoder, TextTokenizer

import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)

if __name__ == '__main__':
    with open('models/vocab.json', 'r', encoding='utf8') as f:
        vocab = json.load(f)
    with open('models/merges.txt', 'r', encoding='utf8') as f:
        merges = f.read().split("\n")[1:-1]

    decoder_s_id = int(sys.argv[1])

    tokenizer = TextTokenizer(vocab, merges)
    image_count = 1
    tokens = tokenizer.tokenize('cat face with sunglasses', is_verbose=False)[:64]

    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, :len(tokens)] = tokens
    text_tokens = torch.tensor(
        text_tokens, 
        dtype=torch.long, 
        device='cpu'
    )

    encoder_state = torch.randn(2, 64, 2048)
    print('encoder: ', text_tokens.dtype, text_tokens.shape)
    torch.cuda.empty_cache()

    with torch.cuda.amp.autocast(dtype=torch.float32) and torch.no_grad():
        expanded_indices = [0] * image_count + [1] * image_count
        text_tokens = text_tokens[expanded_indices]
        encoder_state = encoder_state[expanded_indices]
        attention_mask = text_tokens.not_equal(1).long()
        attention_state = torch.zeros(size=(24, image_count * 4, 256, 2048), device='cpu')
        image_tokens = torch.full((256 + 1, image_count), 16415, dtype=torch.long, device='cpu')
        torch.manual_seed(0)
        token_indices = torch.arange(256, device='cpu')
        settings = torch.tensor([1.0, 256, 16.0], dtype=torch.float32, device='cpu')

    decoder = DalleBartDecoder(
        image_vocab_count=16415,
        attention_head_count=32,
        embed_count=2048,
        glu_embed_count=4096,
        layer_count=24,
        device='cpu',
    ).eval()
    decoder.load_state_dict(torch.load('models/decoder.pt'), strict=False)
    if decoder_s_id == 0:
        decoder_s = DalleBartDecoderS0v2(decoder)
    elif decoder_s_id == 1:
        decoder_s = DalleBartDecoderS1v2(decoder)
    elif decoder_s_id == 2:
        decoder_s = DalleBartDecoderS2v2(decoder)
    elif decoder_s_id == 3:
        decoder_s = DalleBartDecoderS3v2(decoder)
    else:
        raise ValueError
    
    for p in decoder_s.parameters():
        p.requires_grad = False

    if decoder_s_id == 0:
        inputs = (
            attention_mask,
            encoder_state,
            attention_state[:6],
            image_tokens[0],
            token_indices[[0]],
        )
    elif decoder_s_id == 1:
        inputs = (
            attention_mask,
            encoder_state,
            encoder_state[:, :1, :],
            attention_state[6:12],
            token_indices[[0]],
        )
    elif decoder_s_id == 2:
        inputs = (
            attention_mask,
            encoder_state,
            encoder_state[:, :1, :],
            attention_state[12:18],
            token_indices[[0]],
        )
    elif decoder_s_id == 3:
        inputs = (
            attention_mask,
            encoder_state,
            encoder_state[:, :1, :],
            attention_state[18:],
            token_indices[[0]],
        )
    else:
        raise ValueError
    torch.onnx.export(decoder_s, inputs, f'onnx/decoder{decoder_s_id}v2/decoder{decoder_s_id}.onnx', output_names=["image_tokens", "attention_state"], verbose=True, opset_version=15, export_params=True, do_constant_folding=True)
