import json
import numpy
import torch
import torch_tensorrt
from dalle import VQGanDetokenizer, DalleBartEncoder, DalleBartDecoder, TextTokenizer
from dalle import DalleBartDecoderS0, DalleBartDecoderS1, DalleBartDecoderS2
from dalle import DalleBartEncoderS0, DalleBartEncoderS1


import random
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
numpy.random.seed(42)

with open('models/vocab.json', 'r', encoding='utf8') as f:
    vocab = json.load(f)
with open('models/merges.txt', 'r', encoding='utf8') as f:
    merges = f.read().split("\n")[1:-1]

tokenizer = TextTokenizer(vocab, merges)
encoder = DalleBartEncoder(
    attention_head_count=32,
    embed_count=2048,
    glu_embed_count=4096,
    text_token_count=64,
    text_vocab_count=50272,
    layer_count=24,
    device='cuda'
).eval()
encoder.load_state_dict(torch.load('models/encoder.pt'), strict=False)
encoder0 = DalleBartEncoderS0(encoder).to(torch.float16).cuda()
encoder1 = DalleBartEncoderS1(encoder).to(torch.float16).cuda()

image_count = 1
tokens = tokenizer.tokenize('cat head with sunglasses', is_verbose=False)[:64]
tokens

text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
text_tokens[0, :2] = [tokens[0], tokens[-1]]
text_tokens[1, :len(tokens)] = tokens
text_tokens = torch.tensor(
    text_tokens, 
    dtype=torch.long, 
    device='cuda'
)

with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
    encoder_state = encoder0.forward(text_tokens)
    encoder_state = encoder1.forward(encoder_state, text_tokens)
del encoder
del encoder0
del encoder1
print('encoder: ', text_tokens.dtype, text_tokens.shape)
print('encoder: ', encoder_state.dtype, encoder_state.shape)
torch.cuda.empty_cache()

decoder = DalleBartDecoder(
    image_vocab_count=16415,
    attention_head_count=32,
    embed_count=2048,
    glu_embed_count=4096,
    layer_count=24,
    device='cuda',
).eval()
decoder.load_state_dict(torch.load('models/decoder.pt'), strict=False)
decoder0 = DalleBartDecoderS0(decoder).to(torch.float16).cuda()
decoder1 = DalleBartDecoderS1(decoder).to(torch.float16).cuda()
decoder2 = DalleBartDecoderS2(decoder).to(torch.float16).cuda()

torch.cuda.empty_cache()

with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
    expanded_indices = [0] * image_count + [1] * image_count
    text_tokens = text_tokens[expanded_indices]
    encoder_state = encoder_state[expanded_indices]
    attention_mask = text_tokens.not_equal(1).long()
    attention_state = torch.zeros(size=(24, image_count * 4, 256, 2048), device='cuda').half()
    image_tokens = torch.full((256 + 1, image_count), 16415, dtype=torch.long, device='cuda')
    torch.manual_seed(42)
    token_indices = torch.arange(256, device='cuda')
    settings = torch.tensor([1.0, 256, 16.0], dtype=torch.float16, device='cuda')

from tqdm import tqdm

for i in tqdm(range(256)):
    torch.cuda.empty_cache()
    with torch.cuda.amp.autocast(dtype=torch.float16) and torch.no_grad():
        decoder_state, attention_state[:8] = decoder0.forward(
            attention_mask=attention_mask,
            encoder_state=encoder_state,
            attention_state=attention_state[:8],
            prev_tokens=image_tokens[i],
            token_index=token_indices[[i]]
        )
        decoder_state, attention_state[8:16] = decoder1.forward(
            attention_mask=attention_mask,
            encoder_state=encoder_state,
            decoder_state=decoder_state,
            attention_state=attention_state[8:16],
            token_index=token_indices[[i]]
        )
        logits, attention_state[16:] = decoder2.forward(
            attention_mask=attention_mask,
            encoder_state=encoder_state,
            decoder_state=decoder_state,
            attention_state=attention_state[16:],
            token_index=token_indices[[i]]
        )
        if i == 0:
            print('logits: ', logits.dtype, logits.shape)
        logits = logits[:, -1, : 2 ** 14]
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]
        logits = (
            logits[:image_count] * (1 - supercondition_factor) + 
            logits[image_count:] * supercondition_factor
        )
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        logits -= logits_sorted[:, [0]]
        logits /= temperature
        logits.exp_()
        logits *= is_kept.to(torch.float32)
        image_tokens[i + 1] = torch.multinomial(logits, 1)[:, 0]
        if i == 0:
            print(settings.dtype, settings.shape)
            print(attention_mask.dtype, attention_mask.shape)
            print(encoder_state.dtype, encoder_state.shape)
            print(decoder_state.dtype, decoder_state.shape)
            print(attention_state.dtype, attention_state.shape)
            print(image_tokens[i].dtype, image_tokens[i].shape)
            print(token_indices[[i]].dtype, token_indices[[i]].shape)
            

del decoder
torch.cuda.empty_cache()

with torch.cuda.amp.autocast(dtype=torch.float32) and torch.no_grad():
    detokenizer = VQGanDetokenizer()
    detokenizer.load_state_dict(torch.load('models/detoker.pt'))
    detokenizer = detokenizer.cuda().eval()
    images = detokenizer.forward(image_tokens[1:].T)

import torchvision
grid = torchvision.utils.make_grid(images.cpu().detach().movedim(-2, -1).movedim(-3, -2), nrow=2)

import pylab as plt
plt.figure(figsize=(16,16))
plt.imshow(grid.movedim(0, -1) / 255.)
plt.axis('off')
plt.show()
