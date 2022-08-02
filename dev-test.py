import json
import numpy
import torch
import torch_tensorrt
import torchvision
torch_tensorrt.logging.set_reportable_log_level(torch_tensorrt.logging.Level.Error)

from dalle import VQGanDetokenizer, DalleBartEncoder, DalleBartDecoder, TextTokenizer

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
    tokenizer = TextTokenizer(vocab, merges)
    image_count = 4
    tokens = tokenizer.tokenize('cat face', is_verbose=False)[:64]

    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, :len(tokens)] = tokens
    text_tokens = torch.tensor(
        text_tokens, 
        dtype=torch.int32, 
        device='cuda'
    )
    attention_mask = text_tokens.not_equal(1).float()

    trt_encoder0 = torch.jit.load("encoder0.ts")
    trt_encoder1 = torch.jit.load("encoder1.ts")

    encoder_state = trt_encoder0(text_tokens.to(torch.int32), attention_mask.half())
    encoder_state = trt_encoder1(encoder_state.half(), attention_mask.half())

    del trt_encoder0
    del trt_encoder1

    trt_decoder0 = torch.jit.load("decoder0.ts")
    trt_decoder1 = torch.jit.load("decoder1.ts")
    trt_decoder2 = torch.jit.load("decoder2.ts")

    with torch.cuda.amp.autocast(dtype=torch.float32) and torch.no_grad():
        expanded_indices = [0] * image_count + [1] * image_count
        text_tokens = text_tokens[expanded_indices]
        encoder_state = encoder_state[expanded_indices]
        attention_mask = text_tokens.not_equal(1)
        attention_state = torch.zeros(size=(24, image_count * 4, 256, 2048))
        image_tokens = torch.full((256 + 1, image_count), 16415, dtype=torch.long)
        torch.manual_seed(0)
        token_indices = torch.arange(256)
        settings = torch.tensor([1.0, 256, 16.0], dtype=torch.float16, device='cuda')
    torch.cuda.empty_cache()
    for i in range(256):
        token_index_batched = token_indices[[i]][[0] * image_count * 2]
        prev_tokens = image_tokens[i][list(range(image_count)) * 2]
        prev_tokens.clamp_(0, 16415)
        token_mask = torch.zeros(16, 256, 2048)
        token_mask[:, token_index_batched[0]] = 1
        token_indices = torch.arange(-1, 255)
        self_attn_mask = (token_indices < token_index_batched[0][None])
        self_attn_mask = self_attn_mask.repeat(encoder_state.shape[0],1)
        decoder_state, attention_state = trt_decoder0(attention_mask.half().cuda(), encoder_state.half().cuda(), attention_state.half().cuda(), prev_tokens.to(torch.int32).cuda(), token_index_batched.to(torch.int32).cuda(), token_mask.half().cuda(), self_attn_mask.half().cuda())
        decoder_state, attention_state = trt_decoder1(attention_mask.half().cuda(), encoder_state.half().cuda(), decoder_state.half().cuda(), attention_state.half().cuda(), token_index_batched.to(torch.int32).cuda(), token_mask.half().cuda(), self_attn_mask.half().cuda())
        logits, attention_state = trt_decoder2(attention_mask.half().cuda(), encoder_state.half().cuda(), decoder_state.half().cuda(), attention_state.half().cuda(), token_index_batched.to(torch.int32).cuda(), token_mask.half().cuda(), self_attn_mask.half().cuda())
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]
        logits = logits[:, -1, : 2 ** 14]
        logits = (
            logits[:image_count] * (1 - supercondition_factor) + 
            logits[image_count:] * supercondition_factor
        )
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = (logits >= logits_sorted[:, top_k - 1]).to(decoder_state.dtype)
        logits -= logits_sorted[:, [0]]
        logits /= temperature
        logits.exp_()
        logits *= is_kept
        image_tokens[i + 1] = torch.multinomial(logits, 1)[:, 0]
    del trt_decoder0, trt_decoder1, trt_decoder2
    with torch.cuda.amp.autocast(dtype=torch.float32) and torch.no_grad():
        detokenizer = VQGanDetokenizer()
        detokenizer.load_state_dict(torch.load('models/detoker.pt'))
        detokenizer = detokenizer.cuda().eval()
        images = detokenizer.forward(image_tokens[1:].T)
    grid = torchvision.utils.make_grid(images.cpu().detach().movedim(-2, -1).movedim(-3, -2), nrow=2)
    import pylab as plt
    plt.figure(figsize=(16,16))
    plt.imshow(grid.movedim(0, -1) / 255.)
    plt.axis('off')
    plt.show()
    
