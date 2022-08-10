#!/usr/bin/python
import os
import sys
import argparse
import numpy
import pylab as plt
import json
import torch
import torchvision
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
import common
import onnxruntime
onnxruntime.disable_telemetry_events()
from dalle import TextTokenizer
from utils_sr import *
from PIL import Image
from tqdm import tqdm


class HostDeviceMem(object):
    def __init__(self, host_mem):
        self.host = host_mem
        self.device = cuda.mem_alloc(host_mem.nbytes)
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

def to_numpy(tensor):
    if isinstance(tensor, numpy.ndarray):
        return tensor
    else:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_engine_info(engine0):
    for b in engine0:
        print(b, trt.nptype(engine0.get_binding_dtype(b)), trt.volume(engine0.get_binding_shape(b)) * engine0.max_batch_size)


if __name__ == '__main__':
    import random
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    numpy.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text',
        type=str,
        required=True
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=2048
    )
    parser.add_argument(
        '--topp',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--sfactor',
        type=float,
        default=32.0
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='./renders'
    )
    parser.add_argument(
        '--srgan',
        action='store_true',
    )
    args = parser.parse_args()
    print(args)
    TEXT = args.text
    TEMPERATURE = args.temperature
    TOPK = args.topk
    TOPP = args.topp
    SFACTOR = args.sfactor
    SEED = args.seed
    DIR = args.dir
    DIR = os.path.abspath(DIR)
    DIR = os.path.join(DIR, '_'.join(TEXT.split(' ')))
    assert SEED >= 0
    os.makedirs(DIR, exist_ok=True)
    with open('models/vocab.json', 'r', encoding='utf8') as f:
        vocab = json.load(f)
    with open('models/merges.txt', 'r', encoding='utf8') as f:
        merges = f.read().split("\n")[1:-1]
    ort_session0 = onnxruntime.InferenceSession('./onnx/encoder0/encoder0.onnx', providers=['CPUExecutionProvider'])
    ort_session1 = onnxruntime.InferenceSession('./onnx/encoder1/encoder1.onnx', providers=['CPUExecutionProvider'])
    ort_session2 = onnxruntime.InferenceSession(f'engines/vqgan2x1.onnx', providers=['CPUExecutionProvider'])
    tokenizer = TextTokenizer(vocab, merges)
    runtime = trt.Runtime(TRT_LOGGER)
    stream = cuda.Stream()
    with open("engines/decoder0.trt32", mode="rb") as f:
        engine0 = runtime.deserialize_cuda_engine(f.read())
        context0 = engine0.create_execution_context()
    with open("engines/decoder1.trt", mode="rb") as f:
        engine1 = runtime.deserialize_cuda_engine(f.read())
        context1 = engine1.create_execution_context()
    with open("engines/decoder2.trt", mode="rb") as f:
        engine2 = runtime.deserialize_cuda_engine(f.read())
        context2 = engine2.create_execution_context()
    tokens = tokenizer.tokenize(TEXT, is_verbose=False)[:64]
    image_count = 1
    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, :len(tokens)] = tokens
    text_tokens = torch.tensor(
        text_tokens, 
        dtype=torch.long, 
    )
    ort_inputs = {ort_session0.get_inputs()[0].name: to_numpy(text_tokens)}
    ort_outs = ort_session0.run(None, ort_inputs)
    ort_inputs = {ort_session1.get_inputs()[0].name: to_numpy(ort_outs[0]), ort_session1.get_inputs()[1].name: to_numpy(text_tokens)}
    ort_outs = ort_session1.run(None, ort_inputs)
    encoder_state = torch.from_numpy(ort_outs[0])
    seed_add = 0
    tAM = HostDeviceMem(cuda.pagelocked_empty(128 * image_count, numpy.int32))
    tES = HostDeviceMem(cuda.pagelocked_empty(262144 * image_count, numpy.float32))
    tIT = HostDeviceMem(cuda.pagelocked_empty(1 * image_count, numpy.int32))
    tTI = HostDeviceMem(cuda.pagelocked_empty(1, numpy.int32))
    tDS = HostDeviceMem(cuda.pagelocked_empty(4096 * image_count, numpy.float32))
    tOut = HostDeviceMem(cuda.pagelocked_empty(32832 * image_count, numpy.float32))
    tAS0 = HostDeviceMem(cuda.pagelocked_empty(16777216 * image_count, numpy.float32))
    tAS1 = HostDeviceMem(cuda.pagelocked_empty(16777216 * image_count, numpy.float32))
    tAS2 = HostDeviceMem(cuda.pagelocked_empty(16777216 * image_count, numpy.float32))
    WINDOW = list(range(256))
    WIDTH = 512
    WSIZE1 = 64
    WSIZE2 = 64
    try:
        while True:
            expanded_indices = [0] * image_count + [1] * image_count
            text_tokens = text_tokens[expanded_indices]
            encoder_state = encoder_state[expanded_indices]
            attention_mask = text_tokens.not_equal(1).long()
            attention_state = torch.zeros(size=(24, image_count * 4, 256, 2048))
            image_tokens = torch.full((WIDTH + 1, image_count), 16415, dtype=torch.long)
            if SEED > 0: torch.manual_seed(SEED + seed_add)
            token_indices = torch.arange(256)
            settings = torch.tensor([TEMPERATURE, TOPK, SFACTOR])
            numpy.copyto(tAM.host, to_numpy(attention_mask).ravel())
            numpy.copyto(tES.host, to_numpy(encoder_state).ravel())
            numpy.copyto(tAS0.host, to_numpy(attention_state[:8]).ravel())
            numpy.copyto(tAS1.host, to_numpy(attention_state[8:16]).ravel())
            numpy.copyto(tAS2.host, to_numpy(attention_state[16:]).ravel())
            cuda.memcpy_htod_async(tAM.device, tAM.host, stream)
            cuda.memcpy_htod_async(tES.device, tES.host, stream)
            cuda.memcpy_htod_async(tAS0.device, tAS0.host, stream)
            cuda.memcpy_htod_async(tAS1.device, tAS1.host, stream)
            cuda.memcpy_htod_async(tAS2.device, tAS2.host, stream)
            for i in tqdm(range(WIDTH)):
                if i < WSIZE1:
                    ti = i
                else:
                    if i >= WIDTH - WSIZE2:
                        ti = i % 256
                    else:
                        ti = WINDOW[WSIZE1:-WSIZE2][(i - WSIZE1) % len(WINDOW[WSIZE1:-WSIZE2])]
                numpy.copyto(tIT.host, to_numpy(image_tokens[i]).ravel())
                numpy.copyto(tTI.host, to_numpy(token_indices[[ti]]).ravel())
                cuda.memcpy_htod_async(tIT.device, tIT.host, stream)
                cuda.memcpy_htod_async(tTI.device, tTI.host, stream)
                queue = [tAM, tES, tAS0, tIT, tTI, tDS, tAS0]
                context0.execute_async_v2(bindings=[v.device for v in queue], stream_handle=stream.handle)
                queue = [tAM, tES, tDS, tAS1, tTI, tDS, tAS1]
                context1.execute_async_v2(bindings=[v.device for v in queue], stream_handle=stream.handle)
                queue = [tAM, tES, tDS, tAS2, tTI, tAS2, tOut]
                context2.execute_async_v2(bindings=[v.device for v in queue], stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(tOut.host, tOut.device, stream)
                stream.synchronize()
                logits = torch.from_numpy(tOut.host).reshape(2, 1, 16416)
                logits = logits[:, -1, : 2 ** 14]
                temperature = settings[[0]]
                top_k = settings[[1]].to(torch.long)
                supercondition_factor = settings[[2]]
                logits = logits[:image_count] * (1 - supercondition_factor) + logits[image_count:] * supercondition_factor
                logits = logits[0]
                _, pindices = logits.topk(logits.shape[0] - TOPK, largest=False)
                probas = logits.softmax(-1)
                min_val = min(probas.max().item(), TOPP)
                logits[pindices] = 0
                logits[probas < min_val] = 0
                image_tokens[i + 1] = torch.multinomial(logits.softmax(-1), 1)[0]
            ort_inputs = {ort_session2.get_inputs()[0].name: to_numpy(image_tokens[1:].T.reshape(-1, 256))}
            image = ort_session2.run(None, ort_inputs)[0]
            image_path = os.path.join(DIR, f'image_{SEED + seed_add}_v.png')
            Image.fromarray((image[0]).astype(numpy.uint8)).save(image_path)
            seed_add += 1
    except KeyboardInterrupt:
        print(f'\nGeneration Interrupted, running {"ESRGAN" if args.srgan else "Real-ESRGAN"}')
        del engine0, engine1, engine2, context0, context1, context2, stream
        stream = cuda.Stream()
        if not args.srgan:
            with open(f"engines/esrgan1x2.trt", mode="rb") as f:
                engine3 = runtime.deserialize_cuda_engine(f.read())
                context3 = engine3.create_execution_context()
        else:
            with open(f"engines/srgan1x2.trt", mode="rb") as f:
                engine3 = runtime.deserialize_cuda_engine(f.read())
                context3 = engine3.create_execution_context()  
        inputs, outputs, bindings = common.allocate_buffers(engine3)
        sr_dir = 'esrgan' if args.srgan else 'real-esrgan'
        os.makedirs(os.path.join(DIR, sr_dir), exist_ok=True)
        for fname in os.listdir(DIR):
            if fname.startswith('image_'):
                if not args.srgan:
                    lr_image = Image.open(os.path.join(DIR, fname)).convert('RGB')
                    lr_image = numpy.array(lr_image).transpose((1,0,2))
                    lr_image = pad_reflect(lr_image, 15)
                    patches, p_shape = split_image_into_overlapping_patches(lr_image, patch_size=192, padding_size=24)
                    # 0,1,2,3 -> 0,2,1,3 -> 0,3,1,2
                    patches = numpy.swapaxes(numpy.swapaxes(patches, 1, 2), 1, 3)
                    numpy.copyto(inputs[0].host, patches.ravel() / 255.)
                    res = common.do_inference_v2(context3, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0]
                    np_sr_image = res.reshape(6, 1920, 1920, 3)
                    padded_size_scaled = tuple(numpy.multiply(p_shape[0:2], 8)) + (3,)
                    scaled_image_shape = tuple(numpy.multiply(lr_image.shape[0:2], 8)) + (3,)
                    np_sr_image = stich_together(np_sr_image, padded_image_shape=padded_size_scaled, target_shape=scaled_image_shape, padding_size=24 * 8)
                    sr_img = (np_sr_image*255).astype(numpy.uint8)
                    sr_img = unpad_image(sr_img, 15*8).transpose((1,0,2))
                    print(fname)
                    Image.fromarray(sr_img).save(os.path.join(DIR, sr_dir, fname))
                else:
                    lr_image = Image.open(os.path.join(DIR, fname)).convert('RGB')
                    lr_image = numpy.array(lr_image).transpose((1,0,2))
                    numpy.copyto(inputs[0].host, numpy.moveaxis(lr_image[None], -1, 1).ravel() / 255.)
                    res = common.do_inference_v2(context3, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0]
                    np_sr_image = numpy.moveaxis(res.reshape(3, 256*4, -1), 0, -1).clip(0, 1)
                    np_sr_image = numpy.moveaxis(np_sr_image, 0, 1)
                    sr_img = (np_sr_image*255).astype(np.uint8)
                    print(fname)
                    Image.fromarray(sr_img).save(os.path.join(DIR, sr_dir, fname))
