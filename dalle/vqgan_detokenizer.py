import torch
from torch import nn
from torch import FloatTensor, LongTensor
from math import sqrt


class ResnetBlock(nn.Module):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2 ** 5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2 ** 5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        h = x
        h = self.norm1.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        n = 2 ** 9
        self.norm = nn.GroupNorm(2 ** 5, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x: FloatTensor, t1: int, t2: int) -> FloatTensor:
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        h = h.reshape(m, n, t1, t2)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h: FloatTensor, t1: int, t2: int) -> FloatTensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h, t1, t2)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self, 
        log2_count_in: int, 
        log2_count_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        
        self.block = nn.ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out)
        ])

        if has_attention:
            self.attn = nn.ModuleList([
                AttentionBlock(),
                AttentionBlock(),
                AttentionBlock()
            ])

        if has_upsample:
            self.upsample = Upsample(log2_count_out)


    def forward(self, h: FloatTensor, t1: int, t2: int) -> FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h, t1, t2)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(2 ** 8, 2 ** 9, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = nn.ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = nn.GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = nn.Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: FloatTensor, t1: int, t2: int) -> FloatTensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z, t1, t2)

        for i in reversed(range(5)):
            z = self.up[i].forward(z, t1, t2)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.embedding = nn.Embedding(vocab_count, embed_count)
        self.post_quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()
        self.dx = 1
        self.dy = 1

    def forward(self, z: LongTensor) -> FloatTensor:
        z.clamp_(0, self.vocab_count - 1)
        t1 = self.dx * 2 ** 4
        t2 = self.dy * 2 ** 4
        z = z.view([1, 1, 2 ** 4, 2 ** 4])
        z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
        z = z.flatten().unsqueeze(1)
        z = self.embedding.forward(z)
        z = z.repeat(self.dx,1,self.dy)
        z = z.view((1, t1, t2, 2 ** 8))
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z, t1, t2)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255
        return z


class VQGanDetokenizerB(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        self.embedding = nn.Embedding(vocab_count, embed_count)
        self.post_quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()
        self.dx = 1
        self.dy = 1

    def forward(self, z: LongTensor) -> FloatTensor:
        z.clamp_(0, self.vocab_count - 1)
        t1 = self.dx * 2 ** 4
        t2 = self.dy * 2 ** 4
        z = self.embedding.forward(z)
        z = z.view((z.shape[0], t1, t2, 2 ** 8))
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z, t1, t2)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255
        return z
