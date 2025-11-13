import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, max_len=80):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt, memory):
        b, t = tgt.shape
        positions = torch.arange(0, t, device=tgt.device).unsqueeze(0)
        tgt = self.token_embed(tgt) + self.pos_embed(positions)
        tgt = tgt.permute(1, 0, 2)
        out = self.decoder(tgt, memory)
        return self.fc(out.permute(1, 0, 2))