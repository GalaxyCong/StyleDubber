import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock, FFTBlock_CBN, FFTBlock_CBN_encoder
from text.symbols import symbols

def get_sinusoid_encoding_table_512(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Lip_Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Lip_Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = 512
        n_layers = config["Lip_transformer"]["encoder_layer"]
        n_head = config["Lip_transformer"]["encoder_head"]
        d_k = d_v = (
            config["Lip_transformer"]["encoder_hidden"]
            // config["Lip_transformer"]["encoder_head"]
        )
        d_model = config["Lip_transformer"]["encoder_hidden"]
        d_inner = config["Lip_transformer"]["conv_filter_size"]
        kernel_size = config["Lip_transformer"]["conv_kernel_size"]
        dropout = config["Lip_transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model
        self.ln = nn.LayerNorm(512)

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table_512(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.fc_out = nn.Linear(self.d_model, 256)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        
        

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        enc_output = self.ln(src_seq) + self.position_enc[
                                                  :, :max_len, :
                                                  ].expand(batch_size, -1, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = self.fc_out(enc_output)
        return enc_output



class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        
        self.proj_life = nn.Conv1d(256, 256, kernel_size=1, padding=0, bias=False)

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq)
        else:
            enc_output = self.src_word_emb(src_seq)
        enc_output = self.proj_life(enc_output.transpose(1, 2)).transpose(1, 2)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        return enc_output





class Decoder_Condition(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder_Condition, self).__init__()
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]
        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model
        style_dim = config["CBN"]["dim"]

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock_CBN(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, style_dim, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_life = nn.Conv1d(256, 256, kernel_size=1, padding=0, bias=False)

    def forward(self, enc_seq, mask, spk=None, return_attns=False):
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq
        else:
            max_len = min(max_len, self.max_seq_len)
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] 
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]
        for dec_layer in self.layer_stack:
            if spk is not None:    
                dec_output, dec_slf_attn = dec_layer(
                    dec_output, spk, mask=mask, slf_attn_mask=slf_attn_mask
                )
                if return_attns:
                    dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask




class MelEncoder(nn.Module):
    """ Reference Mel Encoder """

    def __init__(self, config):
        super(MelEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, config["transformer"]["encoder_hidden"]).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.IN = nn.InstanceNorm1d(config["transformer"]["encoder_hidden"])


    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = src_seq + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = src_seq + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = enc_output.transpose(1, 2)
        enc_output = self.IN(enc_output)
        enc_output = enc_output.transpose(1, 2)


        return enc_output