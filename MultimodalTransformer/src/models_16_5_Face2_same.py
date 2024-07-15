import torch
from torch import nn
from inspect import getsourcefile
import torch.nn.functional as F
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
sys.path.append('/modules')
from modules.transformer import TransformerEncoder
from transformer import Encoder
from utils.tools import get_mask_from_lengths
from text.symbols import symbols
from style_dubber.modules import AdversarialClassifier
import os
import json
from stylespeech.Modules import Mish, LinearNorm, Conv1dGLU, MultiHeadAttention


class MULTModel(nn.Module):
    def __init__(self, hyp_params, model_config, preprocess_config):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        
        self.emoID_classifier = AdversarialClassifier(
            in_dim=model_config["downsample_encoder"]["out_dim"],
            out_dim=8,
            hidden_dims=model_config["classifier"]["cls_hidden"]
        )
        
        self.style_encoder = MelStyleEncoder(model_config)

        self.ds_speaker_encoder = DownsampleEncoder(
            in_dim=model_config["frame_encoder"]["out_dim"],
            conv_channels=model_config["downsample_encoder"]["conv_filters"],
            kernel_size=model_config["downsample_encoder"]["kernel_size"],
            stride=model_config["downsample_encoder"]["stride"],
            padding=model_config["downsample_encoder"]["padding"],
            dropout=model_config["downsample_encoder"]["dropout"],
            pooling_sizes=model_config["downsample_encoder"]["pooling_sizes"],
            out_dim=model_config["downsample_encoder"]["out_dim"],
        )
        
        self.V_encoder = DownsampleEncoder(
            in_dim=model_config["frame_encoder"]["out_dim"],
            conv_channels=model_config["V_downsample_encoder"]["conv_filters"],
            kernel_size=model_config["V_downsample_encoder"]["kernel_size"],
            stride=model_config["V_downsample_encoder"]["stride"],
            padding=model_config["V_downsample_encoder"]["padding"],
            dropout=model_config["V_downsample_encoder"]["dropout"],
            pooling_sizes=model_config["V_downsample_encoder"]["pooling_sizes"],
            out_dim=model_config["V_downsample_encoder"]["out_dim"],
        )

        self.ds_times = 1
        for i in model_config["downsample_encoder"]["pooling_sizes"]:
            self.ds_times *= i 
            
            
        self.ds_times_V = 1
        for i in model_config["V_downsample_encoder"]["pooling_sizes"]:
            self.ds_times_V *= i 

        # If using speaker classification loss after local speaker embeddings
        self.use_spkcls = model_config["use_spkcls"]
        if self.use_spkcls:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_classifier = AdversarialClassifier(
                in_dim=model_config["downsample_encoder"]["out_dim"],
                out_dim=n_speaker,
                hidden_dims=model_config["classifier"]["cls_hidden"]
            )
        #######################
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params['orig_d_l'], hyp_params['orig_d_a'], hyp_params['orig_d_v']
        self.d_l, self.d_a, self.d_v = 128, 128, 128
        self.vonly = True
        self.aonly = True
        self.lonly = True
        self.num_heads = hyp_params['num_heads']
        self.layers = hyp_params['layers']
        self.attn_dropout = hyp_params['attn_dropout']
        self.attn_dropout_a = hyp_params['attn_dropout_a']
        self.attn_dropout_v = hyp_params['attn_dropout_v']
        self.relu_dropout = hyp_params['relu_dropout']
        self.res_dropout = hyp_params['res_dropout']
        self.out_dropout = hyp_params['out_dropout']
        self.embed_dropout = hyp_params['embed_dropout']
        self.attn_mask = hyp_params['attn_mask']
        self.encoder = Encoder(model_config)

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, 1)
        
        self.lin = nn.Linear(180, 256)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out
         
    def forward(self, x_l, x_a, src_lens, max_src_len, ref_mels, ref_mel_lens, face_lens, MaxfaceL):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        max_ref_mel_lens = ref_mels.shape[1]
        ref_mel_masks = get_mask_from_lengths(ref_mel_lens, max_ref_mel_lens)
        ref_local_speaker_emb = self.style_encoder(ref_mels, ref_mel_masks)
        ref_local_speaker_emb = self.ds_speaker_encoder(ref_local_speaker_emb)
        if self.use_spkcls:
            ref_local_lens = ref_mel_lens // self.ds_times
            ref_local_lens[ref_local_lens == 0] = 1
            max_ref_local_lens = max_ref_mel_lens // self.ds_times
            ref_local_spk_masks = (1 - get_mask_from_lengths(ref_local_lens, max_ref_local_lens).float()).unsqueeze(-1).expand(-1, -1, 256)
            spkemb = torch.sum(ref_local_speaker_emb * ref_local_spk_masks, axis=1) / ref_local_lens.unsqueeze(-1).expand(-1, 256)
        else:
            speaker_predicts = None
            spkemb = None
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        x_l = self.encoder(x_l, src_masks)
        text_encoder = x_l
        x_v = ref_local_speaker_emb
        x_a = self.V_encoder(x_a)
        ref_face_lens = face_lens // self.ds_times_V
        ref_face_lens[ref_face_lens == 0] = 1
        max_ref_face_lens = MaxfaceL // self.ds_times_V
        ref_local_face_masks = (1 - get_mask_from_lengths(ref_face_lens, max_ref_face_lens).float()).unsqueeze(-1).expand(-1, -1, 256)
        emotion_id_embedding = torch.sum(x_a * ref_local_face_masks, axis=1) / ref_face_lens.unsqueeze(-1).expand(-1, 256)
        emotion_id_embedding = self.emoID_classifier(emotion_id_embedding, is_reversal=False)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)  
        x_v = x_v.transpose(1, 2)  
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)  # ----> [Len, BN, Dm]
        proj_x_v = proj_x_v.permute(2, 0, 1)  # ----> [Len, BN, Dm]
        proj_x_l = proj_x_l.permute(2, 0, 1)  # ----> [Len, BN, Dm]
        if self.lonly:
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)   
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)  
            h_ls = self.trans_l_mem(h_ls).transpose(0, 1)         
        return (h_ls + text_encoder)/2, src_masks, spkemb, emotion_id_embedding, text_encoder



class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''

    def __init__(self, model_config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = model_config["Stylespeech"]["n_mel_channels"]
        self.hidden_dim = model_config["Stylespeech"]["style_hidden"]
        self.out_dim = model_config["Stylespeech"]["style_vector_dim"]
        self.kernel_size = model_config["Stylespeech"]["style_kernel_size"]
        self.n_head = model_config["Stylespeech"]["style_head"]
        self.dropout = model_config["Stylespeech"]["dropout"]

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim,
                                           self.hidden_dim // self.n_head, self.hidden_dim // self.n_head, self.dropout)

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        content, _ = self.slf_attn(x, mask=slf_attn_mask)
        content = self.fc(content)
        return content




class DownsampleEncoder(nn.Module):
    def __init__(self, in_dim=256, conv_channels=[256, 256, 256, 256], kernel_size=3, stride=1, padding=1, dropout=0.2, pooling_sizes=[2, 2, 2, 2], out_dim=256):
        super(DownsampleEncoder, self).__init__()
        K = len(conv_channels)
        filters = [in_dim] + conv_channels
        self.conv1ds = nn.ModuleList(
            [nn.Conv1d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features=conv_channels[i])
             for i in range(K)])
        self.pools = nn.ModuleList(
            [nn.AvgPool1d(kernel_size=pooling_sizes[i]) for i in range(K)]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.local_outlayer = nn.Sequential(
            nn.Linear(in_features=conv_channels[-1],
                      out_features=out_dim),
            nn.Tanh())
    def forward(self, inputs):
        out = inputs.transpose(1, 2)
        for conv, bn, pool in zip(self.conv1ds, self.bns, self.pools):
            out = conv(out)    
            out = self.relu(out)  
            out = bn(out) 
            out = self.dropout(out)
            out = pool(out)

        out = out.transpose(1, 2)  
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1) 

        local_output = self.local_outlayer(out)
        return local_output



