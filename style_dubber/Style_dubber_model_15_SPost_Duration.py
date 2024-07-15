"""
Note: 
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Decoder_Condition, PostNet, Lip_Encoder
from .modules import VarianceAdaptor_softplus1
from utils.tools import get_mask_from_lengths
from MMAttention.sma import StepwiseMonotonicMultiheadAttention

class Style_dubber_model_15_SPost_Duration(nn.Module):
    """ Style_dubber_model """
    def __init__(self, preprocess_config, model_config):
        super(Style_dubber_model_15_SPost_Duration, self).__init__()
        self.model_config = model_config
        # There is no pitch and energy predicting in VarianceAdaptor_softplus1
        self.variance_adaptor = VarianceAdaptor_softplus1(preprocess_config, model_config)
        self.decoder_Condition = Decoder_Condition(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],)
        self.postnet = PostNet() # Style, mel
        self.lip_encoder = Lip_Encoder(model_config)
        # we set the "is_tunable = False", it equals traditional multi-head attention. See https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention
        self.attn_lip_text  = StepwiseMonotonicMultiheadAttention(256, 256//8, 256//8)  # multihead == 8
        
    def parse_batch(self, batch):
        id_basename = batch["id"]
        emotion_ids = torch.from_numpy(batch["emotion_ids"]).long().cuda()
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        emos_embedding = torch.from_numpy(batch["emos_embedding"]).float().cuda()
        return id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_ids

    def parse_batch_Setting3(self, batch):
        id_basename = batch["id"]
        zeroref_basename = batch["zerorefs"]
        emotion_ids = torch.from_numpy(batch["emotion_ids"]).long().cuda()
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        emos_embedding = torch.from_numpy(batch["emos_embedding"]).float().cuda()
        return id_basename, zeroref_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_ids

    def parse_batch_GRID(self, batch):
        id_basename = batch["id"]
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        return id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding


    def parse_batch_Setting3_GRID(self, batch):
        id_basename = batch["id"]
        zeroref_basename = batch["zerorefs"]
        speakers = torch.from_numpy(batch["sid"]).long().cuda()
        text = torch.from_numpy(batch["text"]).long().cuda()
        ref_linguistics = torch.from_numpy(batch["ref_linguistics"]).long().cuda()
        mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
        ref_mels = torch.from_numpy(batch["ref_mels"]).float().cuda()
        durations = torch.from_numpy(batch["D"]).long().cuda()
        pitches = torch.from_numpy(batch["f0"]).float().cuda()
        energies = torch.from_numpy(batch["energy"]).float().cuda()
        src_len = torch.from_numpy(batch["src_len"]).long().cuda()
        mel_lens = torch.from_numpy(batch["mel_len"]).long().cuda()
        ref_mel_lens = torch.from_numpy(batch["ref_mel_lens"]).long().cuda()
        max_src_len = np.max(batch["src_len"]).astype(np.int32)
        max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
        lip_embedding = torch.from_numpy(batch["Lipmotion"]).float().cuda()
        face_embedding = torch.from_numpy(batch["face_embedding"]).float().cuda()
        face_lens = torch.from_numpy(batch["face_lens"]).long().cuda()
        MaxfaceL = np.max(batch["face_lens"]).astype(np.int32)
        spk_embedding = torch.from_numpy(batch["spk_embedding"]).float().cuda()
        return id_basename, zeroref_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding

    def forward(
        self,
        output, 
        text_encoder,
        src_masks,
        ref_mels,
        ref_mel_lens,
        face_lens,
        max_face_lens,
        lip_embedding,
        spk_embedding,
        mel_lens=None,
        max_mel_len=None,
        d_targets=None,
        p_targets=None,
        e_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        max_ref_mel_lens = ref_mels.shape[1]
        ref_mel_masks = get_mask_from_lengths(ref_mel_lens, max_ref_mel_lens)

        lip_masks = get_mask_from_lengths(face_lens, max_face_lens)
        lip_embedding = self.lip_encoder(lip_embedding, lip_masks)
        
        if src_masks is not None:
            slf_attn_mask_text = src_masks.unsqueeze(1).expand(-1, max_face_lens, -1)
            slf_attn_mask_lip = lip_masks.unsqueeze(1).expand(-1, src_masks.size(1), -1)
            slf_attn_mask = slf_attn_mask_text.transpose(1,2) | slf_attn_mask_lip
        
        output_text_lip, AV_attn, _ =self.attn_lip_text(text_encoder, lip_embedding, lip_embedding, face_lens, mask=slf_attn_mask, query_mask=src_masks.unsqueeze(2))
        
        (
            output,
            log_d_predictions,
            d_rounded_pred,
            mel_lens_pred,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            output_text_lip,
            src_masks,
            mel_masks,
            max_mel_len,
            mel_lens,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        """ self.decoder_Condition
        """
        output, mel_masks = self.decoder_Condition(output, mel_masks, spk_embedding)
        output = self.mel_linear(output)

        """ Style_Postnet
        """
        postnet_output = self.postnet(output, spk_embedding) + output

        if d_targets is not None:
            return (
                output,
                postnet_output,
                log_d_predictions,
                d_rounded_pred,
                src_masks,
                mel_masks,
                mel_lens,
                ref_mel_masks,
                AV_attn,
                lip_masks,
            )
        else:
            return postnet_output, mel_lens_pred
        

