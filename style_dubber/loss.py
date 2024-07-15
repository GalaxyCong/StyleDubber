import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# V2C Loss
class Style_dubber_model_loss_15_Emo(nn.Module):
    """ V2C Loss """

    def __init__(self, preprocess_config, model_config):
        super(Style_dubber_model_loss_15_Emo, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.emo_ce_loss = nn.CrossEntropyLoss()
        self.weight_SpkLoss = model_config["weight"]["weight_SpkLoss"]
        self.Diar = model_config["weight"]["Dia"]
        self.weight_emo_embedding = model_config["weight"]["weight_emo_embedding"]
        self.DurationMSE = model_config["weight"]["DurationMSE"]
        self.MelLW = model_config["weight"]["MelL"]
        self.DialossW = model_config["weight"]["DialossW"]

    def forward(self, texts, speaker_targets, mel_targets, pitch_targets, energy_targets, mel_lens, duration_targets, ref_linguistic_targets, mel_predictions, postnet_mel_predictions, log_duration_predictions, src_masks, mel_masks, ref_mel_masks, speaker_predicts, spk_embedding, attn_scores, x_lengths, lip_length, lip_masks, emotion_id_embedding, emotion_id):
        mel_targets.requires_grad = False
        ref_linguistic_targets.requires_grad = False
        speaker_targets.requires_grad = False
        texts.requires_grad = False
        emotion_id.requires_grad = False
        
        """Diagonal Constraint"""
        attn_scores = attn_scores.transpose(1,2)
        attn_ks = lip_length/x_lengths
        da = lip_length/self.Diar
        diagonal_me = torch.zeros(mel_targets.shape[0], attn_scores.shape[1], attn_scores.shape[2], dtype=mel_targets.dtype, device=mel_targets.device)  # torch.Size([16, 80, 172])
        
        for i, (attn_ks_, da_, lip_length_, x_lengths_) in enumerate(zip(attn_ks, da, lip_length, x_lengths)):
            for ll in range(x_lengths_):
                y1 = int(attn_ks_*ll + da_)
                y2 = int(attn_ks_*ll - da_ if (attn_ks_*ll - da_)>0 else 0)
                diagonal_me[i,y2:y1,ll] = 1.0
        if src_masks is not None:
            attn_scores = attn_scores * (1 - src_masks.float())[:, None, :]
        if lip_masks is not None:
            attn_scores = attn_scores * (1 - lip_masks.float())[:, :, None]
        
        diagonal_attn = attn_scores * diagonal_me
        diagonal_focus_rate = diagonal_attn.sum(-1).sum(-1) / attn_scores.sum(-1).sum(-1)
        loggg = True
        diagonal_loss = self.DialossW * -diagonal_focus_rate.mean().log().float() if loggg else -diagonal_focus_rate.mean()

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        ref_mel_masks = ~ref_mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.MelLW * self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.MelLW * self.mae_loss(postnet_mel_predictions, mel_targets)
        
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        duration_loss = self.DurationMSE*self.mse_loss(log_duration_predictions, log_duration_targets).float()
        
        if emotion_id_embedding is not None:
            emos_loss = self.weight_emo_embedding*self.emo_ce_loss(emotion_id_embedding, emotion_id)
        else:
            emos_loss = 0.0
        
        if speaker_predicts is not None:
            speaker_loss = self.weight_SpkLoss*(1 - F.cosine_similarity(speaker_predicts, spk_embedding, -1, 1e-6)).float().mean()
        else:
            speaker_loss = 0.0

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + speaker_loss + emos_loss + diagonal_loss)
        
        return (
            total_loss,
            [
            total_loss.item(),
            mel_loss.item(),
            postnet_mel_loss.item(),
            speaker_loss.item(),
            duration_loss.item(),
            diagonal_loss.item(),
            emos_loss.item(),
            ],
        )





# GIRD Loss
class Style_dubber_model_loss_15_GRID(nn.Module):
    """ GRID Loss """
    def __init__(self, preprocess_config, model_config):
        super(Style_dubber_model_loss_15_GRID, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.emo_ce_loss = nn.CrossEntropyLoss()
        self.weight_SpkLoss = model_config["weight"]["weight_SpkLoss"]
        self.Diar = model_config["weight"]["Dia"]
        self.weight_emo_embedding = model_config["weight"]["weight_emo_embedding"]
        self.DurationMSE = model_config["weight"]["DurationMSE"]
        self.MelLW = model_config["weight"]["MelL"]
        self.DialossW = model_config["weight"]["DialossW"]
        
    def forward(self, texts, speaker_targets, mel_targets, pitch_targets, energy_targets, mel_lens, duration_targets, ref_linguistic_targets, mel_predictions, postnet_mel_predictions, log_duration_predictions, src_masks, mel_masks, ref_mel_masks, speaker_predicts, spk_embedding, attn_scores, x_lengths, lip_length, lip_masks):
        mel_targets.requires_grad = False
        ref_linguistic_targets.requires_grad = False
        speaker_targets.requires_grad = False
        texts.requires_grad = False
        """Diagonal Constraint"""
        attn_scores = attn_scores.transpose(1,2)
        attn_ks = lip_length/x_lengths
        da = lip_length/self.Diar
        diagonal_me = torch.zeros(mel_targets.shape[0], attn_scores.shape[1], attn_scores.shape[2], dtype=mel_targets.dtype, device=mel_targets.device) 
        for i, (attn_ks_, da_, lip_length_, x_lengths_) in enumerate(zip(attn_ks, da, lip_length, x_lengths)):
            for ll in range(x_lengths_):
                y1 = int(attn_ks_*ll + da_)
                y2 = int(attn_ks_*ll - da_ if (attn_ks_*ll - da_) >0 else 0)
                diagonal_me[i,y2:y1,ll] = 1.0
        if src_masks is not None:
            attn_scores = attn_scores * (1 - src_masks.float())[:, None, :]
        if lip_masks is not None:
            attn_scores = attn_scores * (1 - lip_masks.float())[:, :, None]
        
        diagonal_attn = attn_scores * diagonal_me
        diagonal_focus_rate = diagonal_attn.sum(-1).sum(-1) / attn_scores.sum(-1).sum(-1)
        loggg = True
        diagonal_loss = self.DialossW * -diagonal_focus_rate.mean().log().float() if loggg else -diagonal_focus_rate.mean()

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        ref_mel_masks = ~ref_mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.MelLW * self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.MelLW * self.mae_loss(postnet_mel_predictions, mel_targets)
        
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        duration_loss = self.DurationMSE*self.mse_loss(log_duration_predictions, log_duration_targets).float()
        
        if speaker_predicts is not None:
            speaker_loss = self.weight_SpkLoss*(1 - F.cosine_similarity(speaker_predicts, spk_embedding, -1, 1e-6)).float().mean()
        else:
            speaker_loss = 0.0

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + speaker_loss + diagonal_loss)

        return (
            total_loss,
            [
            total_loss.item(),
            mel_loss.item(),
            postnet_mel_loss.item(),
            speaker_loss.item(),
            duration_loss.item(),
            diagonal_loss.item(),
            0.0,
            ],
        )

