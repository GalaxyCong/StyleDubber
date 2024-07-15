import os
import json

import torch
import numpy as np

import hifigan
# from style_dubber import Style_dubber_model, ScheduledOptim, Style_dubber_model_NoDownSample, Style_dubber_model_change, Style_dubber_model_SAP1, Style_dubber_model_SAP2, Style_dubber_model_styleEn, Style_dubber_model_styleEn_Down, Style_dubber_model_Encoderstyle, Style_dubber_model_CTCEncoder, Style_dubber_model_Monotonic, Style_dubber_model_Dia, Style_dubber_model_Dia_LipCTC, Style_dubber_model_Dia_AttenCTC, Style_dubber_model_Dia_LipCTC_SOFTPLUS, Style_dubber_model_Dia_softplus, Style_dubber_model_Dia_softplus_P, Style_dubber_model_Dia_softplus_MoreAttention, Style_dubber_model_Dia_softplus_MoreAttention2, Model12_Style_Dubber, Model12_Style_Dubber_CTC, Model12_Style_Dubber_CTC_1, Style_dubber_model_13, Style_dubber_model_14, Style_dubber_model_15, Style_dubber_model_15_SPost, Style_dubber_model_15_SPost_Duration, Style_dubber_model_15_SPost_Duration_Ab2_USL, Style_dubber_model_15_SPost_Duration_AB3_PLA, Style_dubber_model_15_SPost_Duration_AB0, Style_dubber_model_15_SPost_Duration_Ab8_MelDecoder, Style_dubber_model_15_SPost_Duration_Ab9_Post
from style_dubber import ScheduledOptim, Style_dubber_model_15_SPost_Duration
from utils.env import AttrDict
from utils.hifigan_16_models import Generator


from MultimodalTransformer.src import models, models_16_5_Face2_same
def get_model_15_Style_Final1_Single_NoVisual_SPost(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost(preprocess_config, model_config).to(device)
    fusion_model = models_15_5_noVisual.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model




def get_model_15_Style_Final1_Single_NoVisual(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_15_5_noVisual.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_13_Style_Final1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_13(preprocess_config, model_config).to(device)
    fusion_model = models.MULTModel(model_config["Multimodal-Transformer"], model_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_14_Style_Final1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_14(preprocess_config, model_config).to(device)
    fusion_model = models.MULTModel(model_config["Multimodal-Transformer"], model_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_15_Style_Final1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_15.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_15_Style_Final1_Re(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_15_5.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model




def get_model_16_Style_Final1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_16.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_16_Style_Final1_Single(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_16_5.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_16_Style_Final1_Single_Face(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_16_Style_Final1_Single_Face2(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_16_Style_Final1_Single_NoVisual(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_noVisual.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_16_Style_Final1_Single_NoVisual_SPost(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_noVisual.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_ID20_E1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model





def get_model_ID22_E1_Duration(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    """
        Paper: https://arxiv.org/pdf/2402.12636
        
        fusion_model = models_16_5_Face2_same.MULTModel ===========> Paper Figure 2, a) Multimodal Phoneme-level Adaptor (Sec. 3.2)
        model = Style_dubber_model_15_SPost_Duration ===========> Paper Figure 2, b) Phoneme-guided Lip Aligner (PLA) (Sec. 3.3), and c) Utterance-level Style Learning (USL) (Sec. 3.4)
    """
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_ID22_E1_Duration2(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_ID22_E1_Duration_AB0(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration_AB0(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_ID22_E1_Duration_AB1MSA(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same_AB1_MSA.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_ID22_E1_Duration_AB2USL(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration_Ab2_USL(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_ID22_E1_Duration_AB8_MelDecoder(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration_Ab8_MelDecoder(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_ID22_E1_Duration_AB9_Post(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration_Ab9_Post(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model




def get_model_ID22_E1_Duration_AB3PLA(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration_AB3_PLA(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model




def get_model_ID22_E1_Duration_AB5_V_to_L(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same_AB5_V_to_L.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_ID22_E1_Duration_AB7_RR(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same_AB7_RR.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model



def get_model_ID22_E1_Duration_AB4_A_to_L(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same_AB4_A_to_L.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_model_ID22_E1_Duration_6_framelevel(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = Style_dubber_model_15_SPost_Duration(preprocess_config, model_config).to(device)
    fusion_model = models_16_5_Face2_same_6_Frame.MULTModel(model_config["Multimodal-Transformer"], model_config, preprocess_config).to(device)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        
        fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, fusion_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        return model, fusion_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


############################################||||################################################
############################################||||################################################
############################################||||################################################
############################################||||################################################


def get_model_Style_dubber_model_Encoderstyle(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Encoderstyle(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Style_dubber_model_styleEn_Down(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_styleEn_Down(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Style_dubber_model_styleEn(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_styleEn(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Style_dubber_model_SAP2(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_SAP2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_Style_dubber_model_SAP1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_SAP1(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


# 5_Monotonic1
def get_model_Monotonic(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Monotonic(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


# 5_Monotonic1
def get_model_Dia(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Dia_softplus(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_softplus(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Dia_softplus_P(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_softplus_P(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Dia_softplus_MoreAttention(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_softplus_MoreAttention(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_Dia_softplus_MoreAttention2(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_softplus_MoreAttention2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_12_Style(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Model12_Style_Dubber(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_12_Style_CTC(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Model12_Style_Dubber_CTC(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_12_Style_CTC_E1(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Model12_Style_Dubber_CTC_1(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_Dia_LipCTC(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_LipCTC(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_Dia_LipCTC_SOFTPLUS(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_LipCTC_SOFTPLUS(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_Dia_AttenCTC(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_Dia_AttenCTC(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_V2C(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model(preprocess_config, model_config).to(device)
    if args.restore_step:
        # ckpt_path = os.path.join(
        #     train_config["path"]["file"], train_config["path"]["ckpt_path"],
        #     "{}.pth.tar".format(args.restore_step),
        # )
        ckpt_path = os.path.join(
            train_config["path"]["pretrained_path"]
        )
        ckpt = torch.load(ckpt_path)
        
        # 
        model_dict = model.state_dict()
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() \
                                if k in model_dict and k != "speaker_classifier.layers.0.weight" and  k != 'speaker_classifier.layers.0.bias'}
        model.load_state_dict(ckpt["model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_CTCEncoder(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_CTCEncoder(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_model_Watch_E1_train_deep_DeNoise2_change(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_change(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_model_NoDownSample(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    # model = FastSpeech2(preprocess_config, model_config).to(device)
    model = Style_dubber_model_NoDownSample(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["file"], train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        if speaker.split('_')[1] == '22k':
            with open("hifigan/config_22k.json", "r") as f:
                config = json.load(f)
        elif speaker.split('_')[1] == '16k':
            with open("hifigan/config_16k.json", "r") as f:
                config = json.load(f)

        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)

        if speaker == "LibriTTS_22k":
            ckpt = torch.load("hifigan/pretrained/generator_universal.pth.tar")
        elif speaker == "AISHELL3_22k":
            ckpt = torch.load("hifigan/pretrained/generator_aishell3.pth.tar")
        
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    
    elif name == "realHiFi-GAN_UniverVersion":
        config_file = os.path.join("./vocoder/UNIVERSAL_V1", "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)
        vocoder = Generator(h).to(device)
        state_dict_g = load_checkpoint(os.path.join("./vocoder/UNIVERSAL_V1", "g_02500000"),
                                       device)
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()

    return vocoder

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "realHiFi-GAN_UniverVersion":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
