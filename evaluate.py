import argparse
import os
import glob
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.tools import log, synth_one_sample
# from style_dubber import Style_dubber_model_loss, Style_dubber_model_loss_change, Style_dubber_model_loss_noACL, Style_dubber_model_loss_Dia, Style_dubber_model_loss_Dia_1, Style_dubber_model_loss_Dia_LipACL, Style_dubber_model_loss_Dia_P1, Style_dubber_model_loss_Dia_P2, Style_dubber_model_loss_Dia_P1_ACL, Style_dubber_model_loss_Dia_P1_ACL_E1, Style_dubber_model_loss_13, Style_dubber_model_loss_15, Style_dubber_model_loss_15_Emo, Style_dubber_model_loss_15_Emo_AB0
from style_dubber import Style_dubber_model_loss_15_Emo
from dataset import Dataset_denoise2_Setting1_Run, Dataset_GRIDdataset
from joblib import Parallel, delayed
from scipy.io.wavfile import write
from tqdm import tqdm
from pymcd.mcd import Calculate_MCD
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

def Test_more_MCD_with_GT(i, audio_path, mcd_toolbox, mcd_toolbox_dtw, mcd_toolbox_dtw_sl):
    #
    name_i = i.split("/")[-1].split("_pred_")[-1].split(".wav")[0]
    base_name = name_i.split("-")[0]
    # Target
    target_wav = os.path.join(audio_path, base_name, "{}.wav".format(name_i))
    # # Predict_wav
    Predict_wav = i
    plain_value = mcd_toolbox.calculate_mcd(Predict_wav, target_wav)
    dtw_value = mcd_toolbox_dtw.calculate_mcd(Predict_wav, target_wav)
    dtw_value_sl = mcd_toolbox_dtw_sl.calculate_mcd(Predict_wav, target_wav)
    
    return plain_value, dtw_value, dtw_value_sl
    
 

def Test_more_MCD_with_GT_V2C(i, audio_path, mcd_toolbox, mcd_toolbox_dtw, mcd_toolbox_dtw_sl):
    #
    name_i = i.split("/")[-1].split("_pred_")[-1].split(".wav")[0]
    base_name = name_i.split("_00")[0]
    
    # Target
    target_wav = os.path.join(audio_path, base_name, "{}.wav".format(name_i))
    # # Predict_wav
    Predict_wav = i
    plain_value = mcd_toolbox.calculate_mcd(Predict_wav, target_wav)
    dtw_value = mcd_toolbox_dtw.calculate_mcd(Predict_wav, target_wav)
    dtw_value_sl = mcd_toolbox_dtw_sl.calculate_mcd(Predict_wav, target_wav)
    
    return plain_value, dtw_value, dtw_value_sl
    

def save_wav(sampling_rate, samples_path,
                wav_predictions_batch, tags_batch):
    for i in range(len(wav_predictions_batch)):
        generated_path = os.path.join(samples_path)
        os.makedirs(generated_path, exist_ok=True)
        pred_fpath = os.path.join(generated_path, "wav_pred_{}.wav".format(tags_batch[i]))
        write(pred_fpath, sampling_rate, wav_predictions_batch[i])

def synth_multi_samples_predonly(ids, Post_Mel, mel_len_preout, vocoder, model_config, preprocess_config):
    wav_reconstructions = []
    wav_predictions = []
    for i in range(len(Post_Mel)):
        if vocoder is not None:
            from utils.model import vocoder_infer
            wav_prediction = vocoder_infer(
                Post_Mel[i, :(mel_len_preout[i].item())].detach().transpose(0, 1).unsqueeze(0),
                vocoder,
                model_config,
                preprocess_config,
            )[0]
        else:
            wav_reconstruction = wav_prediction = None
        wav_predictions.append(wav_prediction)
    return wav_predictions, ids


# need V2C when train
def evaluate_Denoise2_ID20Emo_Setting1(model, fusion_model, step, configs, val_log_path, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs
    val_set = "val.txt"
    dataset = Dataset_denoise2_Setting1_Run(
        val_set, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False)
    print("Watch which Dataset:", preprocess_config["dataset"], "Dataset_denoise2 len(test): ", len(dataset))
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    Total_Loss_Val = []
    Mel_Loss1_Val = []
    Mel_Loss_Post_Val = []
    duration_loss_Val = []
    PhnCls_Loss_Val = []
    SpkCls_Loss_Val = []
    Phone_acc_Val = []
    
    Dialoss_Val = []
    EMOClass_Val = []
    wav_reconstructions_batch = []
    wav_predictions_batch = []
    tags_batch =[]
    speakers_batch = []
    emotions_batch = []
    cofs_batch = []
    
    # Get loss function
    # Loss = Style_dubber_model_loss_15_Emo(preprocess_config, model_config).to(device)
    # Evaluation
    # loss_sums = [0 for _ in range(9)]
    for batchs in loader:
        for batch in batchs:
            with torch.no_grad():
                id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_id = model.parse_batch(batch)
                feature, src_masks, speaker_predicts, emotion_id_embedding, text_encoder = fusion_model(text, face_embedding, src_len, max_src_len, ref_mels, ref_mel_lens, face_lens, MaxfaceL)
                Ture_postnet_mel_predictions, mel_lens_pred = model(feature, text_encoder, src_masks, ref_mels, ref_mel_lens, face_lens, MaxfaceL, lip_embedding, spk_embedding, mel_lens=mel_lens, max_mel_len=max_mel_len)
                # Total_Loss_Val.append(item_losses[0])
                # Mel_Loss1_Val.append(item_losses[1])
                # Mel_Loss_Post_Val.append(item_losses[2])
                # duration_loss_Val.append(item_losses[5])
                # SpkCls_Loss_Val.append(item_losses[8])
                # Dialoss_Val.append(item_losses[9])
                # EMOClass_Val.append(item_losses[10])
                
                wav_predictions, tags= synth_multi_samples_predonly(
                    id_basename, Ture_postnet_mel_predictions, mel_lens_pred, 
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                wav_predictions_batch.extend(wav_predictions)
                tags_batch.extend(tags)
                

    AV_attn_path = os.path.join(val_log_path, "AV_attn_image_Step{}".format(step))
    os.makedirs(AV_attn_path, exist_ok=True)
    print("len(dataset):", len(dataset))
    val_samples_path = os.path.join(train_config["path"]["file"], train_config["path"]["result_path"])
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    save_wav(sampling_rate, val_samples_path, wav_predictions_batch, tags_batch)
    print("============MCD=================")
    audio_path = "/data/conggaoxiang/dataset/wav_22050_chenqi_clean_Denoise_version2_all"  # change you path; We use the processed GT audio (e.g., Abs) as the target, which is used to convert mel-spectrograms.
    generated_path = os.path.join(val_samples_path, "*")
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd_toolbox_dtw = Calculate_MCD(MCD_mode="dtw")
    mcd_toolbox_dtw_sl = Calculate_MCD(MCD_mode="dtw_sl")
    all = glob.glob(generated_path) 
    print("test all:", len(all))
    results = Parallel(n_jobs=40, verbose=1)(
        delayed(Test_more_MCD_with_GT_V2C)(i, audio_path, mcd_toolbox, mcd_toolbox_dtw, mcd_toolbox_dtw_sl) for i in all
    )
    avg_mcd_plain = sum(result[0] for result in results)/len(all)
    avg_mcd_dtw = sum(result[1] for result in results)/len(all)
    dtw_value_sl = sum(result[2] for result in results)/len(all)
    log(logger, step, MCD_Value=[avg_mcd_plain, avg_mcd_dtw, dtw_value_sl])
    return [avg_mcd_plain, avg_mcd_dtw, dtw_value_sl]


# need GRID when train
def evaluate_GRID(model, fusion_model, step, configs, val_log_path, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs
    # Get dataset
    val_set = "val.txt"
    dataset = Dataset_GRIDdataset(
        val_set, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=False
    )
    print("Watch which Dataset:", preprocess_config["dataset"], "Dataset_denoise2 len(test): ", len(dataset))
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    Total_Loss_Val = []
    Mel_Loss1_Val = []
    Mel_Loss_Post_Val = []
    duration_loss_Val = []
    PhnCls_Loss_Val = []
    SpkCls_Loss_Val = []
    Phone_acc_Val = []
    
    Dialoss_Val = []
    EMOClass_Val = []
    #
    wav_reconstructions_batch = []
    wav_predictions_batch = []
    tags_batch =[]
    speakers_batch = []
    emotions_batch = []
    cofs_batch = []
    # Get loss function
    # Loss = Style_dubber_model_loss_15_Emo(preprocess_config, model_config).to(device)
    # Evaluation
    # loss_sums = [0 for _ in range(9)]
    for batchs in loader:
        for batch in batchs:
            # batch = to_device(batch, device)
            with torch.no_grad():
                id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding = model.parse_batch_GRID(batch)
                feature, src_masks, speaker_predicts, emotion_id_embedding, text_encoder = fusion_model(text, face_embedding, src_len, max_src_len, ref_mels, ref_mel_lens, face_lens, MaxfaceL)
                Ture_postnet_mel_predictions, mel_lens_pred = model(feature, text_encoder, src_masks, ref_mels, ref_mel_lens, face_lens, MaxfaceL, lip_embedding, spk_embedding, mel_lens=mel_lens, max_mel_len=max_mel_len)
                # Total_Loss_Val.append(item_losses[0])
                # Mel_Loss1_Val.append(item_losses[1])
                # Mel_Loss_Post_Val.append(item_losses[2])
                # duration_loss_Val.append(item_losses[5])
                # SpkCls_Loss_Val.append(item_losses[8])
                # Dialoss_Val.append(item_losses[9])
                # EMOClass_Val.append(item_losses[10])
                wav_predictions, tags= synth_multi_samples_predonly(
                    id_basename, Ture_postnet_mel_predictions, mel_lens_pred, 
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                wav_predictions_batch.extend(wav_predictions)
                tags_batch.extend(tags)
                

    AV_attn_path = os.path.join(val_log_path, "AV_attn_image_Step{}".format(step))
    os.makedirs(AV_attn_path, exist_ok=True)
    print("len(dataset):", len(dataset))
    
    val_samples_path = os.path.join(train_config["path"]["file"], train_config["path"]["result_path"], 'Test_when_train')
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    save_wav(sampling_rate, val_samples_path, wav_predictions_batch, tags_batch)
    print("============MCD=================")
    audio_path = "/data/conggaoxiang/GRID/GRID_dataset/Grid_Wav_22050_Abs"  # change you path; We use the processed GT audio (e.g., Abs) as the target, which is used to convert mel-spectrograms.
    
    print("GT_wav:", audio_path)
    generated_path = os.path.join(val_samples_path, "*")
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd_toolbox_dtw = Calculate_MCD(MCD_mode="dtw")
    mcd_toolbox_dtw_sl = Calculate_MCD(MCD_mode="dtw_sl")
    
    all = glob.glob(generated_path) 
    print("test all:", len(all))
    
    results = Parallel(n_jobs=40, verbose=1)(
        delayed(Test_more_MCD_with_GT)(i, audio_path, mcd_toolbox, mcd_toolbox_dtw, mcd_toolbox_dtw_sl) for i in all
    )
    avg_mcd_plain = sum(result[0] for result in results)/len(all)
    avg_mcd_dtw = sum(result[1] for result in results)/len(all)
    dtw_value_sl = sum(result[2] for result in results)/len(all)
    log(logger, step, MCD_Value=[avg_mcd_plain, avg_mcd_dtw, dtw_value_sl])
    return [avg_mcd_plain, avg_mcd_dtw, dtw_value_sl]



