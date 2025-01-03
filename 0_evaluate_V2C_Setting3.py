import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import json
import ast
import argparse
import glob
# from pymcd.mcd import Calculate_MCD  # no GT, ====>, no MCD-DTW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.model import get_model_ID22_E1_Duration, get_vocoder
from resemblyzer import VoiceEncoder
from dataset import Dataset_denoise2_Setting3
from scipy.io.wavfile import write
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Test_more_MCD_with_GT(i, audio_path, mcd_toolbox, mcd_toolbox_dtw, mcd_toolbox_dtw_sl):
    name_i = i.split("/")[-1].split("_pred_")[-1].split(".wav")[0]
    base_name = name_i.split("_00")[0]
    target_wav = os.path.join(audio_path, base_name, "{}.wav".format(name_i))
    Predict_wav = i
    plain_value = mcd_toolbox.calculate_mcd(Predict_wav, target_wav)
    dtw_value = mcd_toolbox_dtw.calculate_mcd(Predict_wav, target_wav)
    dtw_value_sl = mcd_toolbox_dtw_sl.calculate_mcd(Predict_wav, target_wav)
    return plain_value, dtw_value, dtw_value_sl
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
def save_wav(sampling_rate, samples_path,
                wav_predictions_batch, tags_batch, zeroref_batch):
    for i in range(len(wav_predictions_batch)):
        generated_path = os.path.join(samples_path)
        os.makedirs(generated_path, exist_ok=True)
        pred_fpath = os.path.join(generated_path, "wav_pred_{}_{}.wav".format(tags_batch[i], zeroref_batch[i]))
        write(pred_fpath, sampling_rate, wav_predictions_batch[i])
def evaluate_test_allinference(model, fusion_model, step, configs, vocoder, val_samples_path):
    preprocess_config, model_config, train_config = configs
    val_set = "V2C_Setting3.txt"
    dataset = Dataset_denoise2_Setting3(
        val_set, preprocess_config, train_config, sort=False, drop_last=False, inference_mode=True
    )
    print("Setting3:", len(dataset))
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    wav_predictions_batch = []
    tags_batch =[]
    zeroref_batch = []
    for batchs in loader:
        for batch in batchs:
            with torch.no_grad():
                id_basename, zeroref, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_id = model.parse_batch_Setting3(batch)
                feature, src_masks, speaker_predicts, emotion_id_embedding, text_encoder = fusion_model(text, face_embedding, src_len, max_src_len, ref_mels, ref_mel_lens, face_lens, MaxfaceL)
                Ture_postnet_mel_predictions, mel_lens_pred = model(feature, text_encoder, src_masks, ref_mels, ref_mel_lens, face_lens, MaxfaceL, lip_embedding, spk_embedding, mel_lens=mel_lens, max_mel_len=max_mel_len)
                wav_predictions, tags= synth_multi_samples_predonly(
                    id_basename, Ture_postnet_mel_predictions, mel_lens_pred, 
                    vocoder,
                    model_config,
                    preprocess_config,
                )
                wav_predictions_batch.extend(wav_predictions)
                tags_batch.extend(tags)
                zeroref_batch.extend(zeroref)
    print("len(dataset):", len(dataset))
    print("len(wav_predictions_batch):", len(wav_predictions_batch))
    print("len(tags_batch):", len(tags_batch))
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    save_wav(sampling_rate, val_samples_path, wav_predictions_batch, tags_batch, zeroref_batch)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    aim = "./outout"
    parser.add_argument("--restore_step", type=int, default=47000)

    parser.add_argument('--save_path', default='0_Cemara_Setting3_ZeroGrid/{}'.format(aim.split('/')[-1]))

    
    args = parser.parse_args()
    val_samples_path = os.path.join(args.save_path, 'Who_{}_Wav'.format(args.restore_step))
    # Read Config
    with open(aim+'/model_config/MovieAnimation/config_all.txt', 'r') as file:
        preprocess_config, model_config, train_config = file.readlines()
    preprocess_config = ast.literal_eval(preprocess_config)
    model_config = ast.literal_eval(model_config)
    train_config = ast.literal_eval(train_config)
    configs = (preprocess_config, model_config, train_config)
    vocoder = get_vocoder(model_config, device)
    encoder_emo = VoiceEncoder().to(device)
    # Get model
    model, fusion_model = get_model_ID22_E1_Duration(args, configs, device, train=False)
    evaluate_test_allinference(model, fusion_model, args.restore_step, configs, vocoder, val_samples_path)
    print("Generated all done (wav)..., save in {}".format(val_samples_path))
    
    
    
    