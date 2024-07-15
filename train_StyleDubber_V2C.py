
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'

import argparse
import matplotlib.pyplot as plt
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ast
from utils.model import get_model_ID22_E1_Duration, get_vocoder, get_param_num
from utils.tools import log
from dataset import Dataset_denoise2_V2Cdataset
from evaluate import evaluate_Denoise2_ID20Emo_Setting1 
from style_dubber import Style_dubber_model_loss_15_Emo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import shutil

def move_all_files(source_folder, destination_folder):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
        os.makedirs(destination_folder)
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_file)
            print(f"Moved: {source_file} -> {destination_file}")
            
def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    train_set = "train.txt"
    dataset = Dataset_denoise2_V2Cdataset(
        train_set, preprocess_config, train_config, sort=True, drop_last=True)
    
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4 
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,)

    # Prepare model
    model, fusion_model, optimizer = get_model_ID22_E1_Duration(args, configs, device, train=True)
    
    num_param = get_param_num(model) + get_param_num(fusion_model)
    print("Number of the StyleDubber Parameters:", num_param)
    
    Loss = Style_dubber_model_loss_15_Emo(preprocess_config, model_config).to(device)

    os.makedirs(os.path.join(train_config["path"]["file"], train_config["path"]["model_config"]), exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["file"], train_config["path"]["ckpt_path"]), exist_ok=True)
    with open(os.path.join(train_config["path"]["file"], train_config["path"]["model_config"], "model.txt"), "w") as f_log:
        f_log.write(str(model))
    
    with open(os.path.join(train_config["path"]["file"], train_config["path"]["model_config"], "config_all.txt"), "w") as f_log:
        for config in configs:
            f_log.write(str(config) + "\n")
            
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    train_log_path = os.path.join(train_config["path"]["file"] ,train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["file"], train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                id_basename, text, src_len, max_src_len, speakers, ref_mels, ref_mel_lens, mel_target, mel_lens, max_mel_len, pitches, energies, durations, ref_linguistics, face_lens, MaxfaceL, lip_embedding, spk_embedding, face_embedding, emos_embedding, emotion_id = model.parse_batch(batch)
                feature, src_masks, speaker_predicts, emotion_id_embedding, text_encoder = fusion_model(text, face_embedding, src_len, max_src_len, ref_mels, ref_mel_lens, face_lens, MaxfaceL)
                mel_predictions, postnet_mel_predictions, log_duration_predictions, _, src_masks, mel_masks, _, ref_mel_masks, AV_attn, lip_masks = model(feature, text_encoder, src_masks, ref_mels, ref_mel_lens, face_lens, MaxfaceL, lip_embedding, spk_embedding, mel_lens=mel_lens, max_mel_len=max_mel_len, d_targets=durations, p_targets=pitches, e_targets=energies)

                total_loss, item_losses = Loss(text, speakers, mel_target, pitches, energies, mel_lens, durations, ref_linguistics, mel_predictions, postnet_mel_predictions, log_duration_predictions, src_masks, mel_masks, ref_mel_masks, speaker_predicts, spk_embedding, AV_attn, src_len, face_lens, lip_masks, emotion_id_embedding, emotion_id)

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    str1 = "Step [{}/{}]:".format(step, total_step)
                    str2 = "Total_Loss: {:.4f}\n Mel_Loss1: {:.4f}, Mel_Loss_Post: {:.4f}," \
                            "speaker_loss: {:.4f}, Lip_Duration_loss: {:.4f}, Diagonal_loss: {:.4f}, Emos_class_loss: {:.4f};" \
                            .format(item_losses[0], item_losses[1], item_losses[2], item_losses[3], item_losses[4], item_losses[5], item_losses[6])
                    outer_bar.write(str1 + str2)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(str1 + str2 + "\n")
                    log(train_logger, step, losses=item_losses)
                
                # Saveimage
                if step % val_step == 0:
                    AV_attn_path = os.path.join(train_log_path, "AV_attn_image_Step{}".format(step))
                    os.makedirs(AV_attn_path, exist_ok=True)
                    for idx, attn in enumerate(AV_attn):
                        save_plot(attn.cpu().detach().numpy(), os.path.join(AV_attn_path, f"{id_basename[idx]}.png")) 
 
                # if step % val_step == 0 and step != 0 and step >= 16000:
                if step % val_step == 0 and step != 0:
                    print("START_Ablation EVAL...")
                    model.eval()
                    fusion_model.eval()
                    mel_loss_all_val = evaluate_Denoise2_ID20Emo_Setting1(model, fusion_model, step, configs, val_log_path, val_logger, vocoder)
                    str1 = "Test step [{}/{}]:".format(step, total_step)
                    ## val loss
                    # str2 = "Model1 (have GT duration): Total_Loss: {:.4f} Mel_Loss1: {:.4f}, Mel_Loss_Post: {:.4f}," \
                    #         "speaker_loss_1: {:.4f}, speaker_loss_2: {:.4f}, Lip_Duration_loss: {:.4f}, PhnCls_Loss: {:.4f}, PhnCls_ACC: {:.4f}, CTC_loss_MEL: {:.4f}, Dia_loss: {:.4f}, Emos_class_loss : {:.4f};" \
                    #         .format(loss_all_val[0], loss_all_val[1], loss_all_val[2], loss_all_val[3], loss_all_val[4], loss_all_val[5], loss_all_val[6], loss_all_val[7], loss_all_val[8], loss_all_val[9], loss_all_val[10])
                    ## val score
                    str3 = "Model2 (None GT duration) Metrics MCD: {:.4f} MCD-DTW: {:.4f}, MCD-DTW-SL: {:.4f} ;" \
                            .format(mel_loss_all_val[0], mel_loss_all_val[1], mel_loss_all_val[2])

                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        # f.write(str1 + str2 + str3 +"\n")
                        f.write(str1 + str3 +"\n")
                    outer_bar.write(str1 + str3)
                    
                    
                    torch.save(
                    {
                        "model": model.state_dict(),
                        "fusion_model": fusion_model.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                    },
                    os.path.join(
                        train_config["path"]["file"], train_config["path"]["ckpt_path"],
                        "{}.pth.tar".format(step),
                    ),)

                    #
                    model.train()
                    fusion_model.train()


                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


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



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    aim = "./ModelConfig_V2C"
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-t", "--train_config", type=str,
        default="config/MovieAnimation/train.yaml"
    )
    args = parser.parse_args()
    
    with open(aim+'/model_config/MovieAnimation/config_all.txt', 'r') as file:
        preprocess_config, model_config = file.readlines()

    # load styledubber model_config and preprocess_config. (see "./ModelConfig_V2C/model_config/MovieAnimation/config_all.txt")
    preprocess_config = ast.literal_eval(preprocess_config)
    model_config = ast.literal_eval(model_config)

    # load styledubber train_config.  (see "./config/MovieAnimation/train.yaml")
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    
    configs = (preprocess_config, model_config, train_config)  
    main(args, configs)
    




