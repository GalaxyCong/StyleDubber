import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2mel = torch.jit.load('ckpts/wav2mel.pt')
dvector = torch.jit.load('ckpts/dvector.pt').eval().to(device)


def find_file_in_directory(directory, filename):
    for root, _, files in os.walk(directory):
        for file in files:
            if file == filename:
                return os.path.join(root, file)
    print('{} not found'.format(filename))
    return None  


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = norm(vector1)
    norm_vector2 = norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def eval(output_folder, gt_folder):

    similarity_list = []
    
    # Change you path: the .txt file of GRID setting2
    with open(
            "/data/conggaoxiang/GRID/GRID_dataset/Grid_Wav_22050_Abs_Feature/Refrence_list_self.txt", 
            "r",
             encoding="utf-8"
        ) as f:

            for line in tqdm(f.readlines()):
                name, text, speaker, ref_name, ref_text = line.strip("\n").split("|")

                output_file_path = os.path.join(output_folder, "wav_pred_{}.wav".format(name))
                

                gt_file_path = os.path.join(gt_folder, speaker, "{}.wav".format(ref_name))

                wav_tensor_output, sample_rate = torchaudio.load(output_file_path)
                if wav_tensor_output.shape[1] < 2205:
                    continue
                with torch.no_grad():
                    try:
                        mel_tensor_output = wav2mel(wav_tensor_output, sample_rate)
                    except RuntimeError:
                        continue
                wav_tensor_gt, sample_rate = torchaudio.load(gt_file_path)
                with torch.no_grad():
                    mel_tensor_gt = wav2mel(wav_tensor_gt, sample_rate)

                with torch.no_grad():
                    emb_output = dvector.embed_utterance(mel_tensor_output.to(device))
                    emb_output = emb_output.detach().cpu().numpy()
                    emb_gt = dvector.embed_utterance(mel_tensor_gt.to(device))
                    emb_gt = emb_gt.detach().cpu().numpy()
                    similarity_list.append(cosine_similarity(emb_output, emb_gt))
    
    similarity_list = np.array(similarity_list)
    print('the mean cosine similarity between {} and {} is {}'.format(output_folder, gt_folder, similarity_list.mean()))
    return similarity_list.mean()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pred_path",
        type=str,
        required=True,
        help="path to pred wav files",
    )
    parser.add_argument(
        "-t", 
        "--target_path", 
        type=str,
        required=True, 
        help="path to the target wav files",
    )
    args = parser.parse_args()

    eval(output_folder=args.pred_path, gt_folder=args.target_path)



    

    