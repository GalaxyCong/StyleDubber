import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


import whisper
import argparse

from joblib import Parallel, delayed
# import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import json

from jiwer import wer
import re

def replace_numbers(text):
    number_mapping = {
        '0': ' zero', '1': ' one', '2': ' two', '3': ' three', '4': ' four',
        '5': ' five', '6': ' six', '7': ' seven', '8': ' eight', '9': ' nine'
    }
    for digit, word in number_mapping.items():
        text = text.replace(digit, word)

    return text


def read(pred_path, preprocess_path, filename, model):

    if filename[0:9] == 'meta_rec_':
        basename = filename[9:]
    elif filename[0:9] == 'wav_pred_':
        basename = filename[9:]
    elif filename[0:4] == 'rec_':
        basename = filename[4:]
    elif filename[0:8] == 'wav_rec_':
        basename = filename[8:]
    else:
        basename = filename

    basename = basename.split('_s')[0] # here, yes setting3, wav_pred_Bossbaby@BossBaby_00_0559_00_s3-pwax5a.wav =========> 

    if 'Denoise' in preprocess_path:
        firstname = basename.split("_00")[0]
    else:
        firstname = basename.split("-")[0]

    GT_path = os.path.join(preprocess_path, firstname, "{}.lab".format(basename))
    wav_path = os.path.join(pred_path, filename)
    
    result = model.transcribe(wav_path)
    predicted = result["text"]

    predicted_line = predicted.strip()
    
    if predicted_line:
        text_lines = []
        with open(GT_path, 'r', encoding='utf-8') as file:  
            for line in file:
                cleaned_line = line.rstrip()
                text_lines.append(cleaned_line)
        text_content = ' '.join(text_lines)

        text_content = re.sub(r'[^\w\s]', '', text_content)
        text_content = replace_numbers(text_content)
        predicted = re.sub(r'[^\w\s]', '', predicted)
        predicted = replace_numbers(predicted)

        error = wer(predicted.lower(), text_content.lower())
        return error, basename, predicted.lower(), text_content.lower()
            

if __name__ == '__main__':


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

    # path to the groundtruth text
    preprocess_path = args.target_path

    # path to the target wav
    pred_path = args.pred_path
    print("The test path: {}".format(pred_path))

    wav_list = os.listdir(pred_path)
    print("len_name:", len(wav_list))
    
    model = whisper.load_model("large-v3")
    
    

    results_P = []
    for i in tqdm(wav_list):
        try:
            results_P.append(read(pred_path, preprocess_path, i, model))
        except Exception:
            print("{} is not normal".format(i))
            continue


    results = [i[0] for i in results_P if i is not None and i[0] is not None]
    nonename = [i[1] for i in results_P if i is not None and i[0] is not None]
    predicted_text = [i[2] for i in results_P if i is not None and i[0] is not None]
    gt_text = [i[3] for i in results_P if i is not None and i[0] is not None]
    

    print("len(results)", len(results))
    print("len(nonename)", len(nonename))

    print("The test path: {}".format(pred_path))
    print("SUM:", sum(results), "Length:", len(results), "WER_Currtly:", sum(results)/len(results), "None_number: ", len(wav_list)-len(results))
    print("============================over=========================")
    print("\n")

    output_file_path = os.path.join('wer_log_large', '{}.txt'.format(args.pred_path.replace('/', '_')))

    with open(output_file_path, 'a') as file:
        for i in range(len(results)):
            file.write("{}: WER-Result:{:.4f}, gt_content:{}, predicted_content:{} \n".format(
                    nonename[i], results[i], gt_text[i], predicted_text[i]
            ))


