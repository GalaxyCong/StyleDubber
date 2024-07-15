numbers=(1 2 4 8)

for num in "${numbers[@]}"; do
    model_path="/data/conggaoxiang/6_PAMI/1_Stage_E1/1_PAMI_Li_Denoiser/Ab/new_Denoiser_1_nocut_transformerP_H${num}/Exp0/test_wav_path/V2C"

    python WER_whisper_large_v3_bashversion.py -p "$model_path"  -t "/data/conggaoxiang/dataset/wav_22050_chenqi_clean_Denoise_version2_all"
done



