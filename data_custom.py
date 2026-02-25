import json
import os 
import soundfile as sf 
from datasets import load_dataset, Audio 
import torchaudio.functional as F
import torch
from tqdm import tqdm

DATASET_NAME = "strongpear/viet_muong_merged_0_200_denoise_silence_speaker101"
OUTPUT_ROOT = "data"
TARGET_SAMPLE_RATE = 24000 

def proces_split(dataset, split_name):
    output_dir = os.path.join(OUTPUT_ROOT, split_name)
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    f_jsonl = open(os.path.join(output_dir, f"{split_name}_raw.jsonl"), "w", encoding="utf-8")

    for idx, item in tqdm(enumerate(dataset)):
        utt_id = f"utt_{idx:05d}"
        text_content = item['text'].strip()
        audio_array = item['audio']['array']
        original_sample_rate = item['audio']['sampling_rate']

        if original_sample_rate != TARGET_SAMPLE_RATE:
            audio_array = F.resample(torch.from_numpy(audio_array), original_sample_rate, TARGET_SAMPLE_RATE).numpy()

        wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
        ref_path = os.path.join(wav_dir, "utt_00000.wav")
        sf.write(wav_path, audio_array, TARGET_SAMPLE_RATE)

        temp_dict = {'audio': f"../{wav_path}", 'text': text_content, 'ref_audio': f"../{ref_path}"}

        json.dump(temp_dict, f_jsonl, ensure_ascii=False)
        f_jsonl.write("\n")

    f_jsonl.close()

    return

if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME, split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)
    proces_split(split["train"], "train")
    proces_split(split["test"], "dev")