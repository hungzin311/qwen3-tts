import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "finetuning/model_weight/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

wavs, sr = tts.generate_custom_voice(
    text="Lạy ông bà, con đến nhà ông bà rồi. Con đến nhà ông bà rồi.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)