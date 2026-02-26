import torch
import soundfile as sf
from transformers.generation.streamers import BaseStreamer
from qwen_tts import Qwen3TTSModel


class CodecTokenStreamer(BaseStreamer):
    def __init__(self, eos_token_id=None):
        self.eos_token_id = eos_token_id
        self.step = 0

    def put(self, value):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            token_ids = value.detach().view(-1).tolist()
        elif isinstance(value, (list, tuple)):
            token_ids = [int(v) for v in value]
        else:
            token_ids = [int(value)]

        for token_id in token_ids:
            self.step += 1
            print(f"[codec step {self.step}] token={token_id}")
            if self.eos_token_id is not None and token_id == self.eos_token_id:
                print("[codec] EOS token detected")

    def end(self):
        print("[codec] stream finished")


device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "finetuning/model_weight/checkpoint-epoch-9",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

eos_token_id = tts.model.config.talker_config.codec_eos_token_id
streamer = CodecTokenStreamer(eos_token_id=eos_token_id)

wavs, sr = tts.generate_custom_voice(
    text="hello , how are you?",
    speaker="speaker_test",
    language="English",
    non_streaming_mode=False,
    max_new_tokens=512,
    eos_token_id=eos_token_id,
    streamer=streamer,
)
sf.write("output.wav", wavs[0], sr)