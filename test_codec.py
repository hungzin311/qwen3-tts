import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "./Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

enc = tokenizer.encode("data/train/wavs/utt_00000.wav")
wavs, sr = tokenizer.decode(enc)
sf.write("decode_output.wav", wavs[0], sr)
