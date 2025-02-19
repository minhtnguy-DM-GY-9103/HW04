from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Text prompt
text_prompt = "A happy birthday song"

# Process the text input
inputs = processor(
    text=text_prompt,
    padding=True,
    return_tensors="pt"
)

# Generate audio
audio_values = model.generate(**inputs, max_new_tokens=1024)

# Convert to numpy array and save as WAV file
audio_array = audio_values[0, 0].numpy()
scipy.io.wavfile.write("generated_music4.wav", rate=32000, data=audio_array)