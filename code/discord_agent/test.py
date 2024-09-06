import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts

# Load a pre-trained TTS model
model = nemo_tts.models.Tacotron2Model.from_pretrained(model_name="Tacotron2-22050Hz")

# Convert text to speech
audio = model.convert_text_to_waveform("Hello, this is a test.")

# Save the audio to a file
with open("output.wav", "wb") as f:
    f.write(audio)