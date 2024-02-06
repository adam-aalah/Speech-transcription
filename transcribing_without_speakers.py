"""
@author: AALAH Adam
"""

from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
from speechbrain.pretrained import WhisperASR
import torchaudio
import torch
from os import listdir
from os.path import isfile, join
from speechbrain.pretrained import EncoderDecoderASR
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gc
import logging
logging.basicConfig(level=logging.ERROR)  

# # Check if dispatcher is enabled
# if torchaudio.get_audio_backend() == 'soundfile':
#     torchaudio.set_audio_backend("soundfile")
#     print("The torchaudio backend is switched to 'soundfile'. Note that 'sox_io' is not supported on Windows.")
# else:
#     print("Dispatcher is not enabled. The torchaudio backend is not switched.")

# path to folder containing audio files
audio_folder_path = "C:/Users/laaalah/Downloads/Audio_C102/"

# list of all audio files in the folder
audio_files = [f for f in listdir(audio_folder_path) if isfile(join(audio_folder_path, f)) and f.endswith('.wav')]

# At the start of the script or before entering the loop:
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

chunk_duration_ms = 30000  # Process audio in 30 seconds chunks

def read(k):
    y = np.array(k.get_array_of_samples())
    return np.float32(y) / 32768

def transcribe_with_openai_whisper_large_v2(audio_np_array, processor, model, sr=16000):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

    # Process audio input
    input_features = processor(audio_np_array, sampling_rate=sr, return_tensors="pt").input_features

    # Generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

output_folder = "C:/Users/laaalah/Downloads/Audio_C102_Transcripts_without_speakers_test2/"

for audio_file in audio_files:
    # Construct the full path to the audio file
    audio_file_path = join(audio_folder_path, audio_file)

    # Load the audio file
    audio = AudioSegment.from_wav(audio_file_path)
    audio = audio.set_frame_rate(16000)

    # Get the total duration of the audio in milliseconds
    total_duration_ms = len(audio)

    # Transcribe the audio without diarization (Output  = Transcription without speakers)
    with open(join(output_folder, f"{audio_file}_transcription.txt"), "a", encoding="utf-8") as f:
        for start_ms in range(0, total_duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)

            # Extract the chunk
            chunk = audio[start_ms:end_ms]

            # Transcribe the chunk
            tr_array = read(chunk)
            transcription = transcribe_with_openai_whisper_large_v2(tr_array, processor, model)

            if not transcription:
                print(f"Unexpected result format for file {audio_file}, chunk {start_ms}-{end_ms}")
                continue

            f.write(f"{transcription} ")
            f.flush()
            print(f"Written transcription for {audio_file}, chunk {start_ms}-{end_ms}")

            del tr_array
            gc.collect()
