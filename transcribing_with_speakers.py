"""
@author: AALAH Adam
"""

from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
from speechbrain.pretrained import WhisperASR
import torchaudio
import torch
from speechbrain.pretrained import EncoderDecoderASR
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gc
import logging
logging.basicConfig(level=logging.ERROR)  




# Initialize the diarization pipeline
# diar_pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization", use_auth_token="...") #hugginface token
diar_pipeline = Pipeline.from_pretrained(
    #"pyannote/speaker-diarization", 
    "pyannote/speaker-diarization@2.1",
    use_auth_token="...") #hugginface token

k = str(diar_pipeline("C:/Users/laaalah/Desktop/Audios/DS500484.wav", num_speakers=3)).split('\n')

# Convert pydub AudioSegment to numpy array
def read(k):
    y = np.array(k.get_array_of_samples())
    return np.float32(y) / 32768

def millisec(timeStr):
    spl = timeStr.split(":")
    if len(spl) != 3:
        return None
    try:
        return (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    except ValueError:
        return None

def extract_times(line):
    start_str = line.split('-->')[0].split('[')[1].strip()
    end_str = line.split('-->')[1].split(']')[0].strip()
    return start_str, end_str

def transcribe_with_openai_whisper_large_v2(audio_np_array, processor, model, sr=16000):
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

    # Process audio input
    input_features = processor(audio_np_array, sampling_rate=sr, return_tensors="pt").input_features

    # Generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

 

audio = AudioSegment.from_wav("C:/Users/laaalah/Desktop/Audios/DS500484.wav")
audio = audio.set_frame_rate(16000)

# At the start of the script or before entering the loop:
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")


with open("C:/Users/laaalah/Desktop/Audios/Transcripts/DS500484.txt", "a", encoding="utf-8") as f:
    for l in range(len(k)):
        j = k[l].split(" ")
        if len(j) < 4:  
            print(f"Unexpected format in line {l}: {k[l]}")
            continue
        
        start_str, end_str = extract_times(k[l])
        start = millisec(start_str)
        end = millisec(end_str)

        if start is None or end is None:
            print(f"Skipping invalid timestamp in line {l}: {start_str} to {end_str}")
            continue

        tr_segment = audio[start:end]
        tr_array = read(tr_segment)
        #transcription = transcribe_with_openai_whisper_large_v2(tr_array)
        transcription = transcribe_with_openai_whisper_large_v2(tr_array, processor, model)

        if not transcription:
            print(f"Unexpected result format for segment {start_str} to {end_str}")
            continue
        
        f.write(f'\n[ {start_str} -- {end_str} ] {j[6]} : {transcription}')
        f.flush()
        print(f'Written: [ {start_str} -- {end_str} ] {j[6]} : {transcription}')  

        del tr_array, tr_segment, j
        gc.collect()
