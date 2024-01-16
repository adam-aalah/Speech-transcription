from pydub import AudioSegment
from os import listdir
from os.path import isfile, join
from speechbrain.pretrained import WhisperASR
import gc
import logging

logging.basicConfig(level=logging.ERROR)

# Path to folder containing audio files
audio_folder_path = "C:/Users/laaalah/Downloads/Audio_D002/"

# List of all audio files in the folder
audio_files = [f for f in listdir(audio_folder_path) if isfile(join(audio_folder_path, f)) and f.endswith('.wav')]

# Initialize the SpeechBrain WhisperASR model
asr_model = WhisperASR.from_hparams(source="speechbrain/asr-whisper-medium-commonvoice-fr", savedir="pretrained_models/asr-whisper-medium-commonvoice-fr")

output_folder = "C:/Users/laaalah/Downloads/Audio_D002_Speechbrain/"
chunk_duration_ms = 10000  # Process audio in 10 seconds chunks (change to suit the desired chunk size)

for audio_file in audio_files:
    # full path to the audio file
    audio_file_path = join(audio_folder_path, audio_file)

    # Load the audio file
    audio = AudioSegment.from_wav(audio_file_path)
    audio = audio.set_frame_rate(16000)  # sample rate

    total_duration_ms = len(audio)
    with open(join(output_folder, f"{audio_file}_transcription.txt"), "a", encoding="utf-8") as f:
        for start_ms in range(0, total_duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
            chunk = audio[start_ms:end_ms]

            # Save the chunk as a temporary WAV file
            temp_filename = "temp_chunk.wav"
            chunk.export(temp_filename, format="wav")

            # Transcribe the chunk
            transcription_chunks = asr_model.transcribe_file(temp_filename)

            # Flatten the list of lists and join into a string
            transcription = ' '.join(word for sublist in transcription_chunks for word in sublist)

            if not transcription.strip():
                print(f"Unexpected result format for file {audio_file}, chunk {start_ms}-{end_ms}")
                continue

            f.write(f"{transcription} ")
            f.flush()
            print(f"Written transcription for {audio_file}, chunk {start_ms}-{end_ms}")

            # Memory management
            del chunk
            gc.collect()
