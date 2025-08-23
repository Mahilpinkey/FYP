

import os

import yt_dlp

from pydub import AudioSegment

import speech_recognition as sr

from datetime import datetime



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_folder")

def ensure_folder(folder):

    if not os.path.exists(folder):

        os.makedirs(folder)

def download_youtube_audio(url, output_audio_name="extracted_audio.mp3"):

    ensure_folder(OUTPUT_FOLDER)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



    # Remove extension here (yt-dlp adds it back)

    base_name = os.path.splitext(output_audio_name)[0]

    audio_path = os.path.join(

        OUTPUT_FOLDER,

        f"{base_name}_{timestamp}.mp3"

    )



    print("Downloading audio only...")

    ydl_opts = {

        'outtmpl': os.path.join(OUTPUT_FOLDER, f"{base_name}_{timestamp}"),  # no extension!
        'verbose': True,
        'format': 'bestaudio/best',
        "cookies": os.path.join(BASE_DIR, "cookies.txt"),
        'quiet': False,

        'postprocessors': [{

            'key': 'FFmpegExtractAudio',

            'preferredcodec': 'mp3',

            'preferredquality': '192',

        }]

    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        ydl.download([url])



    print(f"✅ Audio saved to {audio_path}")

    return audio_path





def audio_to_text(audio_file, output_text_file="transcription.txt"):

    ensure_folder(OUTPUT_FOLDER)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



    # Convert MP3 → WAV (mono, 16kHz)

    wav_path = os.path.join(OUTPUT_FOLDER, f"converted_audio_{timestamp}.wav")

    sound = AudioSegment.from_file(audio_file, format="mp3")

    sound = sound.set_channels(1)

    sound = sound.set_frame_rate(16000)

    sound.export(wav_path, format="wav")



    recognizer = sr.Recognizer()

    text_chunks = []



    print("Converting audio to text...")

    with sr.AudioFile(wav_path) as source:

        while True:

            audio_data = recognizer.record(source, duration=30)  # 30 sec chunks

            if len(audio_data.frame_data) == 0:

                break

            try:

                chunk_text = recognizer.recognize_google(audio_data)

                text_chunks.append(chunk_text)

                

            except sr.UnknownValueError:

                text_chunks.append("[Unintelligible]")

            except sr.RequestError as e:

                text_chunks.append(f"[API Error: {e}]")

                break



    full_text = " ".join(text_chunks)



    # Save to file with timestamp

    output_file_path = os.path.join(

        OUTPUT_FOLDER,

        f"{os.path.splitext(output_text_file)[0]}_{timestamp}.txt"

    )

    with open(output_file_path, "w", encoding="utf-8") as f:

        f.write(full_text)



    print(f"\n✅ Transcription saved to: {output_file_path}")

    return None





if __name__ == "__main__":

    youtube_url = "https://www.youtube.com/watch?v=s3vpH3A_eTA&t=10s"



    audio_path = download_youtube_audio(youtube_url)  # ✅ fixed function call

    audio_to_text(audio_path, "video_transcription.txt")   


