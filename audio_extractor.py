import os
import yt_dlp
from pydub import AudioSegment
import speech_recognition as sr
from datetime import datetime

# ✅ NLP imports
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Punctuation restoration model
from deepmultilingualpunctuation import PunctuationModel

# Download nltk resources (only first run)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_folder")

# Load punctuation model once at startup
punct_model = PunctuationModel()

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def download_youtube_audio(url, output_audio_name="extracted_audio.mp3"):
    ensure_folder(OUTPUT_FOLDER)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = os.path.splitext(output_audio_name)[0]
    audio_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_{timestamp}.mp3"
    )

    print("Downloading audio only...")
    ydl_opts = {
        'outtmpl': os.path.join(OUTPUT_FOLDER, f"{base_name}_{timestamp}"),
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

    # ✅ Restore punctuation
    print("⏳ Restoring punctuation...")
    punctuated_text = punct_model.restore_punctuation(full_text)
    print("✅ Punctuation restored")

    # Save raw punctuated transcript
    transcript_file = os.path.join(
        OUTPUT_FOLDER,
        f"{os.path.splitext(output_text_file)[0]}_{timestamp}.txt"
    )
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(punctuated_text)
    print(f"✅ Transcript saved: {transcript_file}")

    # ✅ NLP processing: Sentence segmentation + tokenization + ISL gloss
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(punctuated_text)

    all_tokens = []
    all_gloss = []

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        processed_tokens = [
            lemmatizer.lemmatize(tok) for tok in tokens if tok.isalnum() and tok not in stop_words
        ]
        isl_gloss = [tok.upper() for tok in processed_tokens]

        if processed_tokens:
            all_tokens.append(" ".join(processed_tokens))
            all_gloss.append(" ".join(isl_gloss))

    # Save all tokens (one sentence per line)
    tokens_file = os.path.join(OUTPUT_FOLDER, f"all_tokens_{timestamp}.txt")
    with open(tokens_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_tokens))

    # Save all gloss (one sentence per line)
    gloss_file = os.path.join(OUTPUT_FOLDER, f"all_gloss_{timestamp}.txt")
    with open(gloss_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_gloss))

    print(f"✅ All tokens saved: {tokens_file}")
    print(f"✅ All gloss saved: {gloss_file}")

    return punctuated_text

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=ZB4lOVRMt5I&t=1s"
    audio_path = download_youtube_audio(youtube_url)
    audio_to_text(audio_path, "video_transcription.txt")
