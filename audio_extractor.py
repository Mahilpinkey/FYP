# audio_extractor.py
# This script extracts audio from a YouTube video (URL) 
# Requirements: pytube, moviepy

import os
import yt_dlp
from moviepy import VideoFileClip

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_folder")
VIDEO_FOLDER = os.path.join(BASE_DIR, "downloaded_videos")

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def download_youtube_video(url, video_name):
    ensure_folder(VIDEO_FOLDER)
    output_path = os.path.join(VIDEO_FOLDER, video_name)
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def extract_audio_from_file(video_path, output_audio_name):
    ensure_folder(OUTPUT_FOLDER)
    output_audio_path = os.path.join(OUTPUT_FOLDER, output_audio_name)
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path)

# Run
video_path = download_youtube_video(
    'https://www.youtube.com/watch?v=R2sbzQ9T_eA', 'video.mp4'
)
extract_audio_from_file(video_path, 'extracted_audio.mp3')
