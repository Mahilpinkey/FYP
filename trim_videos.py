import cv2
import numpy as np
import subprocess
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe as mp
from shutil import copy2

def extract_pose_hand_features(video_path, sample_rate=2, max_frames=None):
    """Extract pose + hand landmarks as features."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mp_hol = mp.solutions.holistic

    features = []
    frame_idx = 0
    with mp_hol.Holistic(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        refine_face_landmarks=False) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(img_rgb)

            def lm_to_array(lms, expected_n):
                arr = np.zeros((expected_n, 3), dtype=np.float32)
                if lms is not None:
                    for i, lm in enumerate(lms.landmark):
                        if i >= expected_n:
                            break
                        arr[i, 0] = lm.x
                        arr[i, 1] = lm.y
                        arr[i, 2] = lm.z if hasattr(lm, "z") else 0.0
                return arr

            pose_arr = lm_to_array(res.pose_landmarks, 33)
            left_arr = lm_to_array(res.left_hand_landmarks, 21)
            right_arr = lm_to_array(res.right_hand_landmarks, 21)

            torso_dist = 1.0
            if res.pose_landmarks:
                ls = res.pose_landmarks.landmark[mp_hol.PoseLandmark.LEFT_SHOULDER.value]
                rs = res.pose_landmarks.landmark[mp_hol.PoseLandmark.RIGHT_SHOULDER.value]
                torso_dist = np.hypot(ls.x - rs.x, ls.y - rs.y)
                if torso_dist < 1e-4:
                    torso_dist = 1.0

            all_pts = np.concatenate([pose_arr, left_arr, right_arr], axis=0).astype(np.float32)
            all_pts[:, :2] /= torso_dist
            features.append(all_pts.flatten())

            if max_frames and len(features) >= max_frames:
                break
            frame_idx += 1

    cap.release()
    features = np.vstack(features) if features else np.zeros((0, 33*3+21*3+21*3))
    return features, fps

def find_repetition_start(features, fps, sample_rate=2,
                          min_window_sec=1.2, max_window_sec=4.0,
                          threshold_factor=1.3, min_sign_sec=1.2):
    """
    Detect when the repeated sign starts using adaptive sliding windows.
    Ensures the first full sign is captured.
    """
    n, d = features.shape
    if n < 2:
        return None

    eff_fps = fps / sample_rate
    min_len = int(round(min_window_sec * eff_fps))
    max_len = int(round(max_window_sec * eff_fps))
    min_sign_frames = int(round(min_sign_sec * eff_fps))

    ref_window = features[:min_len]
    ref_variability = np.mean(np.linalg.norm(ref_window - ref_window[0], axis=1))
    adaptive_threshold = max(ref_variability * threshold_factor, 0.02)

    for start in range(min_sign_frames, n - min_len + 1):
        for window_len in range(min_len, max_len + 1):
            end = start + window_len
            if end > n:
                continue
            cand_window = features[start:end]
            dist, _ = fastdtw(ref_window, cand_window, dist=euclidean)
            norm_dist = dist / (window_len * d)
            if norm_dist < adaptive_threshold:
                return start * sample_rate / fps

    return None

def trim_video_at(input_path, output_path, end_time_sec):
    """Trim video using ffmpeg from 0 to end_time_sec."""
    if end_time_sec <= 0:
        copy2(input_path, output_path)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", "0",
        "-to", f"{end_time_sec:.3f}",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, check=True)

def process_and_trim(input_path, output_path,
                     sample_rate=2,
                     min_window_sec=1.2,
                     max_window_sec=4.0,
                     threshold_factor=1.3,
                     buffer_sec=1.5,
                     min_sign_sec=1.2):
    print("Extracting pose/hand features...")
    features, fps = extract_pose_hand_features(input_path, sample_rate=sample_rate)
    print(f"Extracted {features.shape[0]} frames at fps={fps}")

    cap = cv2.VideoCapture(input_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()

    # Scale factor based on video length, capped for stability
    scale_factor = min(total_duration / 10.0, 3.0)  # Relative to 10 seconds baseline

    adj_min_sign_sec = max(min_sign_sec, scale_factor * min_sign_sec)
    adj_max_window_sec = max(max_window_sec, scale_factor * max_window_sec)
    adj_buffer_sec = max(buffer_sec, scale_factor * buffer_sec)

    print(f"Video length: {total_duration:.2f}s")
    print(f"Adjusted min_sign_sec: {adj_min_sign_sec:.2f}s")
    print(f"Adjusted max_window_sec: {adj_max_window_sec:.2f}s")
    print(f"Adjusted buffer_sec: {adj_buffer_sec:.2f}s")

    rep_time = find_repetition_start(
        features,
        fps,
        sample_rate=sample_rate,
        min_window_sec=min_window_sec,
        max_window_sec=adj_max_window_sec,
        threshold_factor=threshold_factor,
        min_sign_sec=adj_min_sign_sec
    )

    if rep_time is None:
        print("No repetition detected, keeping original video.")
        copy2(input_path, output_path)
    else:
        cut_time = rep_time + adj_buffer_sec
        cut_time = min(cut_time, total_duration)  # Ensure cut_time does not exceed video length
        print(f"Repetition detected at {rep_time:.2f}s, trimming to {cut_time:.2f}s")
        trim_video_at(input_path, output_path, cut_time)

if __name__ == "__main__":
    # Update these paths for your input and output videos
    in_v = r"C:/Users/priya/Documents/final_year_project/FYP/youtube_videos/Abstract of Account.mp4"
    out_v = r"C:/Users/priya/Documents/final_year_project/FYP/youtube_videos/trimmed_Abstract of Account.mp4"

    process_and_trim(
        in_v,
        out_v,
        sample_rate=2,
        min_window_sec=1.2,
        max_window_sec=4.0,
        threshold_factor=1.3,
        buffer_sec=1.5,
        min_sign_sec=1.2
    )
