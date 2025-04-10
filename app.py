import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pickle
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import speech_recognition as sr
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model

@st.cache_resource
def load_audio_model():
    audio_model = load_model("audio_emotion.h5")
    with open("label_encoder.pkl", "rb") as f:
        audio_label_encoder = pickle.load(f)
    return audio_model, audio_label_encoder

@st.cache_resource
def load_text_model():
    text_model_dir = "trnsformer"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_dir)
    text_model = AutoModelForSequenceClassification.from_pretrained(text_model_dir)
    text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model.to(text_device)
    text_label_map = {0: "sad", 1: "happy", 2: "love", 3: "angry", 4: "fear", 5: "surprise"}
    return text_model, text_tokenizer, text_device, text_label_map

@st.cache_resource
def load_video_model():
    video_model = load_model("video_emotion.h5")
    video_reverse_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    return video_model, video_reverse_map

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.85):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sample_rate):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    return mfcc

def transform_audio(data, fns, sample_rate):
    fn = random.choice(fns)
    if fn == pitch:
        return fn(data, sample_rate)
    elif fn == "None":
        return data
    elif fn in [noise, stretch]:
        return fn(data)
    else:
        return data

def get_features(audio_path):
    data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
    fns = [noise, pitch, "None"]
    features_list = []
    for _ in range(3):
        data_aug = transform_audio(data, fns, sample_rate)
        mfcc = extract_features(data_aug, sample_rate)
        mfcc = mfcc[:, :108]
        features_list.append(mfcc)
    return features_list

def predict_audio_emotion(audio_path, audio_model, audio_label_encoder):
    features = get_features(audio_path)
    predictions = []
    for feat in features:
        feat = np.expand_dims(feat, axis=0)
        feat = np.expand_dims(feat, axis=3)
        feat = np.swapaxes(feat, 1, 2)
        pred = audio_model.predict(feat)
        predictions.append(pred)
    avg_pred = np.mean(predictions, axis=0)
    emotion = audio_label_encoder.inverse_transform(avg_pred)[0]
    return emotion[0]

def predict_text_emotion(text, text_model, text_tokenizer, text_device, text_label_map):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(text_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
        logits = outputs.logits
    pred = logits.argmax(dim=-1).item()
    return text_label_map.get(pred, "Unknown")

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio_data)
    except Exception as e:
        transcription = f"Transcription failed: {e}"
    return transcription

def extract_audio(video_path, output_audio_path="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path

def extract_video_sequence(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        st.error("Unable to read frames from video.")
        return None
    if total_frames < num_frames:
        indices = range(total_frames-1)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames_dict = {}
    frame_idx = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            frames_dict[frame_idx] = normalized
        frame_idx += 1
    cap.release()
    try:
        seq = [frames_dict[i] for i in indices]
    except KeyError:
        st.error("Error extracting video frames. Please try a different video.")
        return None
    seq = np.array(seq)
    seq = np.expand_dims(seq, axis=-1)
    return seq

def predict_video_emotion(video_path, video_model, video_reverse_map):
    seq = extract_video_sequence(video_path, num_frames=100000)
    if seq is None:
        return "Unknown"

    smaller_sequences = [np.expand_dims(seq[i:i+10], axis=0) for i in range(0, len(seq), 10)]
    predictions = []
    
    for smaller_seq in smaller_sequences:
        pred = video_model.predict(smaller_seq)
        predicted_class = np.argmax(pred, axis=1)[0]
        predictions.append(predicted_class)

    most_common_prediction = Counter(predictions).most_common(1)[0][0]
    return video_reverse_map.get(most_common_prediction, "Unknown")

def weighted_voting(audio_emotion, video_emotion, text_emotion, audio_weight=2, video_weight=1, text_weight=1):
    all_possible_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'Unknown']
    weights = {emotion: 0 for emotion in all_possible_emotions}
    weights[audio_emotion] += audio_weight
    weights[video_emotion] += video_weight
    weights[text_emotion] += text_weight
    max_weight = max(weights.values())
    max_emotions = [emotion for emotion, weight in weights.items() if weight == max_weight]
    if len(max_emotions) > 1:
        if audio_emotion in max_emotions:
            return video_emotion
    final_prediction = max(weights, key=weights.get)
    return final_prediction

def main():
    st.title("Multimodal Emotion Classification")
    st.write("Upload a video file to extract audio, transcribe it, and classify emotions from audio, text, and video frames.")

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            video_path = temp_video.name

        st.video(video_path)

        if st.button("Predict"):
            audio_temp_path = "temp_audio.wav"
            extract_audio(video_path, audio_temp_path)

            transcription = transcribe_audio(audio_temp_path)

            audio_model, audio_label_encoder = load_audio_model()
            audio_emotion = predict_audio_emotion(audio_temp_path, audio_model, audio_label_encoder)

            text_model, text_tokenizer, text_device, text_label_map = load_text_model()
            text_emotion = predict_text_emotion(transcription, text_model, text_tokenizer, text_device, text_label_map)

            video_model, video_reverse_map = load_video_model()
            video_emotion = predict_video_emotion(video_path, video_model, video_reverse_map)

            st.info("Combining results...")
            final_prediction = weighted_voting(audio_emotion, video_emotion, text_emotion)
            st.markdown("**Final Emotion Prediction:**")
            st.write(final_prediction)

            try:
                os.remove(audio_temp_path)
                os.remove(video_path)
            except Exception as e:
                st.warning(f"Cleanup error: {e}")

if __name__ == "__main__":
    main()