import streamlit as st
import pytesseract
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
from tqdm import tqdm  # For progress bar
from extract_frames import extract_frames
from extract_text import extract_text_from_frames

# Specify the path to the trained model
model_path = 'fine_tuned_bert.pth'

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_path))

st.title("10-Digit Number Detection from Video Frames")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Use a flag to track if frames have been extracted
    frames_extracted = False

    # Extract frames only once
    if not frames_extracted:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.video(uploaded_file)

        # Extract frames
        frames_folder = "frames_folder"
        extract_frames("uploaded_video.mp4", frames_folder)
        frames_extracted = True  # Update flag after extraction

    # Display frames and predictions
    pattern_found = False
    progress_bar = st.progress(0)
    frame_files = sorted(os.listdir(frames_folder))

    for i, frame in enumerate(tqdm(frame_files, desc="Analyzing Frames", unit="frame")):
        if pattern_found:
            break

        img = Image.open(os.path.join(frames_folder, frame))
        st.image(img, caption=frame, use_column_width=True)

        # Extract text using OCR
        text = pytesseract.image_to_string(img)
        st.write(f"Extracted Text: {text}")

        # Predict 10-digit number
        processed_text = re.sub(r'\D', '', text)
        if len(processed_text) == 10:
            inputs = tokenizer(processed_text, return_tensors="pt")
            outputs = model(**inputs)
            prediction = outputs.logits.argmax().item()

            if prediction == 1:  # Assuming label 1 indicates a valid pattern
                st.write("Pattern has been found!")
                pattern_found = True
                break

        # Update progress bar
        progress_bar.progress((i + 1) / len(frame_files))

    progress_bar.empty()  # Remove progress bar after completion

    if not pattern_found:
        st.write("No pattern found in the video.")
