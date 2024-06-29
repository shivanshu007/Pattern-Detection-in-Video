import streamlit as st
import pytesseract
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import os
from tqdm import tqdm  # For progress bar
from extract_frames import extract_frames

# Load the model and tokenizer and move to CPU
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert')
model = BertForSequenceClassification.from_pretrained('fine_tuned_bert').to(device)

st.title("10-Digit Number Detection from Video Frames")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Extract frames
    frames_folder = "frames_folder"
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(uploaded_file)

    extract_frames("uploaded_video.mp4", frames_folder)

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

        # Detect any 10-digit number
        digit_sequences = re.findall(r'\d{10}', text)
        for sequence in digit_sequences:
            inputs = tokenizer(sequence, return_tensors="pt").to(device)
            outputs = model(**inputs)
            prediction = outputs.logits.argmax().item()

            if prediction == 1:  # Assuming label 1 indicates a valid pattern
                st.write(f"Pattern has been found: {sequence}")
                pattern_found = True
                break

        # Update progress bar
        progress_bar.progress((i + 1) / len(frame_files))

    progress_bar.empty()  # Remove progress bar after completion

    if not pattern_found:
        st.write("No pattern found in the video.")
