import pytesseract
from PIL import Image
import os

def extract_text_from_frames(frames_folder):
    texts = []
    for frame in sorted(os.listdir(frames_folder)):
        img = Image.open(os.path.join(frames_folder, frame))
        text = pytesseract.image_to_string(img)
        texts.append(text)
    return texts
