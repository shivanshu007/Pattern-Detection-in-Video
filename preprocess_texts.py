import re

def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        processed_text = re.sub(r'\D', '', text)
        if len(processed_text) == 10:
            processed_texts.append(processed_text)
    return processed_texts
