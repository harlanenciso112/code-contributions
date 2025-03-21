import sys
import os
from transformers import pipeline
from bs4 import BeautifulSoup # type: ignore

# Load the toxicity detection model
classifier = pipeline("text-classification", model="unitary/toxic-bert")

def extract_text_from_html(file_path):
    """Extract visible text from an HTML file."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text(separator=" ").strip()  # Extract visible text

def split_text(text, max_length=512):
    """Split text into chunks of max_length tokens."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunks.append(" ".join(words[i:i+max_length]))
    return chunks

def check_text(text):
    """Check if any part of the text is toxic."""
    chunks = split_text(text)
    for chunk in chunks:
        result = classifier(chunk)  # Analyze the text
        label = result[0]["label"]  # Get the label (e.g., "toxic", "severe_toxic")
        score = result[0]["score"]  # Get the confidence score
        if label == "toxic" and score > 0.7:
            return True
    return False

def check_file(file_path):
    """Check if an HTML file contains toxic content."""
    text = extract_text_from_html(file_path)
    return check_text(text)

def main():
    offensive_content_found = False
    
    # Check only modified HTML files in the PR
    for file in os.popen("git diff --name-only HEAD^ HEAD").read().split():
        if file.endswith(".html") and os.path.exists(file):
            if check_file(file):
                print(f"❌ Offensive content detected in {file}")
                offensive_content_found = True

    if offensive_content_found:
        sys.exit(1)  # Block the PR
    else:
        print("✅ No offensive content found.")
        sys.exit(0)  # Approve the PR

if __name__ == "__main__":
    main()
