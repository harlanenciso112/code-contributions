import sys
import os
from transformers import pipeline

# Load the toxicity detection model from Hugging Face
classifier = pipeline("text-classification", model="unitary/toxic-bert")

def check_text(text):
    result = classifier(text)  # Analyze the text
    label = result[0]["label"]  # Get the label (e.g., "toxic", "severe_toxic")
    score = result[0]["score"]  # Get the confidence score
    
    if label == "toxic" and score > 0.7:  # If toxic and confidence > 70%, block PR
        return True
    return False

def check_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        return check_text(content)

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
