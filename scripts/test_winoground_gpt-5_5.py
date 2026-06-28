import base64
import io
import os
import time
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)

# 1. Load Dataset
print("Loading Winoground dataset...")
dataset = load_dataset("facebook/winoground")
test_set = dataset["test"]

# For safety and debugging, you can slice this (e.g., test_set = test_set.select(range(10)))
# To run the entire dataset, leave it as: test_set = dataset["test"]
# LIMITING TO 5 FOR INITIAL TESTING:
test_subset = test_set.select(range(10))  # Change to test_set for the full run

# Track scores
results = {"total": 0, "text_correct": 0, "image_correct": 0, "group_correct": 0}


# Helper function to convert a PIL Image to base64
def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Helper function to ask the VLM if an image matches a caption
def check_match(base64_image, caption):
    prompt = f"""Analyze the provided image and the caption below.
Caption: "{caption}"

Does this caption accurately describe what is happening in the image? 
Answer with exactly 'YES' or 'NO' and nothing else."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90b-Vision-Instruct",   # model="gpt-4o-mini"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"     # Forces 85 tokens/image
                            },
                        },
                    ],
                }
            ]
            #temperature=0.0
            #max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"API Error: {e}. Retrying in 2 seconds...")
        time.sleep(2)
        return False


print(f"\nStarting evaluation loop over {len(test_subset)} samples...")

for idx, sample in enumerate(test_subset):
    print(f"Evaluating item {idx + 1}/{len(test_subset)}...")

    # Encode both images
    b64_img_0 = pil_to_base64(sample["image_0"])
    b64_img_1 = pil_to_base64(sample["image_1"])

    cap_0 = sample["caption_0"]
    cap_1 = sample["caption_1"]

    # Evaluate all 4 possible combinations
    # Format: match_ImageIndex_CaptionIndex
    match_0_0 = check_match(b64_img_0, cap_0)  # Should be True
    match_0_1 = check_match(b64_img_0, cap_1)  # Should be False
    match_1_0 = check_match(b64_img_1, cap_0)  # Should be False
    match_1_1 = check_match(b64_img_1, cap_1)  # Should be True

    # --- WINOGROUND SCORING METRICS ---

    # Text Score: Does the model correctly identify the right image for each caption?
    # For cap_0, it must prefer img_0 over img_1. For cap_1, it must prefer img_1 over img_0.
    text_score = (match_0_0 and not match_1_0) and (
        match_1_1 and not match_0_1
    )

    # Image Score: Does the model correctly identify the right caption for each image?
    # For img_0, it must prefer cap_0 over cap_1. For img_1, it must prefer cap_1 over cap_0.
    image_score = (match_0_0 and not match_0_1) and (
        match_1_1 and not match_1_0
    )

    # Group Score: Complete structural understanding (Both Text and Image must be correct)
    group_score = text_score and image_score

    # Record results
    results["total"] += 1
    if text_score:
        results["text_correct"] += 1
    if image_score:
        results["image_correct"] += 1
    if group_score:
        results["group_correct"] += 1

    # Optional: Small cooldown to avoid aggressive rate-limiting
    time.sleep(0.2)

# 3. Final calculation and reporting
total = results["total"]
if total > 0:
    text_acc = (results["text_correct"] / total) * 100
    image_acc = (results["image_correct"] / total) * 100
    group_acc = (results["group_correct"] / total) * 100

    print("\n=================================")
    print("      FINAL BENCHMARK REPORT     ")
    print("=================================")
    print(f"Total Samples Evaluated: {total}")
    print(f"Text Accuracy:           {text_acc:.2f}%")
    print(f"Image Accuracy:          {image_acc:.2f}%")
    print(f"Group Accuracy:          {group_acc:.2f}%")
    print("=================================")
print(f"Results: {results}")