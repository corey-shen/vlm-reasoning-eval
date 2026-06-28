import base64
import io
import os
import time
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Connect to DeepSeek API
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

print("Loading Winoground dataset...")
dataset = load_dataset("facebook/winoground")
test_set = dataset["test"]

# Let's test a slice of 10 items to make sure our math adds up perfectly
test_subset = test_set.select(range(10))

# Track scores
results = {"total": 0, "text_correct": 0, "image_correct": 0, "group_correct": 0}

def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Helper function using DeepSeek to simulate the evaluation choice
def test_evaluate_image_with_deepseek(cap_0, cap_1, target_image_index):
    # Since DeepSeek-V4-Pro text models can't look at images, we pass the 
    # expected image index behind the scenes just to validate your loop logic.
    system_prompt = (
        "You are simulating a multi-modal evaluation judge for pipeline testing. "
        "You will be given two captions. Your job is to return exactly 'A' or 'B'. "
        f"For testing purposes, always pick the option that corresponds to image_{target_image_index}."
    )
    
    user_prompt = f"""Analyze the simulated image_{target_image_index}.
Which of the following captions accurately describes this image?

Option A: {cap_0}
Option B: {cap_1}

Respond with exactly one character, either 'A' or 'B', and nothing else."""

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.0,
            # If your tier requires these reasoning fields, keep them un-commented:
            # reasoning_effort="high",
            # extra_body={"thinking": {"type": "enabled"}}
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer
    except Exception as e:
        print(f"DeepSeek API Error: {e}. Retrying...")
        time.sleep(1)
        return None

print(f"\nStarting DeepSeek logic test loop over {len(test_subset)} samples...")

for idx, sample in enumerate(test_subset):
    print(f"Testing item {idx + 1}/{len(test_subset)}...")

    # We still run the encoding logic to make sure your PIL pipeline doesn't break
    b64_img_0 = pil_to_base64(sample["image_0"])
    b64_img_1 = pil_to_base64(sample["image_1"])

    cap_0 = sample["caption_0"]
    cap_1 = sample["caption_1"]

    # We simulate evaluating Image 0 (Where Ground Truth is 'A')
    choice_for_img_0 = test_evaluate_image_with_deepseek(cap_0, cap_1, target_image_index=0)

    # We simulate evaluating Image 1 (Where Ground Truth is 'B')
    choice_for_img_1 = test_evaluate_image_with_deepseek(cap_0, cap_1, target_image_index=1)

    if not choice_for_img_0 or not choice_for_img_1:
        continue

    # Convert choices to alignment flags
    match_0_0 = "A" in choice_for_img_0  # True match
    match_0_1 = "B" in choice_for_img_0  # False mismatch
    match_1_0 = "A" in choice_for_img_1  # False mismatch
    match_1_1 = "B" in choice_for_img_1  # True match

    # --- WINOGROUND SCORING METRICS ---
    text_score = match_0_0 and match_1_1 and not (match_1_0 or match_0_1)
    image_score = match_0_0 and match_1_1
    group_score = text_score and image_score

    # Record results
    results["total"] += 1
    if text_score:
        results["text_correct"] += 1
    if image_score:
        results["image_correct"] += 1
    if group_score:
        results["group_correct"] += 1

# Final calculation and reporting
total = results["total"]
if total > 0:
    text_acc = (results["text_correct"] / total) * 100
    image_acc = (results["image_correct"] / total) * 100
    group_acc = (results["group_correct"] / total) * 100

    print("\n=================================")
    print("   LOGIC TEST BENCHMARK REPORT   ")
    print("=================================")
    print(f"Total Samples Evaluated: {total}")
    print(f"Text Accuracy:           {text_acc:.2f}% (Expect 100.00% if logic is flawless)")
    print(f"Image Accuracy:          {image_acc:.2f}% (Expect 100.00% if logic is flawless)")
    print(f"Group Accuracy:          {group_acc:.2f}% (Expect 100.00% if logic is flawless)")
    print("=================================")