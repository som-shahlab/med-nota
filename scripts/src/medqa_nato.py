#!/usr/bin/env python3
"""
For each question in medqa_data.csv, we do 5 evaluations:
  1) No CoT
  2) Original CoT
  3) NATO COT

Usage:
  python3 medqa_nato.py --platform gpt --num_questions 10

Where `platform` can be "gpt", "claude", "gemini", "llama", "deepseek" or "o3-mini"
"""

import argparse
import csv
import json
import time
import requests
import re
import sys
import math

import config  

# Primary Model Endpoints
GPT_DEPLOYMENT_NAME = config.DEPLOYMENT_NAME
GPT_API_VERSION     = config.API_VERSION
GPT_API_BASE        = config.OPENAI_API_BASE
GPT_API_KEY         = config.OPENAI_API_KEY

CLAUDE_API_BASE = "https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa"
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
CLAUDE_API_KEY  = GPT_API_KEY  

GEMINI_API_BASE = "https://apim.stanfordhealthcare.org/gcpgemini/apim-gcp-oauth-fa"
GEMINI_MODEL_ID = "gemini-1.5-pro"
GEMINI_API_KEY  = GPT_API_KEY

GEMINI_FLASH_API_BASE = "https://apim.stanfordhealthcare.org/gcp-gem20flash-fa/apim-gcp-gem20flash-fa"
GEMINI_FLASH_MODEL_ID = "gemini-2.0-flash"
GEMINI_FLASH_API_KEY = GPT_API_KEY

LLAMA_API_BASE  = "https://apim.stanfordhealthcare.org/llama3370b/v1/chat/completions"
LLAMA_MODEL_ID  = "Llama-3.3-70B-Instruct"
LLAMA_API_KEY   = GPT_API_KEY

O3_MINI_API_BASE = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/o3-mini/chat/completions?api-version=2024-12-01-preview"
O3_MINI_MODEL    = "o3-mini"
O3_MINI_API_KEY  = GPT_API_KEY

DEEPSEEK_R1_API_BASE = "https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions"
DEEPSEEK_R1_MODEL_ID = "deepseek-chat"

INPUT_FILE  = "../../data/medqa_data.csv"
OUTPUT_FILE = "../../data/medqa_nato_results_{}.csv"

NO_COT_PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    "You are an experienced physician. Do not explain your reasoning. "
    "Only output your final answer in this exact format:\nAnswer: [Letter]"
)

ORIGINAL_COT_PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    "You are an experienced physician. Provide a detailed step-by-step reasoning, "
    "Make sure to output your final answer in this exact format:\nAnswer: [Letter]"
)

NOTO_COT_PROMPT = (
    "{question}\n\nOptions:\n{noto_options}\n\n"
    "You are an experienced physician. Provide a detailed step-by-step reasoning, "
    "then conclude with your final answer in the exact format:\nAnswer: [Letter]"
    "Make sure to output your final answer in this exact format:\nAnswer: [Letter]"
)

# LLM calls through the azure endpoint
def call_llm_with_retries(platform, messages=None, prompt_text=None, max_tokens=4000, temperature=0.0):
    """
    Calls the chosen LLM [gpt, claude, gemini, llama].
    Returns (success, text, error).
    """
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        retry_count += 1
        try:
            # GPT
            if platform.lower() == "gpt":
                url = f"{GPT_API_BASE}/deployments/{GPT_DEPLOYMENT_NAME}/chat/completions?api-version={GPT_API_VERSION}"
                headers = {
                    "Content-Type": "application/json",
                    "Ocp-Apim-Subscription-Key": GPT_API_KEY,
                }
                data = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if messages:
                    data["messages"] = messages
                else:
                    data["messages"] = [{"role": "user", "content": prompt_text}]

                resp = requests.post(url, headers=headers, json=data)
                resp.raise_for_status()
                j = resp.json()
                text = j["choices"][0]["message"]["content"].strip()
                return (True, text, "")

            elif platform.lower() == "claude":
                url = CLAUDE_API_BASE
                headers = {
                    "Ocp-Apim-Subscription-Key": CLAUDE_API_KEY,
                    "Content-Type": "application/json"
                }
                data = {
                    "model_id": CLAUDE_MODEL_ID,
                    "prompt_text": prompt_text if prompt_text else (messages[0]["content"] if messages else "")
                }
                resp = requests.post(url, headers=headers, data=json.dumps(data))
                resp.raise_for_status()
                j = resp.json()
                if "content" in j and isinstance(j["content"], list):
                    combined = ""
                    for item in j["content"]:
                        if isinstance(item, dict) and "text" in item:
                            combined += item["text"]
                    if not combined:
                        combined = j.get("completion", "[Claude structure unknown]")
                    text = combined.strip()
                else:
                    text = j.get("completion", "[Claude structure unknown]")
                return (True, text, "")
            
            elif platform.lower() == "o3-mini":
                url = O3_MINI_API_BASE
                headers = {
                    "Content-Type": "application/json",
                    "Ocp-Apim-Subscription-Key": O3_MINI_API_KEY,
                }
                # Only include the required messages parameter, other parameters are not supported
                data = {}
                if messages:
                    data["messages"] = messages
                else:
                    data["messages"] = [{"role": "user", "content": prompt_text}]

                resp = requests.post(url, headers=headers, json=data, timeout=60)
                                
                resp.raise_for_status()
                
                j = resp.json()
                if "choices" in j and len(j["choices"]) > 0 and "message" in j["choices"][0]:
                    text = j["choices"][0]["message"]["content"].strip()
                    return (True, text, "")
                else:
                    return (False, "", f"Unexpected response structure: {json.dumps(j)[:200]}...")

            elif platform.lower() == "gemini":
                url = GEMINI_API_BASE
                headers = {
                    "Ocp-Apim-Subscription-Key": GEMINI_API_KEY,
                    "Content-Type": "application/json"
                }
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt_text if prompt_text else messages[0]["content"]}]
                        }
                    ]
                }
                resp = requests.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                j = resp.json()
                full_text = ""
                if isinstance(j, list):
                    for chunk in j:
                        if "candidates" in chunk and isinstance(chunk["candidates"], list):
                            for c in chunk["candidates"]:
                                if "content" in c and "parts" in c["content"]:
                                    for part in c["content"]["parts"]:
                                        if "text" in part:
                                            full_text += part["text"]
                if not full_text:
                    full_text = "[Gemini structure unknown]"
                return (True, full_text.strip(), "")
            
            elif platform.lower() == "gemini-flash":
                url = GEMINI_FLASH_API_BASE
                headers = {
                    "Ocp-Apim-Subscription-Key": GEMINI_FLASH_API_KEY,
                    "Content-Type": "application/json"
                }
                user_content = prompt_text if prompt_text else (messages[0]["content"] if messages else "")
                payload = {
                    "contents": {
                        "role": "user",
                        "parts": {
                            "text": user_content
                        }
                    },
                    "safety_settings": {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    "generation_config": {
                        "temperature": temperature,
                        "topP": 0.8,
                        "topK": 40
                    }
                }

                resp = requests.post(url, headers=headers, data=json.dumps(payload))
                resp.raise_for_status()
                chunks = resp.json()

                # Concatenate text from all parts:
                full_answer = []
                for chunk in chunks:
                    if "candidates" in chunk:
                        for candidate in chunk["candidates"]:
                            content = candidate.get("content", {})
                            parts = content.get("parts", [])
                            for p in parts:
                                text_segment = p.get("text", "")
                                full_answer.append(text_segment)
                text = "".join(full_answer).strip()
                return (True, text, "")

            elif platform.lower() == "llama":
                url = LLAMA_API_BASE
                headers = {
                    "Ocp-Apim-Subscription-Key": LLAMA_API_KEY,
                    "Content-Type": "application/json"
                }
                data = {
                    "model": LLAMA_MODEL_ID,
                    "messages": messages if messages else [{"role": "user", "content": prompt_text}],
                    "max_tokens": 4000,
                    "temperature": 0.1
                }
                resp = requests.post(url, headers=headers, json=data, timeout=60)
                resp.raise_for_status()
                j = resp.json()
                text = j["choices"][0]["message"]["content"].strip()
                return (True, text, "")
            
            elif platform.lower() == "deepseek":
                url = DEEPSEEK_R1_API_BASE
                headers = {
                    "Ocp-Apim-Subscription-Key": GPT_API_KEY,  
                    "Content-Type": "application/json"
                }
                data = {
                    "model": DEEPSEEK_R1_MODEL_ID,
                    "messages": messages if messages else [{"role": "user", "content": prompt_text}],
                    "temperature": temperature,
                    "max_tokens": max_tokens if platform.lower() != "deepseek" else 32000,
                    "top_p": 1,
                    "stream": False
                }
                resp = requests.post(url, headers=headers, json=data)
                resp.raise_for_status()
                j = resp.json()
                text = j["choices"][0]["message"]["content"].strip()
                return (True, text, "")
            else:
                return (False, "", f"Unsupported platform: {platform}")
        except requests.exceptions.RequestException as e:
            err_msg = f"Request error: {e}"
            print(f"[Attempt {retry_count}/{max_retries}] {err_msg}", file=sys.stderr)
            if retry_count >= max_retries:
                return (False, "", err_msg)
            time.sleep(3)
        except Exception as e:
            err_msg = f"Unexpected error: {str(e)}"
            return (False, "", err_msg)

    return (False, "", "Max retries exceeded")


# Extract answer
def extract_answer_letter(text):
    # Split the text into nonempty, stripped lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Define a regex pattern that matches lines beginning with optional "**" or "##",
    # then an optional "final", then "answer:" (ignoring case), and captures everything after.
    pattern = re.compile(
        r'^(?:\*\*|##)?\s*(?:final\s+)?answer\s*:\s*(.+)$',
        re.IGNORECASE
    )

    # Iterate over lines in reverse (so that later answers override earlier ones)
    for line in reversed(lines):
        match = pattern.match(line)
        if match:
            candidate = match.group(1).strip()
            # Remove surrounding square brackets if present.
            if candidate.startswith('[') and candidate.endswith(']'):
                candidate = candidate[1:-1].strip()
            # Look for the first alphabetical character in the candidate.
            for char in candidate:
                if char.isalpha():
                    return char.upper()

    # As a fallback, you can include your previous regex approaches that search for asterisks
    pattern_asterisk = re.compile(
        r'\*\*\s*(?:final\s+)?answer\s*:\s*\[?([A-E])\]?',
        re.IGNORECASE
    )
    for line in reversed(lines):
        match = pattern_asterisk.search(line)
        if match:
            return match.group(1).upper()

    return ""

# Generate NOTO options by replacing correct option with "None of the other answers"
def create_noto_options(options_dict, correct_letter):
    """
    Replace the correct answer with "None of the other answers"
    Returns new options dictionary and the new correct letter (same as original)
    """
    # Create a copy to avoid modifying the original
    noto_options = options_dict.copy()
    
    # Replace the correct answer with "None of the other answers"
    noto_options[correct_letter] = "None of the other answers"
    
    return noto_options


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate medQA with No CoT, Original CoT, and NATO approach"
    )
    parser.add_argument("--platform", type=str, default="gpt", 
                        help="Primary model: [gpt, claude, gemini, llama, deepseek]")
    parser.add_argument("--num_questions", type=int, default=10, 
                        help="Number of questions from medqa_data.csv")
    parser.add_argument("--skip_empty", action="store_true", default=False,
                        help="Skip questions with empty responses in accuracy calculations")
    args = parser.parse_args()

    # Format the output filename with the platform name
    global OUTPUT_FILE
    OUTPUT_FILE = OUTPUT_FILE.format(args.platform.lower())

    # Track total processed vs total valid responses
    processed_questions = 0
    
    # Use total counters only (remove valid and skipped counters)
    total_no_cot = total_cot = total_noto = 0
    correct_no_cot = correct_cot = correct_noto = 0
    
    # Counters for tracking answer changes (remain the same)
    total_cot_correct = 0
    noto_changed = 0
    noto_changed_when_orig_correct = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        fieldnames = [
            "question",
            "ground_truth_letter",

            # No CoT
            "no_cot_prompt",
            "no_cot_response",
            "no_cot_letter",
            "no_cot_correct",

            # Original CoT
            "original_cot_prompt",
            "original_cot_response",
            "original_cot_letter",
            "original_cot_correct",

            # NOTO Method
            "noto_options",
            "noto_prompt",
            "noto_response",
            "noto_letter",
            "noto_correct",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        reader = csv.DictReader(infile)
        for row in reader:
            if processed_questions >= args.num_questions:
                break
            processed_questions += 1

            question = row["question"].strip()
            gt_letter = row["answer"].strip().upper()
            options_dict = json.loads(row["options"])
            options_text = "\n".join([f"{k}: {v}" for k, v in options_dict.items()])

            # No COT
            no_cot_prompt = NO_COT_PROMPT.format(question=question, options=options_text)
            ok_no, resp_no, err_no = call_llm_with_retries(
                platform=args.platform,
                messages=[{"role": "user", "content": no_cot_prompt}],
                max_tokens=1500,
                temperature=0.0
            )
            if not ok_no:
                resp_no = f"[Error: {err_no}]"
            letter_no = extract_answer_letter(resp_no)
            total_no_cot += 1
            correct_no = 1 if letter_no == gt_letter else 0
            correct_no_cot += correct_no

            # Original CoT
            original_cot_prompt = ORIGINAL_COT_PROMPT.format(question=question, options=options_text)
            ok_cot, resp_cot, err_cot = call_llm_with_retries(
                platform=args.platform,
                messages=[{"role": "user", "content": original_cot_prompt}],
                max_tokens=1500,
                temperature=0.0
            )
            if not ok_cot:
                resp_cot = f"[Error: {err_cot}]"
            letter_cot = extract_answer_letter(resp_cot)
            total_cot += 1
            correct_cot_ = 1 if letter_cot == gt_letter else 0
            correct_cot += correct_cot_
            if correct_cot_:
                total_cot_correct += 1

            # NOTO Method
            noto_options_dict = create_noto_options(options_dict, gt_letter)
            noto_options_text = "\n".join([f"{k}: {v}" for k, v in noto_options_dict.items()])
            noto_prompt = NOTO_COT_PROMPT.format(question=question, noto_options=noto_options_text)
            
            ok_noto, resp_noto, err_noto = call_llm_with_retries(
                platform=args.platform,
                messages=[{"role": "user", "content": noto_prompt}],
                max_tokens=1500,
                temperature=0.0
            )
            if not ok_noto:
                resp_noto = f"[Error: {err_noto}]"
            letter_noto = extract_answer_letter(resp_noto)
            total_noto += 1
            this_question_noto_correct = 1 if letter_noto == gt_letter else 0
            correct_noto += this_question_noto_correct

            if letter_noto and letter_cot and (letter_noto != letter_cot):
                noto_changed += 1
                if letter_cot == gt_letter:
                    noto_changed_when_orig_correct += 1
            
            writer.writerow({
                "question": question,
                "ground_truth_letter": gt_letter,
                "no_cot_prompt": no_cot_prompt,
                "no_cot_response": resp_no,
                "no_cot_letter": letter_no,
                "no_cot_correct": correct_no,
                "original_cot_prompt": original_cot_prompt,
                "original_cot_response": resp_cot,
                "original_cot_letter": letter_cot,
                "original_cot_correct": correct_cot_,
                "noto_options": noto_options_text,
                "noto_prompt": noto_prompt,
                "noto_response": resp_noto,
                "noto_letter": letter_noto,
                "noto_correct": this_question_noto_correct,
            })

            # Print a short console summary
            print(f"\n[Q#{processed_questions}] {question}")
            print(f"  No CoT => {letter_no} (correct: {correct_no})")
            print(f"  Orig CoT => {letter_cot} (correct: {correct_cot_})")
            print(f"  NOTO Method => {letter_noto} (correct: {this_question_noto_correct})")
            print("-"*70)

    # Calculate statistics using total counters instead of counters
    acc_no_cot = correct_no_cot / total_no_cot if total_no_cot > 0 else 0
    acc_cot = correct_cot / total_cot if total_cot > 0 else 0
    acc_noto = correct_noto / total_noto if total_noto > 0 else 0

    print("\n=== Evaluation Complete! ===")
    print(f"Total questions processed: {processed_questions}")

    print("\n=== Accuracy Results (All Questions Included) ===")
    print(f"No CoT:      {acc_no_cot:.2%}  ({correct_no_cot}/{total_no_cot})")
    print(f"Original CoT: {acc_cot:.2%}  ({correct_cot}/{total_cot})")
    print(f"NOTO Method:  {acc_noto:.2%}  ({correct_noto}/{total_noto})")
    
    # Calculate NOTO vs Original Accuracy Drop (using response accuracy)
    noto_drop = (acc_cot - acc_noto) / acc_cot * 100 if acc_cot > 0 else 0
    print(f"\nAccuracy drop from Original to NOTO: {noto_drop:.2f}%")
    print(f"This indicates the model's reliance on pattern matching vs. true reasoning.")

    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()