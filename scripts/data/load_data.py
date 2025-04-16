#!/usr/bin/env python3
"""
Loads the bigbio/med_qa dataset from Hugging Face,
processes each record to extract:
 - question: the question text
 - answer: the ground truth answer (converted to a letter A/B/C/D/E)
 - options: answer options (converted to a mapping letter -> option text)
and then saves the data to a CSV file.
"""

import csv
import json
from datasets import load_dataset

OUTPUT_FILE = "../../data/medqa_data.csv"

def process_options(options):
    """
    Convert the options field to a dictionary mapping letter -> option text.
    If options is a list of dicts with 'key' and 'value', use them.
    Otherwise, assume it is a list of strings and assign letters A, B, C...
    """
    if isinstance(options, list):
        # If the first item is a dict with 'key'/'value', assume [ {key: X, value: Y}, ... ]
        if options and isinstance(options[0], dict) and 'key' in options[0] and 'value' in options[0]:
            options_dict = {opt['key'].strip().upper(): opt['value'].strip() for opt in options}
        else:
            # Plain list of strings: assign letters A, B, C...
            options_dict = {chr(65 + i): opt.strip() for i, opt in enumerate(options)}
    elif isinstance(options, dict):
        # Already letter->text form, just ensure keys/values are strings
        options_dict = {str(k).strip().upper(): str(v).strip() for k, v in options.items()}
    else:
        options_dict = {}
    return options_dict

def match_answer_letter(full_answer_text, options_dict):
    """
    Given the full text of the correct answer (e.g. 'Disclose the error...'),
    find which letter in options_dict has the same text. Comparison is case-insensitive
    and ignores leading/trailing whitespace.
    """
    answer_lower = full_answer_text.strip().lower()
    for letter, text in options_dict.items():
        if text.strip().lower() == answer_lower:
            return letter  # Return 'A', 'B', 'C', etc.
    # If no perfect match, return empty or do some fallback.
    return ""

def main():
    print("Loading the bigbio/med_qa dataset ...")
    dataset = load_dataset("bigbio/med_qa", split="test")
    total = len(dataset)
    print(f"Total questions in dataset: {total}")

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["question", "answer", "options"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in dataset:
            question = record.get("question", "").strip()
            full_answer_text = record.get("answer", "").strip()
            options = record.get("options", None)
            if not question or not full_answer_text or not options:
                continue

            options_dict = process_options(options)
            # Convert the full correct answer text to a letter (A, B, C, D, ...)
            letter = match_answer_letter(full_answer_text, options_dict)
            if not letter:
                # If there's no match, skip or handle however you like
                continue

            # Write the letter to "answer" in the CSV instead of the entire text
            writer.writerow({
                "question": question,
                "answer": letter,
                "options": json.dumps(options_dict)
            })

    print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
