# 🧠 Med-NOTA: Evaluating "None of the other Answers" in Medical QA

A simple tool for analyzing how well language models handle **"None of the other Answers" (NOTA)** options in medical question answering, especially under **Chain-of-Thought (CoT)** reasoning.

---

## 📌 What This Does

This project investigates whether large language models (LLMs) like **GPT**, **Claude**, **Deepseek-R1**, and others can reliably identify when **none of the answer choices are correct** in medical multiple-choice questions. It compares performance with and without the need to recognize NOTA.

---

## 🚀 Quick Start

### 1. Set up your environment
```bash
conda env create -f environment.yaml
conda activate cot-eval
```
### 2. Process the data

```bash
cd scripts/data
python3 load_data.py
```
### 3. Run the NOTA experiments

```bash
cd ../src
python3 medqa_nato.py
```
### 4. Analyze the results

```bash
python3 nota_accuracy_stats.py
```
