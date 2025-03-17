# Evaluation of Medical Chain-of-Thought of LLMs

This project tries to identify if faulty CoTs leads to a drop in model's output accuracy

---

## Quick Start

```bash
# Setup environment
conda env create -f environment.yaml
conda activate cot-eval

# Data preprocessing
python3 scripts/load_data.py

# Evaluating model performance on manipulated CoT
python3 medqa_eval.py --platform gpt --num_questions 10