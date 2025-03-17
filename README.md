# Evaluation of Medical Chain-of-Thought of LLMs

This project tries to identify if faulty CoTs leads to a drop in model's output accuracy. Please note, for the `medqa_eval.py` script to work, you have to specify the `OPENAI_API_KEY` in `config.py`

---

## Quick Start

```bash
# Setup environment
conda env create -f environment.yaml
conda activate cot-eval

# Data preprocessing
python3 scripts/load_data.py

# Evaluating model performance on manipulated CoT
python3 scripts/medqa_eval.py --platform gpt --num_questions 10