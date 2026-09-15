# 02: Transfer Learning with Hugging Face

> **Gen AI Batch 9 | Part 2 (Foundations)**  
> **Instructor:** Ashish Jangra  

---

## 📑 Syllabus & Topics

1. **What is Transfer Learning?**
   - Pretraining vs Fine-tuning.
   - Reusing pre-trained weights and training task-specific heads.
2. **The TrainingArgs Function**
   - Configuring `TrainingArguments`:
     - Learning rate, weight decay, warmup steps.
     - Batch sizes, gradient accumulation.
     - Logging steps, save strategy, and evaluation frequency.
3. **The Trainer API**
   - HF `Trainer` class architecture.
   - Passing `model`, `args`, `train_dataset`, `eval_dataset`, `tokenizer`, and `data_collator`.
   - Running `trainer.train()`.

---

## 🎯 Project: Transfer Learning Project Setup
- **Dataset:** [`data_science.csv`](https://github.com/AshishJangra27/datasets/tree/main/Intervew%20Questions)
- **Goal:** Preprocessing domain QA dataset, tokenizing with dynamic padding, and feeding it into the Hugging Face `Trainer` pipeline.

---

## 📓 Notebook Reference
- [LLM_Finetunning.ipynb](./LLM_Finetunning.ipynb)
