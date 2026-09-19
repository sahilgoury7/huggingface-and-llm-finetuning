# 03: QLoRA & Supervised Fine-Tuning (SFT)

> **Gen AI Batch 9 | Part 2 (Advanced Fine-Tuning)**  
> **Instructor:** Ashish Jangra  

---

## 📑 Syllabus & Topics

1. **Low Rank Adaptation (LoRA)**
   - Matrix decomposition: $W = W_0 + \Delta W$, where $\Delta W = B \times A$.
   - Hyperparameters: Rank ($r$), Alpha ($\alpha$), target modules (`q_proj`, `v_proj`, etc.).
   - Trainable parameters reduced from 100% to $< 1\%$.
2. **Quantization**
   - FP32 $\to$ FP16 $\to$ INT8 $\to$ 4-bit NormalFloat (NF4).
   - `bitsandbytes` 4-bit quantization and double quantization.
3. **Supervised Fine-Tuning (SFT)**
   - Instruction formatting (Instruction / Prompt $\to$ Response).
   - Training loss calculated strictly on model completions (response-only loss).
4. **Unsloth Acceleration**
   - Faster execution, 70% VRAM memory reduction on Google Colab T4 (16GB).

---

## 🎯 Capstone Project: Data Science Interview QA Assistant
- **Target Task:** Fine-tuning an open-source LLM into a technical interview prep assistant.
- **Dataset:** [`data_science.csv`](https://github.com/AshishJangra27/datasets/tree/main/Intervew%20Questions).
- **Deliverables:**
  - Tokenization & prompt formatting pipeline.
  - 4-bit QLoRA training run using `Trainer` / `SFTTrainer`.
  - Inference tests on unseen Data Science questions.
  - Merged weights / GGUF local export.

---

## 📓 Notebook Reference
- [LLM_Finetunning.ipynb](./LLM_Finetunning.ipynb)
- [Day 4 Notes: PEFT, LoRA Math & Quantization](./day4_peft_lora_quantization_notes.md)
- [Day 5 Notes: Supervised Fine-Tuning (SFT) & Unsloth](./day5_sft_unsloth_notes.md)
