# 📅 7-Day Intensive Learning Roadmap & Progress Tracker

> **Track:** Gen AI Batch 9 — Hugging Face & LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Daily Time Commitment:** ⚡ **5 – 6 Hours / Day**  
> **Target Goal:** Zero to Production-Ready LLM Fine-Tuning (Pipelines, AutoClasses, Transfer Learning, LoRA, QLoRA, and Unsloth on Colab T4).

---

## ⏰ 5–6 Hour Daily Study Formula
Har din ke 5–6 ghante ko hum **3 structured blocks** me divide karenge:
* 🧠 **Block 1 (1.5 – 2 Hours): Deep Theory & Mental Model** (Concept, architecture, mathematical intuition).
* 💻 **Block 2 (2.5 – 3 Hours): Hands-on Code & Debugging** (Notebook practice, live errors solve karna, experiments).
* 📝 **Block 3 (1 Hour): 3-Part Notes & Interview Quiz** (Glossary, Important Things, Q&A update + self-test).

---

## 📊 7-Day Master Schedule Overview

| Day | Date | Module / Phase | Core Deliverables | Status |
| :---: | :---: | :--- | :--- | :---: |
| **Day 1** | **2026-09-15** | **Phase 1: HF Ecosystem, Pipelines & AutoClasses** | 5 Pipelines run, AutoModel QA, 3-Part Notes compiled | 🟢 **Completed** |
| **Day 2** | **2026-09-16** | **Phase 2 & 3: Tokenization Mechanics & Datasets** | BPE/WordPiece, Padding/Masking, `datasets` `.map()` | 🟡 **Next Up** |
| **Day 3** | **2026-09-17** | **Phase 4: 4 Real-World Portfolio Mini-Projects** | Sentiment tool, NER parser, Summarizer, Zero-shot bot | ⚪ Pending |
| **Day 4** | **2026-09-18** | **Phase 5 & 6: Transfer Learning & HF Trainer API** | `TrainingArguments`, `Trainer` loop, Metrics & Eval | ⚪ Pending |
| **Day 5** | **2026-09-19** | **Phase 7: PEFT, LoRA Mathematics & 4-Bit Quantization** | Matrix decomposition ($A \times B$), NF4 `bitsandbytes` | ⚪ Pending |
| **Day 6** | **2026-09-20** | **Phase 8: SFT & Unsloth Acceleration on Colab T4** | Prompt formatting, Completion loss, Unsloth setup | ⚪ Pending |
| **Day 7** | **2026-09-21** | **Phase 9: Capstone: Interview QA Assistant Fine-Tuning** | `data_science.csv` training run, GGUF/LoRA export | ⚪ Pending |

---

## 🗓️ Day-by-Day 5–6 Hour Action Plan

---

### 🟢 Day 1 (Tuesday, 2026-09-15) — HF Ecosystem & Pipelines
* **Status:** 🟢 **COMPLETED (100%)**
* **Time Spent:** ~5 Hours
* **Covered Topics:**
  - [x] Hugging Face 4 Pillars (Models, Datasets, Spaces, Transformers)
  - [x] Model Cards (Architecture, VRAM limits, License checking: Apache 2.0 vs CC-BY-NC)
  - [x] Inference API vs Local Execution
  - [x] Under the hood of `pipeline()`: Tokenizer $\to$ Forward Pass $\to$ Softmax
  - [x] `pipeline()` vs `AutoTokenizer + AutoModel` (Automatic vs Manual car analogy)
  - [x] Hands-on Implementation:
    - Sentiment Analysis (Single & Batching)
    - NER with `aggregation_strategy="simple"`
    - Extractive QA with `AutoModelForQuestionAnswering` & `torch.argmax()`
    - Summarization with `AutoModelForSeq2SeqLM` & Beam Search (`num_beams=4`)
    - Zero-Shot Classification with NLI (Natural Language Inference)
  - [x] Debugging: Python 3.13 task strings & multimodal DocVQA errors.
  - [x] Structured 3-Part Notes compiled in `01_huggingface_pipelines/notes.md`.

---

### 🟡 Day 2 (Wednesday, 2026-09-16) — Tokenization & Datasets Deep Dive
* **Status:** 🟡 **NEXT UP**
* **Target Time:** 5 – 6 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2 (Theory):**
    - Word vs Character vs Sub-word tokenization (Why sub-words won NLP).
    - Algorithms: Byte-Pair Encoding (BPE), WordPiece (BERT), SentencePiece (Llama).
    - Special Tokens: `[CLS]`, `[SEP]`, `<s>`, `</s>`, `<pad>`, `<unk>`, BOS/EOS.
    - Vocabulary size, Token IDs vs Embeddings.
  - **Hours 3–4.5 (Hands-on Code):**
    - `AutoTokenizer.from_pretrained()` deep exploration.
    - Dynamic Padding vs Static Padding (`padding=True`, `max_length=...`).
    - Truncation strategies (`truncation=True`) and Attention Masks inspection.
    - Returning PyTorch Tensors (`return_tensors="pt"`).
    - The `datasets` library: `load_dataset()`, train/test splits, and `.map(batched=True)` pipeline.
  - **Hours 4.5–5.5 (Notes & Quiz):**
    - Notes in 3-Part format (Glossary, Important Things, Q&A).
    - 5 Interview Questions on Tokenization & Attention Masks.
* **Deliverable:** Working Python script / notebook cells demonstrating tokenization, attention masks, and `.map()` preprocessing.

---

### ⚪ Day 3 (Thursday, 2026-09-17) — 4 Production-Ready Mini-Projects
* **Status:** ⚪ **Upcoming**
* **Target Time:** 5 – 6 Hours
* **Syllabus & Schedule:**
  - **Hours 1–1.5 (Architecture & Design):**
    - Project structuring, clean input-output interfaces, error handling.
  - **Hours 1.5–4.5 (Building the 4 Mini-Projects):**
    1. **Project 1:** Customer Review & Sentiment Classifier Utility.
    2. **Project 2:** Resume & Contract Named Entity Extractor (NER).
    3. **Project 3:** Technical Article & Meeting Notes Summarizer Tool.
    4. **Project 4:** Zero-Shot Customer Support Ticket Router.
  - **Hours 4.5–5.5 (Testing & Documentation):**
    - Test edge cases, empty strings, and long inputs.
    - Document clean usage in `01_huggingface_pipelines/`.

---

### ⚪ Day 4 (Friday, 2026-09-18) — Transfer Learning & Hugging Face `Trainer`
* **Status:** ⚪ **Upcoming**
* **Target Time:** 5 – 6 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2 (Theory):**
    - What is Transfer Learning? Pre-training on trillions of tokens vs Domain Fine-Tuning.
    - Feature Extraction (Freezing backbone) vs Full Fine-Tuning.
    - Deep dive into `TrainingArguments`:
      - `learning_rate`, `lr_scheduler_type` (linear vs cosine), `warmup_ratio`.
      - `per_device_train_batch_size`, `gradient_accumulation_steps`.
      - `fp16` / `bf16` mixed precision, `evaluation_strategy`, `save_steps`.
  - **Hours 2–4.5 (Hands-on Code):**
    - Setting up the Hugging Face `Trainer` class.
    - `DataCollatorWithPadding` for high-performance dynamic batching.
    - Evaluation metrics integration with `evaluate` library (Accuracy, F1, Loss).
    - Running a training loop and evaluating loss curves.
  - **Hours 4.5–5.5 (Notes & Quiz):**
    - Update `02_transfer_learning/notes.md` with Glossary, Important Things, and Q&A.

---

### ⚪ Day 5 (Saturday, 2026-09-19) — PEFT: LoRA & 4-Bit Quantization
* **Status:** ⚪ **Upcoming**
* **Target Time:** 5 – 6 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2.5 (Core Math & Architecture):**
    - Why Full Fine-Tuning fails on consumer GPUs (Memory calculation: 16-bit weights + gradients + optimizer states = 16 bytes per param!).
    - **LoRA (Low-Rank Adaptation):**
      - Math: $W = W_0 + \Delta W$, where $\Delta W = B \times A$.
      - Hyperparameters: Rank ($r$), Alpha ($\alpha$), Dropout, Target Modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
      - How trainable parameters reduce to $< 1\%$.
    - **Quantization:**
      - FP32 $\to$ FP16 / BF16 $\to$ INT8 $\to$ 4-bit NormalFloat (NF4).
      - `bitsandbytes` library and Double Quantization mechanics.
  - **Hours 2.5–4.5 (Hands-on Code):**
    - `LoraConfig` setup via `peft` library.
    - `BitsAndBytesConfig` (4-bit loading, `bnb_4bit_quant_type="nf4"`).
    - Loading a base model in 4-bit and inspecting parameter memory savings.
  - **Hours 4.5–5.5 (Notes & Quiz):**
    - Update `03_qlora_unsloth_finetuning/notes.md` with Glossary, Math Formulas, and Q&A.

---

### ⚪ Day 6 (Sunday, 2026-09-20) — Supervised Fine-Tuning (SFT) & Unsloth
* **Status:** ⚪ **Upcoming**
* **Target Time:** 5 – 6 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2 (Theory):**
    - What is Supervised Fine-Tuning (SFT)? Instruction $\to$ Response mapping.
    - Prompt Templates: Alpaca format vs ChatML vs Llama-3 format.
    - Response-only Loss masking (Data collator that calculates loss ONLY on the assistant's answer, not the prompt!).
    - Why Unsloth? Manual Triton GPU kernels, 2x-5x faster training, 70% less VRAM on Google Colab T4.
  - **Hours 2–4.5 (Hands-on Colab Setup):**
    - Setting up Google Colab with T4 GPU (16GB VRAM).
    - Loading `unsloth` FastLanguageModel.
    - Applying 4-bit QLoRA with Unsloth.
    - Setting up TRL's `SFTTrainer`.
  - **Hours 4.5–5.5 (Notes & Interview Prep):**
    - Documenting SFT prompt masking & Unsloth architecture.

---

### ⚪ Day 7 (Monday, 2026-09-21) — Capstone Project: Technical Interview QA Assistant
* **Status:** ⚪ **Upcoming**
* **Target Time:** 5 – 6 Hours
* **Capstone Focus:**
  - **Domain:** Data Science & AI Interview Prep Assistant.
  - **Dataset:** [`data_science.csv`](https://github.com/AshishJangra27/datasets/tree/main/Intervew%20Questions) (Question-Answer pairs).
* **Execution Plan:**
  - **Hours 1–1.5:** Data cleaning, splitting, and prompt formatting (System prompt + User question + Model answer).
  - **Hours 1.5–3.5:** 4-bit QLoRA Fine-tuning run on Colab T4 GPU with `SFTTrainer`.
  - **Hours 3.5–4.5:** Qualitative testing on unseen, complex Data Science interview questions (Comparing Base vs Fine-Tuned model).
  - **Hours 4.5–5.5:** Saving LoRA adapters, merging with base model, exporting GGUF format for local Ollama/vLLM inference, and final portfolio documentation!

---

## 📌 Daily Learning Log

| Date | Day | Phase | Topics Covered | Daily Reflection & Outcome |
| :---: | :---: | :---: | :--- | :--- |
| **2026-09-15** | **Day 1** | Phase 1 | HF Ecosystem, Model Cards, Pipelines vs AutoClasses, 5 Tasks, Python 3.13 debugging | Completed Phase 1 hands-on. Built deep intuition on pipeline lifecycle, AutoModel QA, Beam Search, and Zero-Shot NLI. Compiled 3-Part Notes. |
