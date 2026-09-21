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
| **Day 2** | **2026-09-16** | **Phase 2: Tokenization Mechanics & Datasets** | BPE/WordPiece, Dynamic Padding, `datasets` `.map()`, VRAM Math | 🟢 **Completed** |
| **Day 3** | **2026-09-17** | **Phase 3: Transfer Learning & HF Trainer API** | `TrainingArguments`, Loss, Metrics, Dynamic Collators, Evaluation | 🟢 **Completed** |
| **Day 4** | **2026-09-18** | **Phase 4: PEFT, LoRA Math & 4-Bit Quantization** | Matrix decomposition ($B \times A$), Rank ($r$), Alpha, NF4 bitsandbytes | 🟢 **Completed** |
| **Day 5** | **2026-09-19** | **Phase 5: SFT & Unsloth GPU Acceleration** | Prompt formatting, Completion-only loss, Unsloth T4 setup | 🟢 **Completed** |
| **Day 6** | **2026-09-20** | **Phase 6: Capstone Part 1: AI Medical Assistant (ChatDoctor QA)** | LLaMA-3 8B 4-bit QLoRA, Loss 2.87 $\to$ 1.89, Gradio Live Demo | 🟢 **Completed** |
| **Day 7** | **2026-09-21** | **Phase 7: Capstone Part 2: Eval, LoRA Merge & GGUF** | RAG/LLM Eval, LoRA merge, GGUF export for local Ollama/vLLM | ⚪ Pending |

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
  - [x] Structured 3-Part Notes compiled in `01_huggingface_pipelines/day1_hf_basics_and_pipelines_notes.md`.

---

### 🟢 Day 2 (Wednesday, 2026-09-16) — Tokenization & Datasets Deep Dive
* **Status:** 🟢 **COMPLETED (100%)**
* **Time Spent:** ~5.5 Hours
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

### 🟢 Day 3 (Thursday, 2026-09-17) — Phase 3: Transfer Learning & Hugging Face `Trainer` API
* **Status:** 🟢 **COMPLETED (100%)**
* **Time Spent:** ~5.5 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2 (Core Concepts & Mechanics):**
    - What is Transfer Learning? Pre-training (trillions of general tokens) vs Domain Fine-Tuning.
    - Feature Extraction (Freezing backbone) vs Full Fine-Tuning.
    - The `TrainingArguments` masterclass:
      - `learning_rate`, `lr_scheduler_type` (linear vs cosine), `warmup_steps`.
      - `per_device_train_batch_size`, `gradient_accumulation_steps` (Virtual large batches!).
      - `fp16` / `bf16` mixed precision, `logging_steps`, `save_steps`, `save_total_limit`.
  - **Hours 2–4.5 (Hands-on Training Loop):**
    - Setting up the Hugging Face `Trainer` class.
    - Integrating `DataCollatorForLanguageModeling(mlm=False)` / `DataCollatorWithPadding`.
    - Monitoring training loss, validation loss curves, and perplexity.
  - **Hours 4.5–5.5 (Notes & Interview Prep):**
    - 3-Part Notes update & Top Interview Q&As on `Trainer` & `TrainingArguments`.

---

### 🟢 Day 4 (Friday, 2026-09-18) — Phase 4: PEFT, LoRA Math & 4-Bit Quantization
* **Status:** 🟢 **COMPLETED (100%)**
* **Time Spent:** ~5 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2.5 (Core Math & Architecture):**
    - Why Full Fine-Tuning fails on consumer GPUs (Memory calculation: 16-18 bytes per param!).
    - **LoRA (Low-Rank Adaptation):**
      - Math: $W = W_0 + \Delta W$, where $\Delta W = B \times A$.
      - Hyperparameters: Rank ($r$), Alpha ($\alpha$), Dropout, Target Modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
      - How trainable parameters reduce to $< 1\%$.
    - **Quantization:**
      - FP32 $\to$ FP16 / BF16 $\to$ INT8 $\to$ 4-bit NormalFloat (NF4).
      - `bitsandbytes` library and Double Quantization mechanics.
  - **Hours 2.5–4.5 (Hands-on Code & Notes):**
    - `LoraConfig` setup via `peft` library.
    - `BitsAndBytesConfig` (4-bit loading, `bnb_4bit_quant_type="nf4"`).
    - Parameter memory savings & zero-latency `merge_and_unload`.
  - **Hours 4.5–5.5 (Notes & Quiz):**
    - Update `03_qlora_unsloth_finetuning/day4_peft_lora_quantization_notes.md` with Glossary, Easy Stories, and Q&A.

---

### 🟢 Day 5 (Saturday, 2026-09-19) — Phase 5: Supervised Fine-Tuning (SFT) & Unsloth
* **Status:** 🟢 **COMPLETED (100%)**
* **Time Spent:** ~5.5 Hours
* **Syllabus & Schedule:**
  - **Hours 1–2 (Theory & Intuition):**
    - What is Supervised Fine-Tuning (SFT)? Pre-training (sentence completer) vs SFT (helpful assistant).
    - Prompt Templates: Alpaca format (`### Instruction:` and `### Response:`).
    - Response-only Loss masking: The `-100` ignore index secret in PyTorch.
    - Why Unsloth? OpenAI Triton GPU kernels, 5x faster training, 70% VRAM reduction.
  - **Hours 2–4.5 (Architecture & Code):**
    - `FastLanguageModel` 4-bit loading and LoRA injection.
    - `SFTTrainer` vs normal `Trainer`.
  - **Hours 4.5–5.5 (Notes & Spoken Pitches):**
    - Created and compiled `03_qlora_unsloth_finetuning/day5_sft_unsloth_notes.md` with Glossary, Kahanis, and 5 Interview Q&As.

---

---

### 🟢 Day 6 (Sunday, 2026-09-20) — Phase 6: Capstone Part 1: AI Medical Assistant (ChatDoctor QA)
* **Status:** 🟢 **COMPLETED (100%)**
* **Time Spent:** ~5.5 Hours
* **Capstone Focus:**
  - **Domain:** AI Medical Healthcare Assistant (ChatDoctor QA).
  - **Dataset:** [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) (100k+ doctor-patient consultations).
* **Accomplishments & Deliverables:**
  - [x] Data inspection, word count EDA (Patient query: ~80 words, Doctor advice: ~102 words).
  - [x] Filtered dataset to < 350 words (1,924 clean samples, strictly under 512 tokens).
  - [x] LLaMA-3 8B loaded in 4-bit NF4 with Unsloth GPU acceleration.
  - [x] LoRA adapters configured ($r=16, \alpha=16$, all linear layers, only **0.52%** trainable params).
  - [x] Stanford Alpaca prompt formatting applied with `tokenizer.eos_token` (`<|end_of_text|>`).
  - [x] SFT training executed with `SFTTrainer` (60 steps, Loss dropped from `2.87` to `1.89`).
  - [x] LoRA adapters saved locally (`medical_llama3_lora`).
  - [x] Deployed live Gradio web demo (`share=True`) featuring Multi-Turn Conversational Memory & Anti-Repetition tuning (`temperature=0.7`, `repetition_penalty=1.15`).

---

### ⚪ Day 7 (Monday, 2026-09-21) — Phase 7: Capstone Part 2: Evaluation, LoRA Merge & GGUF
* **Status:** ⚪ **Upcoming**
* **Target Time:** 5 – 6 Hours
* **Execution Plan:**
  - **Hours 1–2.5:** Rigorous Model Evaluation (Base vs Fine-Tuned qualitative comparison on unseen medical queries, clinical metrics).
  - **Hours 2.5–4:** Saving LoRA adapters & Merging 16-bit weights with the base model.
  - **Hours 4–5.5:** Exporting to **GGUF format** / Hugging Face Hub push + Final portfolio documentation and LinkedIn post write-up!

---

## 📌 Daily Learning Log

| Date | Day | Phase | Topics Covered | Daily Reflection & Outcome |
| :---: | :---: | :---: | :--- | :--- |
| **2026-09-15** | **Day 1** | Phase 1 | HF Ecosystem, Model Cards, Pipelines vs AutoClasses, 5 Tasks, Python 3.13 debugging | Completed Phase 1 hands-on. Built deep intuition on pipeline lifecycle, AutoModel QA, Beam Search, and Zero-Shot NLI. Compiled 3-Part Notes. |
| **2026-09-16** | **Day 2** | Phase 2 & 3 | Sub-word Tokenization (BPE/WordPiece/SentencePiece), Dynamic Padding, Attention Mask, `load_dataset()`, VRAM Math | Mastered tokenization mechanics, the restaurant table padding analogy, Apache Arrow mmap, Ashish Sir's cell 10 & 12 VRAM math, and QLoRA 4-bit necessity. Compiled 3-Part Notes. |
| **2026-09-17** | **Day 3** | Phase 3 | Transfer Learning, Catastrophic Forgetting, `TrainingArguments`, `DataCollatorForLanguageModeling(mlm=False)`, Trainer Loop | Successfully executed live training loop in Colab on healthcare dataset. Observed training loss drop from 4.53 to 4.29, saved model shards, and mastered NaN debugging & effective batch sizes. |
| **2026-09-18** | **Day 4** | Phase 4 | PEFT, LoRA Matrix Decomposition ($B \times A$), Rank ($r$), Alpha, NF4 Quantization | Mastered low-rank decomposition math, double quantization, and bitsandbytes 4-bit VRAM savings. Trainable parameters reduced from 8B to 41.9M (0.52%). |
| **2026-09-19** | **Day 5** | Phase 5 | Supervised Fine-Tuning (SFT), Alpaca Prompt Formatting, Response-Only Loss Masking (`-100`), Unsloth Triton Kernels | Learned SFT mechanics, why pre-trained models need instruction tuning, and how Unsloth accelerates training by 5x while slashing VRAM by 70%. |
| **2026-09-20** | **Day 6** | Phase 6 | Capstone: AI Medical Assistant (ChatDoctor QA), LLaMA-3 8B 4-bit QLoRA, SFTTrainer, Gradio Live Web Demo | Fine-tuned LLaMA-3 8B on Colab Tesla T4 GPU. Loss reduced from 2.87 to 1.89. Deployed live Gradio demo with multi-turn conversational memory and anti-repetition penalty. |
| **2026-09-21** | **Day 7** | Phase 7 | Text-to-SQL Architecture, Mistral-7B Deep-Dive, Sequence Length EDA, LoRA Hyperparameters Breakdown | Explored Text-to-SQL (`b-mc2/sql-create-context`) with Mistral-7B. Analyzed ~32-word sequence lengths, deep-dived into LoRA hyperparameters ($r=16, \alpha=16$, 7 target modules), and saved the complete ChatDoctor notebook to repo. |

