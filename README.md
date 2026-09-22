# 🤗 Hugging Face & LLM Fine-Tuning

> **Gen AI Batch 9 | Applied LLMs & Open-Source AI**  
> **Instructor:** Ashish Jangra  
> 
> 📅 **[View Learning Journey & Progress Timeline](./TIMELINE.md)**  
> 🎤 **[View Master Technical Interview Guide (Days 1–7 Q&As)](./MASTER_INTERVIEW_GUIDE.md)**

---

## 🗺️ Official Learning Path & Syllabus

### 📘 Part 1: Hugging Face Basics
* **What is Hugging Face?** (Ecosystem, vision, and open-source AI community)
* **Hugging Face Models Hub Web Interface** (Exploring models, tags, tasks, and libraries)
* **Understanding the Models Card** (Architecture, intended use, limitations, datasets, and license)
* **The Inference API** (Serverless inference & testing models via API)
* **The Pipeline Abstraction** (`pipeline()` API for instant inference across modalities)
* **The AutoTokenizer Family** (`AutoTokenizer`, subword tokenization, vocabularies)
* **Tokenization Strategies** (Padding, truncation, attention masks, return tensors)
* **The AutoModel Family** (`AutoModel`, `AutoModelForSequenceClassification`, `AutoModelForCausalLM`)
* **Hugging Face Datasets** (`load_dataset`, dataset viewer, streaming, and `.map()` preprocessing)
* **🎯 Project: Multiple Mini-Projects using Pipeline Abstraction**
  * *Mini-Project 1:* Sentiment & Review Classification Utility
  * *Mini-Project 2:* Resume / Text Named Entity Extractor (NER)
  * *Mini-Project 3:* Long Article & Notes Summarization Tool
  * *Mini-Project 4:* Zero-Shot Support Ticket Classifier

---

### 🚀 Part 2: Transfer Learning with Hugging Face, and QLoRA
* **What is Transfer Learning?** (Pretraining vs fine-tuning, feature extraction vs full fine-tuning)
* **The TrainingArgs Function** (`TrainingArguments` setup: LR schedules, batch sizes, gradient accumulation, logging)
* **The Trainer API** (`Trainer` loop, evaluation callbacks, and checkpointing)
* **Low Rank Adaptation (LoRA)** (Low-rank decomposition matrices $A \times B$, rank $r$, scaling $\alpha$)
* **Quantization** (FP32 $\to$ FP16 $\to$ INT8 $\to$ 4-bit NormalFloat NF4 via `bitsandbytes`)
* **Supervised Fine-Tuning (SFT)** (Instruction-response prompt formatting, loss computation on completions)
* **🎯 Project: Transfer Learning & QLoRA Capstone Projects**
  * **Capstone 1 — AI Medical Healthcare Assistant (ChatDoctor QA):**
    * **Selected Dataset:** [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) (100k+ doctor-patient consultations)
    * **Execution:** Train LLaMA-3 8B on Google Colab T4 GPU with 4-bit QLoRA and Unsloth acceleration.
    * **Deployment:** Interactive Gradio Web Demo with multi-turn conversational memory (`share=True`).
  * **Capstone 2 — SQLCoder-Lite (Enterprise Text-to-SQL Assistant):**
    * **Selected Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (78.5k+ cross-domain SQL pairs with DDL schemas)
    * **Execution:** Train Mistral-7B on Google Colab T4 GPU with 4-bit QLoRA and Unsloth acceleration (0.58% parameters trained).
    * **Deployment:** Standalone Gradio Web Interface for instant schema-aware SQL query generation.

---

## 📂 Repository Structure

| Folder | Module Focus | Core Project |
| :--- | :--- | :--- |
| [**01_huggingface_pipelines**](./01_huggingface_pipelines/day1_hf_basics_and_pipelines_notes.md) | Part 1: HF Hub, Pipelines, AutoClasses & Datasets | Multiple Mini-Projects using Pipelines |
| [**02_transfer_learning**](./02_transfer_learning/notes.md) | Part 2A: Transfer Learning, `TrainingArguments` & `Trainer` | Transfer Learning Project Setup |
| [**03_qlora_unsloth_finetuning**](./03_qlora_unsloth_finetuning/notes.md) | Part 2B: LoRA, 4-bit Quantization, SFT & Unsloth | Foundation for QLoRA & SFT |
| [**04_medical_chatdoctor**](./04_medical_chatdoctor/day6_capstone_qa_assistant_notes.md) | Capstone 1: AI Medical Healthcare Assistant | Fine-Tuned LLaMA-3 8B with Gradio Live Demo |
| [**05_sqlcoder_lite**](./05_sqlcoder_lite/README.md) | Capstone 2: Enterprise Text-to-SQL Assistant | Fine-Tuned Mistral-7B with Gradio Web Interface |

---

## 🛠️ Tech Stack & Tooling

* **Core Libraries:** `transformers`, `datasets`, `tokenizers`, `accelerate`, `evaluate`, `huggingface_hub`
* **Fine-Tuning & Quantization:** `peft`, `bitsandbytes`, `unsloth`, `torch`
* **Hardware:** Google Colab T4 GPU (16GB VRAM) / Local NVIDIA CUDA
