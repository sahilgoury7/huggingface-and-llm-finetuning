# 📘 04: Capstone Project — Technical Interview QA Assistant (Day 6 & 7)

> **Track:** Gen AI Batch 9 | LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Project Goal:** LLaMA-3 / Mistral ko Data Science & AI Technical Interview Assistant me fine-tune karna, evaluate karna aur local inference ke liye GGUF export karna!  

---

# 📖 Part 1: Glossary (Capstone Project Key Terms)

| Term | Ek Line Me Asli Matlab | Asli Zindagi Ka Example |
| :--- | :--- | :--- |
| **Domain-Specific Assistant** | Ek aam general AI ko kisi specific field (jaise Data Science) ka expert banana. | MBBS doctor ko sirf Data Science aur AI ke interview questions ka specialist banana. |
| **Dataset (`data_science.csv`)** | Technical interview questions aur unke ideal expert answers ka curated collection. | Technical interview ki master answer key. |
| **Alpaca Formatting** | CSV ke `Question` aur `Answer` ko `### Instruction:` aur `### Response:` me convert karna. | Interview question paper ko standard exam format me convert karna. |
| **Evaluation Loss (Validation)** | Model un sawaalon par kaisa perform kar raha hai jo usne training me kabhi nahi dekhe. | Unseen mock interview test jisme pata chale ki model sach me seekha ya sirf ratta mara! |
| **LoRA Merge (`merge_and_unload`)** | Fine-tuned LoRA adapter ko permanently base model ke weights me jod dena. | Butter paper ke saare notes ko original book ke andar permanently print kar dena. |
| **GGUF Format** | Model ko CPU aur normal laptops par super-fast chalane wala compact format (Ollama / llama.cpp standard). | Badi movie file ko ultra-compressed MP4 me convert karna taaki mobile par bhi bina ruke chale. |

---

# 💡 Part 2: Important Things (Project Architecture & Steps)

### 🎯 Project Overview:
* **Target Domain:** Data Science, Machine Learning, Deep Learning, aur GenAI Technical Interviews.
* **Base Model:** `unsloth/llama-3-8b-bnb-4bit` (ya `unsloth/mistral-7b-v0.3-bnb-4bit`).
* **Hardware:** Google Colab Free T4 GPU (16GB VRAM).
* **Speed & Efficiency:** Unsloth Triton Kernels + 4-bit QLoRA.

---

### 🗺️ Step-by-Step Capstone Execution Roadmap:

```text
┌────────────────────────────────────────────────────────────────────────┐
│               CAPSTONE PROJECT 5-STAGE PIPELINE                        │
├────────────────────────────────────────────────────────────────────────┤
│ Stage 1: Google Colab T4 Setup & Unsloth Installation                  │
│ Stage 2: data_science.csv Load & Alpaca Prompt Formatting Pipeline     │
│ Stage 3: FastLanguageModel 4-bit Loading + LoRA Adapter Configuration  │
│ Stage 4: SFTTrainer Training Loop Execution & Loss Curve Monitoring    │
│ Stage 5: Inference Testing, LoRA Merge & GGUF Local Export             │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 🎯 Part 3: Interview Pitch (Resume / Portfolio Project)

### 🎙️ Aapke Resume & Interview Ka Spoken Pitch:
* **Hinglish:**  
  *"Maine open-source LLaMA-3-8B model ko Technical Data Science Interview Assistant ke roop me fine-tune kiya. Google Colab ke single 16GB T4 GPU par training ko feasible banane ke liye maine 4-bit QLoRA (NF4 quantization) aur Unsloth GPU acceleration ka use kiya, jisse training 5x fast ho gayi aur memory consumption 70% kam ho gayi. Model ko Stanford Alpaca format me curated QA dataset par SFTTrainer ke zariye train kiya gaya, jisme response-only loss masking use karke model ko precise technical answers generate karna sikhaya gaya."*

* **English:**  
  *"I fine-tuned an open-source LLaMA-3-8B model into a specialized Technical Interview QA Assistant for Data Science and AI roles. To achieve high parameter efficiency on a single 16GB Colab T4 GPU, I leveraged 4-bit QLoRA (NF4 quantization) and Unsloth Triton GPU kernels, achieving a 5x training speedup and 70% VRAM reduction. The model was trained using TRL's SFTTrainer with Alpaca prompt templates and response-only loss masking, ensuring robust instruction-following and accurate domain-specific completions."*
