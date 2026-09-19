# 02: Tokenization Mechanics, Datasets & Transfer Learning

> **Track:** Gen AI Batch 9 — Hugging Face & LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Module Status:** 🟢 **Completed & Verified**  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fy8GFShovpkpSRkaKxVAinC7HN6wZfFv?usp=sharing)

---

## 📌 Executive Overview

This module covers the foundational data pipeline and training mechanics required to fine-tune Large Language Models (LLMs) in production. Instead of treating Hugging Face as a black box, this repository deconstructs:
1. **How AI models interpret text** via sub-word tokenization algorithms (BPE, WordPiece, SentencePiece).
2. **GPU Memory Optimization** via Dynamic Padding (`DataCollatorWithPadding`) vs Static Padding.
3. **Zero-RAM Crash Dataset Streaming** using Apache Arrow (`load_dataset`).
4. **End-to-End Training Execution** using Hugging Face's `Trainer` and `TrainingArguments` on a 100k+ healthcare dataset.

---

## 📂 Repository Contents

| File | Description | Quick Link |
| :--- | :--- | :---: |
| 📓 **[`Day_02_03_Tokenization_Datasets_and_HF_Trainer.ipynb`](./Day_02_03_Tokenization_Datasets_and_HF_Trainer.ipynb)** | Complete hands-on Jupyter Notebook containing all 8 executed pipeline steps. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fy8GFShovpkpSRkaKxVAinC7HN6wZfFv?usp=sharing) |
| 📝 **[`notes.md`](./notes.md)** | Comprehensive 3-Part study guide (Glossary, Important Formulas, VRAM Math, and 10 Top Interview Q&As). | [View Notes](./notes.md) |

---

## 🚀 Key Technical Highlights & Benchmarks

### 1. Dynamic Padding vs Static Padding (90% Memory Saving)
In static padding (`max_length=128`), every short sample is padded with useless `<pad>` tokens, resulting in massive GPU compute waste. Using Hugging Face's `DataCollatorWithPadding`, sequences are padded dynamically only to the longest sequence within each batch:

```text
Static Batch Tensor Size  : [2, 128] ──▶ 256 matrix elements
Dynamic Batch Tensor Size : [2, 12]  ──▶  24 matrix elements (90.6% Reduction!)
```

### 2. High-Throughput Dataset Streaming (Apache Arrow)
* **Dataset:** [`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) (112,165 real medical Q&A pairs).
* **Architecture:** Uses Apache Arrow zero-copy memory mapping (`mmap`). Data resides on disk and streams into memory on-demand per batch, allowing 50GB+ datasets to run smoothly on standard 16GB RAM machines without OOM crashes.
* **Preprocessing:** Multi-threaded tokenization with `.map(batched=True)` and automatic raw string stripping (`remove_columns`) for PyTorch tensor compatibility.

### 3. End-to-End `Trainer` Execution
* **Base Model:** `distilbert/distilgpt2` (Causal Language Model).
* **Collator:** `DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)` for autoregressive next-token prediction.
* **Effective Batch Size:** Simulating larger batches on consumer GPUs via gradient accumulation ($2 \text{ batch} \times 2 \text{ accumulation} = 4 \text{ effective batch}$).
* **Loss Trend:** Initial loss dropped from **`4.538`** down to **`4.292`** within 20 steps, and model weights were serialized to disk shards (`Writing model shards: 100%`).

---

## 🛠️ The 5-Step Training Architecture

```text
[ 1. Ingestion ]      load_dataset() via Apache Arrow (Disk mmap, 0 RAM crash)
       │
[ 2. Tokenize ]       AutoTokenizer + .map(batched=True, remove_columns=[...])
       │
[ 3. Collate ]        DataCollatorForLanguageModeling(mlm=False) (Dynamic padding)
       │
[ 4. Configure ]      TrainingArguments(learning_rate=2e-5, gradient_accumulation=2)
       │
[ 5. Execute ]        Trainer.train() ──▶ Track Cross-Entropy Loss ──▶ Save Shards
```

---

## 🏃 How to Run the Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sahilgoury7/huggingface-and-llm-finetuning.git
   cd huggingface-and-llm-finetuning/02_transfer_learning
   ```

2. **Open in Google Colab / Jupyter:**
   - **Direct 1-Click Cloud Execution:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fy8GFShovpkpSRkaKxVAinC7HN6wZfFv?usp=sharing)
   - Or open locally: [`Day_02_03_Tokenization_Datasets_and_HF_Trainer.ipynb`](./Day_02_03_Tokenization_Datasets_and_HF_Trainer.ipynb).
   - Set runtime to **GPU (T4)**.
   - Run all cells sequentially to reproduce the tokenization inspection, dynamic batch collation, and model training run!

---

## 🎯 Core Interview Takeaways
* **Why `learning_rate = 2e-5`?** High learning rates in pre-trained models trigger **Catastrophic Forgetting**; gentle learning rates protect pre-trained foundational knowledge.
* **Why `mlm=False`?** Modern generative LLMs (Llama-3, DeepSeek, GPT) are Causal Language Models that predict the next sequential token, unlike Masked Language Models (BERT) that fill in masked blanks.
* **Why 4-bit Quantization?** An 8B parameter model requires 16GB VRAM just to load weights in FP16, leaving zero memory for optimizer states and gradients on a 16GB GPU. 4-bit NormalFloat (NF4) shrinks the model footprint to ~4GB, enabling fine-tuning on consumer hardware.
