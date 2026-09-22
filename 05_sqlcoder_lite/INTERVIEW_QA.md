# 🎤 SQLCoder-Lite: Technical Interview Questions & Answers

> **Capstone Project 2: Applied LLM Fine-Tuning & Open-Source AI**  
> **Model:** `unsloth/mistral-7b-v0.3-bnb-4bit` | **PEFT:** 4-bit QLoRA ($r=16, \alpha=16$)  
> **Dataset:** `b-mc2/sql-create-context` (78.5k samples)

---

## 🚦 Interview Probability Matrix (Revision Priority)

| Question | Topic | Probability | What It Tests |
| :--- | :--- | :---: | :--- |
| **Q1: Fine-Tuning vs AI Agents / GPT-4** | System Architecture & Business Value | 🔴 **HIGH** | Why fine-tune instead of using APIs/LangChain (Privacy & Cost). |
| **Q2: Temperature=0.1 vs 0.7** | Decoding & Inference Strategies | 🔴 **HIGH** | Greedy vs Stochastic generation in deterministic code tasks. |
| **Q7: Production Evaluation Metrics** | Evaluation & LLMOps | 🔴 **HIGH** | Execution Accuracy (EX) vs Exact Match (EM) vs BLEU/ROUGE. |
| **Q3: Loss Dynamics (0.35 - 0.45)** | Training Dynamics & Analysis | 🟡 **MEDIUM** | Did you actually observe the loss curve or just run code blindly? |
| **Q4: Target Modules (All 7 Layers)** | QLoRA & PEFT Architecture | 🟡 **MEDIUM** | Understanding MLP (domain knowledge) vs Attention (routing). |
| **Q6: Alpaca Prompt & EOS Tokens** | SFT Formatting & Tokenization | 🟡 **MEDIUM** | How to prevent infinite generation loops in production. |
| **Q8: Enterprise Agent Integration** | End-to-End System Design | 🟡 **MEDIUM** | How fine-tuned weights fit into ReAct loops & SQL guardrails. |
| **Q5: Parameter Math (0.58%)** | Mathematical Calculations | 🟢 **LOW** | Formula for Low-Rank decomposition ($2 \times r \times d$). |

---

## 🎯 30-Second Elevator Pitch
> *"I fine-tuned Mistral-7B into an enterprise-grade, privacy-preserving Text-to-SQL engine called **SQLCoder-Lite** using the `b-mc2/sql-create-context` dataset (78.5k schema-context pairs). To make training feasible on a single free Tesla T4 GPU (16GB VRAM), I utilized 4-bit NormalFloat (NF4) quantization and QLoRA via Unsloth, reducing the trainable parameter footprint to just **0.58%** (41.9M parameters across all 7 linear layers). Using TRL's SFTTrainer with Alpaca prompt templates, the model converged smoothly to a loss of 0.41 in under 10 minutes. For production serving, I built a standalone Gradio interface utilizing deterministic decoding (`temperature=0.1`) that empowers non-technical users to query relational databases safely with zero cloud data leakage."*

---

## 🏗️ Architectural & Design Decisions

### 🔴 Q1: Log SQL chatbots banate hain ya LangChain/LlamaIndex se AI Agents banate hain. Aapne Mistral-7B ko fine-tune kyun kiya?
> **Probability:** 🔴 **HIGH (Must-Prepare — 90%+ Interview Chance)**  
> **Core Concept:** Fine-Tuning (The Brain) vs AI Agents (The Hands) vs Third-Party APIs

**Answer:**
Fine-Tuning aur AI Agents competitors nahi hain; yeh ek hierarchy ke building blocks hain:
1. **The Brain vs The Hands:** Fine-Tuning model ke neural weights ko SQL syntax, DDL schemas, aur table joins sikhata hai (**The Brain**). AI Agent us model ko database execution, error retry loops, aur memory deta hai (**The Hands**).
2. **Enterprise Data Privacy:** Banking, Healthcare, aur FinTech enterprises proprietary database schemas aur internal column names third-party APIs (OpenAI/Anthropic) ko nahi bhej sakte due to strict compliance (GDPR, HIPAA, RBI). Self-hosted fine-tuned 7B model company ke private VPC mein 100% offline chalta hai.
3. **Deterministic Output & Zero Token Waste:** Generic LLMs (jaise GPT-4) chatty hote hain aur context mein 5-page prompt mangte hain. Fine-tuned model 1-line schema context se bina kisi hallucination ke directly pure executable SQL return karta hai.
4. **Unit Economics & Latency:** Enterprise scale par har query par thousands of tokens third-party API ko bhejna cost-prohibitive hota hai. Quantized 7B model local low-cost GPUs par zero recurring per-token cost ke saath fast inference deta hai.

---

### 🔴 Q2: Why did you use `temperature=0.1` for Text-to-SQL instead of `0.7`?
> **Probability:** 🔴 **HIGH (Must-Prepare — 85%+ Interview Chance)**  
> **Core Concept:** Decoding Strategy (Greedy vs Stochastic) for Strict Syntax

**Answer:**
SQL ek deterministic programming language hai jisme exact syntax aur valid column names mandatory hote hain. 
* **High Temperature (`0.7`):** Sampling distribution ko flatten karta hai, jisse randomness badhti hai. Natural language writing/storytelling ke liye yeh accha hai, par SQL mein yeh invalid column names hallucinate karega ya syntax error dega.
* **Low Temperature (`0.1` / Greedy Decoding):** Token probability distribution ko sharpen karta hai, jisse model har step par highest-probability keyword aur schema column names ko verbatim choose karta hai.

---

### 🟡 Q3: Why did your training loss stay in the 0.35 - 0.45 range from start to finish instead of dropping from 3.0 to 1.8 like in ChatDoctor?
> **Probability:** 🟡 **MEDIUM (Important — 65% Interview Chance)**  
> **Core Concept:** Cross-Entropy Entropy in Code/SQL vs Open-Ended Natural Language

**Answer:**
Cross-entropy loss prediction entropy par depend karta hai:
* **Conversational QA (ChatDoctor):** Doctor-patient conversations open-ended hoti hain (high entropy). Model ko natural language sentences predict karne hote hain, isliye loss ~3.0 se shuru hoke ~1.8 tak drop hota hai.
* **Code / SQL Generation (SQLCoder-Lite):** Training sample mein DDL schema (`CREATE TABLE ...`), column names, aur SQL keywords pehle se context mein given hote hain. Model ko bas structure assemble karke valid query banani hoti hai, isliye starting loss naturally bohot low (~0.38) hota hai. **0.41 loss SQL task ke liye Golden Zone hai** — high accuracy without overfitting.

---

## ⚡ PEFT & QLoRA Technical Deep Dive

### 🟡 Q4: Why train all 7 linear projection layers (`q, k, v, o, gate, up, down`) instead of only Attention matrices (`q, v`)?
> **Probability:** 🟡 **MEDIUM (Important Technical Round — 60% Interview Chance)**  
> **Core Concept:** MLP Domain Knowledge vs Attention Routing

**Answer:**
Initial LoRA papers ne sirf Query ($W_q$) aur Value ($W_v$) matrices ko adapt kiya tha. Lekin empirical research (QLoRA paper by Dettmers et al.) ne prove kiya ki:
1. MLP / Feed-Forward layers (`gate_proj`, `up_proj`, `down_proj`) model ke **factual domain knowledge aur syntax rules** ko store karte hain.
2. Attention layers (`q, k, v, o`) routing aur context alignment handle karte hain.
3. All 7 layers par LoRA adapters lagane se model domain-specific syntax (SQL grammar) ko much deeper adapt kar pata hai, jabki trainable parameters tab bhi sirf **0.58%** hi rehte hain.

---

### 🟢 Q5: How is the trainable parameter footprint calculated (0.58%)?
> **Probability:** 🟢 **LOW / ADVANCED (Math / Theory Round — 35% Interview Chance)**  
> **Core Concept:** Mathematical Low-Rank Decomposition ($A \times B$)

**Answer:**
* **Total Parameters in Mistral-7B:** ~7.24 Billion parameters.
* **Base Model Quantization:** Base weights ko 4-bit NormalFloat (NF4) mein freeze kiya gaya.
* **LoRA Adapters:** Rank $r=16$, Alpha $\alpha=16$.
* Har target module $W_0 \in \mathbb{R}^{d \times k}$ ke parallel do low-rank matrices add kiye gaye: $A \in \mathbb{R}^{r \times k}$ aur $B \in \mathbb{R}^{d \times r}$.
* Trainable parameters = $2 \times r \times d$ per layer.
* Total trainable parameters = **41,943,040 (~41.9M)**, jo ki total model size ka exact **0.58%** hai.

---

### 🟡 Q6: What is the purpose of Alpaca prompt formatting and EOS stop tokens?
> **Probability:** 🟡 **MEDIUM (Practical SFT Implementation — 65% Interview Chance)**  
> **Core Concept:** Prompt Templates & Preventing Infinite Generation Loops

**Answer:**
* **Alpaca Format:** Model ko structured context provide karta hai:
  ```
  ### Instruction:
  Convert the question to SQL using this database schema.
  ### Input:
  [Schema DDL] + [Question]
  ### Response:
  [SQL Query]
  ```
* **EOS (End-of-Sequence) Token:** Training ke time response ke end mein `tokenizer.eos_token` lagana mandatory hota hai. Agar EOS token na ho, toh model inference ke time stop nahi hoga aur infinite loops ya repetitive garbage output generate karega.

---

## 🚀 Production, Evaluation & Systems Integration

### 🔴 Q7: How do you evaluate a Text-to-SQL model in production?
> **Probability:** 🔴 **HIGH (Must-Prepare / Standard LLMOps — 80%+ Interview Chance)**  
> **Core Concept:** Execution Accuracy (EX) vs Exact Match (EM) vs Semantic Metrics

**Answer:**
Text-to-SQL ko sirf BLEU ya ROUGE score (string matching) se evaluate nahi kiya ja sakta, kyunki do completely alag SQL queries same result return kar sakti hain (e.g. `JOIN` vs subquery).
1. **Execution Accuracy (EX):** Generated SQL aur Ground Truth SQL dono ko actual test database engine par execute karte hain. Agar dono ka returned dataframe/result table identical hai, toh query correct maani jaati hai.
2. **Valid SQL Syntax Rate (VSR):** Kitne percent queries without database syntax error compile aur execute ho rahi hain.
3. **Exact Match (EM):** Generated query ground truth query se character-by-character match karti hai ya nahi (stricter metric).

---

### 🟡 Q8: How does SQLCoder-Lite integrate into an Enterprise Agentic Architecture?
> **Probability:** 🟡 **MEDIUM (System Design & Applied AI — 70% Interview Chance)**  
> **Core Concept:** ReAct Agent Loop, Database Tools & Guardrails

**Answer:**
In a production system:
1. **Orchestrator (Agent Loop):** User question aate hi orchestrator metadata schema fetch karta hai.
2. **Inference Engine (SQLCoder-Lite):** Schema + Question lekar fast, deterministic SQL generate karta hai.
3. **Execution & Guardrails Tool:** Query validation layer (SELECT-only check, prevention of DROP/DELETE) ke baad query safe database par execute hoti hai.
4. **Self-Correction Feedback:** Agar database syntax error ya missing column throw kare, toh error message wapas prompt context mein append karke SQLCoder-Lite ko retry ke liye pass kiya jata hai.
