# 🎤 SQLCoder-Lite: Technical Interview Questions & Answers

> **Capstone Project 2: Applied LLM Fine-Tuning & Open-Source AI**  
> **Model:** `unsloth/mistral-7b-v0.3-bnb-4bit` | **PEFT:** 4-bit QLoRA ($r=16, \alpha=16$)  
> **Dataset:** `b-mc2/sql-create-context` (78.5k samples)

---

## 🚦 Quick Revision Matrix (Interview Probability)

| Question | Topic | Probability | Main Focus |
| :--- | :--- | :---: | :--- |
| **Q1: Fine-Tuning vs AI Agents / GPT-4** | Architecture & Strategy | 🔴 **HIGH (90%+)** | Data Privacy & Zero API Cost vs Cloud APIs |
| **Q2: Temperature=0.1 vs 0.7** | Inference & Decoding | 🔴 **HIGH (85%+)** | Greedy decoding prevents SQL syntax hallucinations |
| **Q7: Evaluation: Execution Accuracy vs String Match** | LLMOps & Metrics | 🔴 **HIGH (80%+)** | Why string match fails in code & how EX works |
| **Q3: Loss Dynamics (0.35 - 0.45)** | Training Analysis | 🟡 **MEDIUM (65%)** | Why SQL loss starts low (Schema given in prompt) |
| **Q4: Target Modules (All 7 Linear Layers)** | QLoRA Architecture | 🟡 **MEDIUM (60%)** | MLP layers hold domain syntax & knowledge |
| **Q6: Alpaca Format & EOS Stop Token** | SFT Data Preparation | 🟡 **MEDIUM (65%)** | Stopping infinite generation loops |
| **Q8: Enterprise Agent Integration** | System Design | 🟡 **MEDIUM (70%)** | Model as the SQL engine inside a ReAct agent loop |
| **Q5: Trainable Parameter Math (0.58%)** | Mathematical Concept | 🟢 **LOW (35%)** | Matrix formula $A \times B$ footprint breakdown |

---

## 🎯 30-Second Elevator Pitch (Apna Intro Aise Dein)

💡 **Aasaan Samajh:** Aapne Mistral-7B model liya, usko 4-bit QLoRA se free Google Colab par 10 minute mein 78,000 SQL queries par train kiya, aur ek live web app banaya jo English question ko exact SQL mein badalta hai bina kisi data privacy risk ke.

🎙️ **Interview Mein Aise Bolein (English):**
> *"In this capstone, I built **SQLCoder-Lite**, an on-premise Text-to-SQL engine fine-tuned from Mistral-7B using the `b-mc2/sql-create-context` dataset (78.5k schema-context pairs). To make training feasible on a free 16GB T4 GPU, I used 4-bit NormalFloat quantization and QLoRA via Unsloth, training only **0.58% of parameters** across all 7 linear layers. The model converged to a steady loss of 0.41 in under 10 minutes. For production, I deployed a Gradio web interface using deterministic greedy decoding (`temperature=0.1`), allowing non-technical users to query internal databases with zero cloud data leakage."*

---

## 🏗️ Section 1: Architecture & Design Decisions

### 🔴 Q1: Log SQL chatbots banate hain ya LangChain se AI Agents banate hain. Aapne Mistral-7B ko fine-tune kyun kiya?
> **Probability:** 🔴 **HIGH (90%+ chance — Har interviewer ka pehla sawal)**

💡 **Aasaan Bhasha Mein Samajhein:**
* Fine-tuning **"Brain (Dimaag)"** hai — yeh model ko SQL ki grammar aur table joins sikhata hai. AI Agent uske **"Hands (Haath-Pair)"** hain jo database mein query run karte hain.
* Banks aur Hospitals apna internal database schema OpenAI ya ChatGPT ke cloud API par **nahi bhej sakte** (Data Privacy / Compliance issue).
* 7B model private server ya local machine par chalta hai, isliye **zero recurring API cost** aati hai aur data safe rehta hai.
* Generic models chatty hote hain aur column hallucinate karte hain; fine-tuned model 1 line ke prompt se direct accurate SQL deta hai.

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"Fine-tuning and AI Agents are not competitors; they complement each other. Fine-tuning builds the specialized domain brain for SQL grammar, while an Agent provides tools to execute queries and handle retry loops."*
> 2. *"In sectors like Banking and Healthcare, proprietary database schemas cannot be sent to third-party APIs like OpenAI due to compliance like GDPR and HIPAA. A self-hosted 7B model guarantees 100% on-premise data privacy."*
> 3. *"It also eliminates recurring per-token API costs and produces clean, deterministic SQL without requiring extensive prompt engineering."*

---

### 🔴 Q2: Why did you use `temperature=0.1` for Text-to-SQL instead of `0.7`?
> **Probability:** 🔴 **HIGH (85%+ chance — Practical LLM Decoding Concept)**

💡 **Aasaan Bhasha Mein Samajhein:**
* Temperature **"Creativity ka knob"** hai. Agar story ya email likhna ho toh creativity (`0.7`) acchi lagti hai.
* Lekin SQL ek programming language hai. Agar SQL mein model "creative" banega, toh woh aisi table ya column imagine (hallucinate) kar lega jo database mein exist hi nahi karti!
* `0.1` temperature (Greedy Decoding) ka matlab hai: *"Model har step par wahi word choose karega jiski probability sabse high ho aur jo schema mein sach mein maujood ho."*

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"SQL is a deterministic programming language with strict syntax. A high temperature introduces randomness, which leads to hallucinated column names and syntax errors."*
> 2. *"A near-zero temperature like 0.1 enforces greedy decoding. It ensures the model picks the highest-probability tokens and matches schema column names verbatim."*

---

### 🟡 Q3: Why did your training loss stay in the 0.35 - 0.45 range instead of starting at 3.0 like ChatDoctor?
> **Probability:** 🟡 **MEDIUM (65% chance — Training Analysis Concept)**

💡 **Aasaan Bhasha Mein Samajhein:**
* ChatDoctor (Medical QA) mein doctor ka answer open-ended hota hai, hazaron words possible hain, isliye shuru mein model confused hota hai aur loss 3.0 se shuru hota hai.
* Lekin SQL dataset mein prompt ke andar hi table ka DDL schema (`CREATE TABLE ...`) aur columns pehle se diye hote hain!
* Model ko naya text invent nahi karna, bas di gayi tables ko SQL syntax mein fit karna hai. Isliye starting loss naturally low (~0.38) hota hai. **0.41 loss SQL ke liye perfect "Golden Zone" hai** (na underfitting, na overfitting).

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"In open-ended conversational tasks like ChatDoctor, vocabulary entropy is high, so loss begins around 3.0. In Text-to-SQL, the task is heavily constrained because the schema DDL and column names are already provided inside the prompt."*
> 2. *"Because the model is simply assembling provided schema tokens into valid SQL structure, the cross-entropy loss is naturally much lower. A stable loss around 0.41 represents high precision and confident generation."*

---

## ⚡ Section 2: PEFT & QLoRA Technical Details

### 🟡 Q4: Why train all 7 linear projection layers (`q, k, v, o, gate, up, down`) instead of only Attention matrices?
> **Probability:** 🟡 **MEDIUM (60% chance — QLoRA Deep Dive)**

💡 **Aasaan Bhasha Mein Samajhein:**
* Shuruati LoRA papers mein log sirf Attention layers (`q, v`) train karte the.
* Lekin research (QLoRA paper) ne prove kiya ki model ki **factual knowledge aur syntax rules** MLP / Feed-Forward layers (`gate, up, down`) mein hoti hain.
* Jab hum all 7 layers par adapter lagate hain, tab bhi total parameters sirf **0.58%** hi train hote hain, lekin SQL grammar seekhne ki accuracy bohot badh jaati hai.

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"Attention layers handle routing and context, but the MLP layers—namely gate, up, and down projections—store factual domain knowledge and syntax rules."*
> 2. *"By targeting all 7 linear layers instead of just Attention, the model learns the SQL domain much more effectively, while still keeping trainable parameters at just 0.58%."*

---

### 🟢 Q5: How is the trainable parameter footprint calculated (0.58%)?
> **Probability:** 🟢 **LOW / THEORY (35% chance — Math Rounds)**

💡 **Aasaan Bhasha Mein Samajhein:**
* Mistral-7B mein total **7.24 Billion** parameters hain.
* Humne base model ke saare weights ko 4-bit mein freeze (lock) kar diya.
* Sirf rank $r=16$ ke chhote LoRA matrices train kiye.
* Total train hone wale weights sirf **41.9 Million (~4.19 Crore)** the.
* Ratio: $\frac{41.9 \text{ Million}}{7.24 \text{ Billion}} \approx \mathbf{0.58\%}$. Yani 99.4% model memory freeze thi!

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"Mistral-7B has ~7.24 billion base parameters, all frozen in 4-bit NormalFloat format."*
> 2. *"With LoRA rank $r=16$ applied across the 7 projection layers, the low-rank adapter matrices added up to 41.9 million trainable parameters, which is exactly 0.58% of the total model."*

---

### 🟡 Q6: What is the purpose of Alpaca prompt formatting and EOS stop tokens?
> **Probability:** 🟡 **MEDIUM (65% chance — Practical SFT Data Engineering)**

💡 **Aasaan Bhasha Mein Samajhein:**
* **Alpaca Format:** Model ko structure sikhata hai ki pehle Instruction aayega, fir Input (Schema + Question), aur fir Response (SQL query).
* **EOS (End-of-Sequence) Token:** Jaise sentence ke end mein full stop (.) lagta hai, waise hi SQL query ke end mein `eos_token` lagana padta hai. Agar EOS token train na karein, toh model query khatam hone ke baad bhi generate karta rahega aur infinite loop mein phas jayega.

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"The Alpaca template provides structured delimiters (`Instruction`, `Input`, `Response`) so the model clearly distinguishes between the schema context and the target SQL output."*
> 2. *"Appending the EOS token at the end of the SQL response is essential during SFT. It teaches the model when to stop generating, preventing infinite repetitive text generation during production inference."*

---

## 🚀 Section 3: Production, Evaluation & Agent Systems

### 🔴 Q7: How do you evaluate a Text-to-SQL model in production?
> **Probability:** 🔴 **HIGH (80%+ chance — MLOps / LLMOps Evaluation)**

💡 **Aasaan Bhasha Mein Samajhein:**
* SQL ko text matching (BLEU / ROUGE score) se judge **nahi kar sakte**! Kyunki do alag SQL queries same result de sakti hain (jaise ek ne `JOIN` use kiya aur doosre ne `WHERE` subquery).
* Real-world evaluation metric hota hai **Execution Accuracy (EX)**:
  - Model ki generated query aur correct query dono ko actual database engine par chalao.
  - Agar dono tables ka result identical aaya, toh model pass hai!
* Doosra metric hai **Valid Syntax Rate (VSR)**: Kitni queries bina kisi syntax error ke run hui.

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"Text similarity metrics like BLEU or ROUGE are ineffective because two syntactically different SQL queries can return the exact same correct result table."*
> 2. *"The primary metric is Execution Accuracy (EX): we execute both the generated SQL and ground-truth SQL on an actual database engine and check if their returned dataframes match."*
> 3. *"We also track Valid Syntax Rate (VSR) to measure how often the model produces executable, error-free SQL."*

---

### 🟡 Q8: How does SQLCoder-Lite fit into an Enterprise Agentic Architecture?
> **Probability:** 🟡 **MEDIUM (70% chance — Full-Stack AI System Design)**

💡 **Aasaan Bhasha Mein Samajhein:**
Production mein model akela nahi hota, ek Agent loop ke andar kaam karta hai:
1. **User Question:** User bolta hai: *"Mujhe top 5 sales wale employees dikhao."*
2. **Schema Injection:** Agent relevant table ka DDL schema nikalta hai aur SQLCoder-Lite ko bhejta hai.
3. **Inference (SQLCoder-Lite):** SQLCoder-Lite instant clean SQL query return karta hai.
4. **Safety Guardrail:** System check karta hai ki query sirf `SELECT` ho (kisi ne `DROP` ya `DELETE` toh nahi daal diya).
5. **DB Execution & Auto-Retry:** Query run hoti hai; agar koi database error aaya toh error message wapas model ko pass karke self-correct kiya jata hai.

🎙️ **Interview Mein Aise Bolein (English):**
> 1. *"SQLCoder-Lite functions as the specialized SQL generation engine inside a ReAct agent loop."*
> 2. *"The agent fetches table schemas, calls SQLCoder-Lite for deterministic SQL generation, passes the query through a security validator to block mutations like DROP or DELETE, executes it against the database, and auto-retries if syntax errors occur."*
