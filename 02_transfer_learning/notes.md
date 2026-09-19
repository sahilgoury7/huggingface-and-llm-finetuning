# 02: Tokenization Mechanics, Datasets & Transfer Learning

> **Session 10 & 11 | Gen AI Batch 9**  
> **Instructor:** Ashish Jangra  
> **Learner:** Sahil Goury  

---

# 📖 Part 1: Glossary (Key Terms & Definitions)

| Term | Simple Meaning | Why It Matters |
| :--- | :--- | :--- |
| **Tokenization** | Raw text ko chote numerical chunks (tokens) me todne ka process. | AI models English nahi samajhte, sirf numbers process karte hain. |
| **Word-level Tokenization** | Text ko direct space se shabdon me split karna. | OOV (Out-of-Vocabulary) problem deta hai aur vocabulary bohot giant banata hai. |
| **Character-level Tokenization** | Har single letter/character ko token banana. | Sequence length 10x badha deta hai, jisse GPU VRAM crash ho sakti hai. |
| **Sub-word Tokenization** | Common words ko poora aur rare/naye words ko meaningful tukdon me todna (`re` + `play` + `ing`). | LLMs ka global standard hai; 0% OOV aur compact vocabulary deta hai. |
| **OOV (Out-Of-Vocabulary)** | Jab user koi aisa word type kare jo model ki dictionary me nahi hai (`[UNK]`). | Sub-word tokenization ne OOV problem ko duniya se khatam kar diya. |
| **BPE (Byte-Pair Encoding)** | Most frequent adjacent character pairs ko merge karke vocab banane wala algorithm. | GPT-2, GPT-4 aur RoBERTa ka core tokenization engine hai. |
| **WordPiece** | Statistical likelihood ke basis par sub-words merge karne wala algorithm. | BERT aur DistilBERT me use hota hai (subwords ke aage `##` lagata hai). |
| **SentencePiece / Byte-BPE** | Whitespace ko bhi normal character (`_`) maankar bina language dependency ke tokenize karna. | Modern LLMs (LLaMA-3, Mistral, DeepSeek) ka standard hai. |
| **Vocabulary (Vocab Size)** | Model ke tokenizer ki unique tokens ki total dictionary size. | BERT = ~30k, LLaMA-2 = ~32k, LLaMA-3 = ~128k tokens. |
| **Special Tokens** | Model architecture ke control signals (`[CLS]`, `[SEP]`, `<pad>`, `<s>`, `</s>`). | Sentence boundaries, classification aur padding batane ke liye mandatory hain. |
| **Dynamic Padding** | Batch ke sabse lambe sentence ke barabar hi padding lagana (fixed max_length ke bajaye). | 4x training speed aur 50% GPU memory bachta hai. |
| **Attention Mask** | 1s aur 0s ki binary switch list (1 = Real Token, 0 = Padded Token). | Model ko batata hai ki mathematical attention sirf asli words par lagani hai, `<pad>` par nahi. |
| **Parameter** | Model ke andar ka ek single learned weight/number (e.g. 0.4852). | 8B model ka matlab hai 800 Crore numbers/parameters. |
| **Precision** | Ek number ko computer me store karne ke liye kitne bits use ho rahe hain. | FP32 (4 Bytes), FP16 (2 Bytes), INT8 (1 Byte), 4-bit NF4 (0.5 Byte). |
| **Quantization** | Model ke weights ko compress karke kam bits (jaise 16-bit se 4-bit) me convert karna. | Bade models ko chote/free GPUs (jaise Colab T4) par train karne ke liye zaroori hai. |

---

# 💡 Part 2: Important Things (Rules, Architecture & Core Mechanics)

### 1. Why Sub-word Tokenization Won NLP?

```text
Raw Word: "replaying"
├── Word-level: ["replaying"] ──▶ If not in vocab: [UNK] (Fails)
├── Character-level: ['r','e','p','l','a','y','i','n','g'] ──▶ 9 tokens (Too long, quadratic attention blowup)
└── Sub-word level: ["re", "play", "ing"] ──▶ 3 meaningful tokens (Understands prefix, root & suffix!)
```

---

### 2. The Big 3 Tokenization Algorithms Comparison

| Feature | 1. BPE (Byte-Pair Encoding) | 2. WordPiece | 3. SentencePiece / Byte-BPE |
| :--- | :--- | :--- | :--- |
| **Models** | **GPT family, RoBERTa** | **BERT, DistilBERT** | **LLaMA-3, DeepSeek-R1, Mistral** |
| **Core Logic** | Frequency based adjacent character pairs merge karta hai. | Likelihood maximize karne wale pairs merge karta hai. | Whitespace ko normal character (`_`) maanta hai, direct bytes process karta hai. |
| **Symbol** | `Ġ` (Space token) | `##` (jaise `##ing`) | ` ` (SentencePiece space) |
| **Multi-lingual** | Moderate | Moderate | **Best (Hindi, Code, Multi-lingual)** |

---

### 3. Special Tokens & Traffic Control

| Token | BERT / DistilBERT | LLaMA / Mistral / DeepSeek | Purpose |
| :--- | :--- | :--- | :--- |
| **BOS (Beginning)** | `[CLS]` | `<s>` / `<\|begin_of_text\|>` | Start of text marker; classification embedding. |
| **EOS (Ending)** | `[SEP]` | `</s>` / `<\|end_of_text\|>` | End of text delimiter. |
| **PAD (Padding)** | `[PAD]` | `<pad>` / `<\|finetune_right_pad_id\|>` | Fills batch rows to make square rectangular tensors. |
| **UNK (Unknown)** | `[UNK]` | `<unk>` | Fallback token for completely unknown bytes. |

---

### 4. Static vs Dynamic Padding & Attention Mask

```text
Sentence:        "Good morning" (Chota sentence)
Tokens:        [ "Good", "morning",  <pad>,  <pad>,  <pad> ]
input_ids:     [  2204,    2851,       0,      0,      0   ]
attention_mask:[    1,       1,        0,      0,      0   ]
                    ▲        ▲         ▲       ▲       ▲
               (Real words: 1)     (Padding: 0, model ignores completely!)
```

* **The Restaurant Table Analogy:** Static padding 2 logon ko bhi 50-seater table deta hai (waste). Dynamic padding group ke size ke hisab se table jodta hai.
* **Static Padding (`padding="max_length", max_length=512`):** Har sample ko 512 tak zabardasti stretch karta hai. 80% VRAM waste hoti hai.
* **Dynamic Padding (`DataCollatorWithPadding`):** Sirf current batch ke longest sentence ke barabar pad karta hai. **50% VRAM bachti hai aur training 3x-4x fast hoti hai.**

#### 🎙️ Interview me Bolne ka Tareeqa (Dynamic Padding):
> *"Sir, dono me main difference **GPU memory optimization** ka hai:*  
> *1. **Static Padding** me hum poore dataset ko ek fixed length (jaise 512 tokens) tak zabardasti pad kar dete hain. Agar hamara sentence sirf 20 tokens ka hai, tab bhi bache hue 492 tokens empty `<pad>` ban jaate hain. Isse GPU faltu ke zero tokens par self-attention calculate karta hai aur bohot saari VRAM waste hoti hai.*  
> *2. Jabki **Dynamic Padding** me hum Hugging Face ka `DataCollatorWithPadding` use karte hain. Yeh har batch ko scan karta hai aur **sirf us batch ke longest sentence ke barabar** padding lagata hai.*  
> *Iska fayda yeh hota hai ki hamari **GPU VRAM 50% bachti hai** aur training time **3 se 4 guna fast** ho jata hai."*

---

### 5. Hugging Face `datasets` Engine (Zero RAM Crash)

```text
Pandas (pd.read_csv) ──▶ Poora Data RAM me copy hota hai ──▶ Crash on large datasets (>8GB)
HF `datasets` ──────────▶ Apache Arrow (mmap on disk) ────▶ Zero RAM crash, instant batch streaming
```

* **`load_dataset()`:** Data SSD/Hard Disk par hi rehta hai. Training ke dauran sirf required batch RAM me aata hai (Zero-Copy Architecture).
* **`.map(batched=True)`:** Multi-core parallel tokenization karta hai (1000 rows per batch), speeding up preprocessing by 10x-50x.
* **`remove_columns` Rule:** Raw string columns (`'input'`, `'output'`) ko tokenization ke baad remove karna mandatory hota hai, warna PyTorch `Trainer` tensor conversion error (`TypeError: can't convert string to Tensor`) de deta hai.

#### 🎙️ Interview me Bolne ka Tareeqa (Pandas vs load_dataset):
> *"Sir, hum 100% **Hugging Face `load_dataset()`** use karenge, Pandas nahi.*  
> *Pandas poore data ko RAM me load karne ki koshish karega jisse hamara system crash (OOM) ho jayega.  
> Jabki Hugging Face **Apache Arrow memory-mapping (`mmap`)** use karta hai — poora data disk/SSD par rehta hai aur training ke time sirf current batch memory me stream hota hai, isliye 50GB dataset bhi 16GB RAM par bina crash hue chalta hai."*

---

### 6. Ashish Sir ka Magic Formula: Model Parameter & VRAM Math (Cell 10 & 12)

#### Step 1: Bits ko Bytes me convert karna ($8\text{ bits} = 1\text{ byte}$)
* **32-bit (FP32):** $32 / 8 = \mathbf{4\text{ Bytes}}$ per parameter
* **16-bit (FP16 / BF16):** $16 / 8 = \mathbf{2\text{ Bytes}}$ per parameter
* **8-bit (INT8):** $8 / 8 = \mathbf{1\text{ Byte}}$ per parameter
* **4-bit (NF4 / QLoRA):** $4 / 8 = \mathbf{0.5\text{ Byte}}$ (Aadha Byte!) per parameter

#### Step 2: Memory Calculation Formula
$$ \text{Model Weight (GB)} \approx \text{Parameters (in Billions)} \times \text{Bytes per Parameter} $$

| Model Size | FP32 (4 Bytes) | FP16 (2 Bytes) | INT8 (1 Byte) | 4-bit NF4 (0.5 Byte) |
| :--- | :--- | :--- | :--- | :--- |
| **1.5B (DeepSeek-R1)** | 6.0 GB | 3.0 GB | 1.5 GB | **0.75 GB (~750 MB)** |
| **2B (Gemma-2B)** | 8.0 GB | 4.0 GB | 2.0 GB | **1.0 GB** |
| **8B (Llama-3 8B)** | 32.0 GB | 16.0 GB | 8.0 GB | **4.0 GB** |

#### Step 3: The "Kamra aur Foldable Sofa" Rule (Why Colab T4 needs 4-bit)
* **16GB Colab GPU = 16-foot Room.**
* **8B Model in FP16 = 16-foot Sofa.**  
  Sofa akele poore kamre me phans jata hai. Training ke rough work (gradients, optimizer) ke liye jagah nahi bachti $\to$ **CUDA OOM Crash!**
* **8B Model in 4-bit = 4-foot Folded Sofa.**  
  Model sirf 4GB leta hai, baaki **12GB free space** bachti hai training aur fine-tuning ke liye!

#### 🎙️ Interview me Bolne ka Tareeqa (VRAM Math & 4-bit Quantization):
> *"Sir, model ki VRAM requirement calculate karne ka formula hai: `Parameters in Billions × Bytes per Parameter`.*  
> *FP16 me har parameter 2 Bytes leta hai, isliye ek 8B model 16GB VRAM sirf weights load karne me le leta hai. Free Colab T4 ke paas sirf 16GB total VRAM hoti hai, isliye FP16 me training ke liye memory nahi bachti aur OOM error aa jata hai.*  
> *Iska solution hai **4-bit Quantization (QLoRA)**. 4-bit me har parameter sirf 0.5 Byte leta hai, jisse wahi 8B model **sirf ~4GB me load** ho jata hai aur baaki 12GB free VRAM me hum aasaani se LoRA adapters train kar lete hain."*

---

# 🎯 Part 3: Q&A (Interview & Concept Mastery)

### Q1: Sub-word tokenization me BPE, WordPiece aur SentencePiece me kya farak hai?
**Answer:**  
- **BPE (GPT family):** Frequency based adjacent characters ko merge karta hai. Space ke liye `Ġ` symbol use karta hai.
- **WordPiece (BERT family):** Merge karte waqt language model likelihood ko maximize karta hai. Sub-words ke aage `##` lagata hai (e.g. `['play', '##ing']`).
- **SentencePiece (LLaMA/DeepSeek):** Raw text ko direct bytes ki tarah leta hai aur spaces ko `_` character maanta hai. Yeh language-independent hai aur multi-lingual text ke liye best hai.

---

### Q2: Static padding ke muqable Dynamic padding se training fast kyun hoti hai?
**Answer:**  
Static padding (`max_length=512`) har sample ko 512 tokens tak kheench deta hai, chahe sentence sirf 20 words ka ho. GPU apna 80% time faltu ke `<pad>` tokens par self-attention matrix multiply karne me barbad karta hai.  
Dynamic padding (`DataCollatorWithPadding`) har batch ko sirf us batch ke longest sentence ke barabar pad karta hai (e.g. 40 tokens), jisse 50% VRAM bachti hai aur training 3x-4x fast hoti hai.

---

### Q3: Model ko fine-tune karte waqt `remove_columns=['input', 'output']` lagana kyun zaroori hai?
**Answer:**  
PyTorch `Trainer` aur `DataLoader` model ko sirf numerical tensors (`input_ids`, `attention_mask`, `labels`) supply karte hain. Agar raw string columns dataset me reh gaye, toh PyTorch data collator text strings ko tensor me convert nahi kar payega aur `TypeError` throw karega.

---

### Q4: Ek 8 Billion parameter model ko FP32, FP16 aur 4-bit me load karne ke liye kitni VRAM chahiye?
**Answer:**  
- **FP32 (4 Bytes/param):** $8 \times 4 = 32\text{ GB}$ (Dual A100 GPU chahiye).
- **FP16 / BF16 (2 Bytes/param):** $8 \times 2 = 16\text{ GB}$ (Single T4 par just fit, training nahi ho sakti).
- **4-bit NF4 (0.5 Bytes/param):** $8 \times 0.5 = 4\text{ GB}$ (Free Colab T4 16GB par aasaani se train ho sakta hai).

---

### Q5: Hugging Face `datasets` library 100GB dataset ko bina RAM crash kiye kaise load kar leti hai?
**Answer:**  
Hugging Face `datasets` **Apache Arrow** format par built hai jo **Memory-Mapping (mmap)** use karta hai. Yeh poore data ko RAM me copy karne ke bajaye disk par map karta hai aur sirf wahi batch RAM me lata hai jo model ko forward pass ke liye chahiye hota hai (Zero-copy architecture).

---

### Q6: Google Colab T4 GPU (16GB) par 8B model ko train karne ke liye 4-bit QLoRA kyun mandatory hai?
**Answer:**  
16GB VRAM me FP16 par 8B model sirf load ho sakta hai (16GB), lekin training ke dauran optimizer states (AdamW takes 8 bytes per param) aur gradients ko store karne ke liye extra 40GB+ VRAM chahiye hoti hai.  
4-bit QLoRA me base model 4GB me freeze ho jata hai aur training sirf tiny LoRA adapters (<100MB) par hoti hai, jisse total VRAM 7-8GB rehti hai aur Colab T4 par bina crash hue training chalti hai.

---

# 🚀 Day 3: Transfer Learning & The Hugging Face `Trainer` API

### 7. Transfer Learning: Pre-training vs Fine-Tuning

```text
Pre-Training ($10M+ cost, Trillions of web tokens) ──▶ Base Model (Llama-3, DeepSeek)
                                                                 │
                                                       Domain Adaptation
                                                                 ▼
Fine-Tuning (Specific task, e.g. Healthcare, Legal) ◀── 2,000 to 50,000 QA pairs
```

* **Feature Extraction (Frozen Backbone):** Model ke transformer layers ko freeze karke sirf aakhri classification head train karna. (Fast, low memory, but limited domain reasoning).
* **Full Fine-Tuning:** Saare layers ke weights ko update karna. (High compute, risks catastrophic forgetting).

---

### 8. Catastrophic Forgetting & Its Prevention

* **Problem:** Jab pre-trained model naye data par train hote waqt apna purana general reasoning aur language knowledge poori tarah bhool jaye.
* **Solutions (Industry Standard):**
  1. **Super Small Learning Rate (`learning_rate = 2e-5`):** Weights ko bohot gently adjust karna taaki pre-trained knowledge destroy na ho.
  2. **Warmup Steps (`warmup_steps = 100` ya `warmup_ratio = 0.1`):** Shuruati 100 steps me learning rate 0 se gradually target tak badhana.

#### 🎙️ Interview me Bolne ka Tareeqa (Catastrophic Forgetting):
> *"Sir, Catastrophic Forgetting tab hota hai jab fine-tuning ke dauran model naye data ko seekhte waqt apna pre-trained general knowledge bhool jata hai.*  
> *Isko prevent karne ke liye hum:*  
> *1. Bohot small learning rate (`2e-5` to `5e-5`) use karte hain.*  
> *2. Learning rate warmup schedule lagate hain.*  
> *3. Parameter-Efficient Fine-Tuning (PEFT/LoRA) use karte hain jisme base model 100% freeze rehta hai."*

---

### 9. The `TrainingArguments` Steering Wheel

```python
training_args = TrainingArguments(
    output_dir="./results",  # Checkpoints save folder
    per_device_train_batch_size=4,  # Actual samples sent to GPU per step
    gradient_accumulation_steps=2,  # Steps before updating weights (Virtual batch = 4 x 2 = 8)
    learning_rate=2e-5,  # Small learning rate to prevent forgetting
    num_train_epochs=3,  # Full passes through the entire dataset
    logging_steps=10,  # Print loss every 10 steps
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=2,  # Keep only last 2 checkpoints (saves disk space)
    fp16=True,  # 16-bit mixed precision (2x speed, 50% memory saved)
)
```

* **Effective Batch Size Formula:**  
  $$ \text{Effective Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{num\_gpus} $$
* **Epoch vs Step:**
  * **1 Step:** Ek batch par forward pass aur weight update.
  * **1 Epoch:** Poore dataset ka ek complete chakkar.

---

# 🎯 Part 3 (Contd.): Transfer Learning & Trainer Interview Q&A

### 10. `DataCollatorForLanguageModeling(mlm=False)` vs `mlm=True`

```text
mlm=True  ──▶ Masked LM (BERT)    ──▶ Sentence ke beech ka word chupana ("The sky is [MASK]")
mlm=False ──▶ Causal LM (DeepSeek)──▶ Left-to-Right Next Token Prediction (Standard for LLM Fine-Tuning!)
```

* **Causal LM Standard:** LLMs (Llama-3, DeepSeek, GPT) prompt padhkar aage ka agla word generate karte hain. Isliye fine-tuning me hamesha **`mlm=False`** mandatory hota hai.

---

### 11. Loss Curves & The "Loss = NaN" Debugging Cheatsheet

| Scenario | Screen par kya dikhega? | Asli Problem | Sahi Ilaaj |
| :--- | :--- | :--- | :--- |
| **Normal Learning** | $2.8 \to 2.1 \to 1.6 \to 1.2$ | Model sahi seekh raha hai. | Continue training. |
| **Exploding Gradients** | $2.5 \to 5.0 \to 12.0 \to \text{NaN}$ | **`learning_rate` bohot badi hai!** | LR ko chota karo (`2e-5`) + `max_grad_norm=1.0`. |
| **Overfitting** | Train Loss = 0.05, Val Loss = 3.5 | **Epochs zyada hain!** (Data ratta maar liya). | Epochs kam karo (2-3 max) ya dropout badhao. |
| **Underfitting** | Loss 2.8 par freeze ho gaya | LR bohot zyada choti hai ya epochs kam hain. | LR thodi badhao (`5e-5`). |

---

### 12. The 5-Step Master Recipe (No-Code Blueprint & Spoken Pitch)

```text
                     LLM TRAINING KA 5-STEP RECIPE
                                   │
┌──────────────────────────────────┼──────────────────────────────────┐
▼                                  ▼                                  ▼
[ Step 1: Raw Data ]     [ Step 2: Tokenization ]          [ Step 3: Waiter ]
Dataset load karo        Text ko Numbers me badlo          Data Collator lagao
(Bina RAM crash kiye)    (Q + A jodkar, Text delete)       (Padding + 1s aur 0s)
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
      [ Step 4: Rules & Steering ]         [ Step 5: Gym Trainer ]
      TrainingArguments set karo           `Trainer` loop chalao
      (Speed, Batches, Checkpoints)        (Loss kam hoga, Model save!)
```

* **Step 1 (Raw Data):** Apache Arrow format (`load_dataset`) se data direct disk par map karna taaki 50GB data bhi bina RAM crash huye stream ho sake.
* **Step 2 (Tokenization):** Tokenizer se text ko sub-words aur `input_ids` me todna, question aur answer ko jodna (`inp + " " + out`), aur raw text columns delete karna.
* **Step 3 (Data Collator):** Chote sentences me padding lagana, `attention_mask` (1s aur 0s) dena, aur `mlm=False` se next-token prediction set karna.
* **Step 4 (TrainingArguments):** Speed (`learning_rate=2e-5`), batch size, gradient accumulation, aur checkpoints save limit set karna.
* **Step 5 (Trainer Loop):** Model, Args, Dataset aur Collator ko `Trainer` me pass karke `trainer.train()` dabana. Loss ko monitor karna aur checkpoints save karna.

#### 🎙️ Interview me Bolne ka Tareeqa (5-Step Training Pipeline):
> *"Sir, kisi bhi custom dataset par LLM train karne ka standard 5-step workflow hota hai:*  
> *1. **Data Ingestion:** `load_dataset` se Apache Arrow format me zero-copy memory mapping karna.*  
> *2. **Tokenization Pipeline:** `.map(batched=True)` se question aur answer ko merge karke numerical input_ids me convert karna aur string columns remove karna.*  
> *3. **Dynamic Collation:** `DataCollatorForLanguageModeling(mlm=False)` se batch-level padding aur causal labels create karna.*  
> *4. **Hyperparameter Configuration:** `TrainingArguments` me small learning rate (`2e-5`) aur gradient accumulation configure karna taaki OOM aur Catastrophic Forgetting na ho.*  
> *5. **Training Execution:** `Trainer` loop execute karke training loss track karna aur final fine-tuned model shards export karna."*

---

# 🎯 Part 3 (Contd.): Trainer & Loss Debugging Interview Q&A

### Q7: Fine-Tuning me learning rate standard ML ke muqable itni choti (e.g. `2e-5`) kyun hoti hai?
**Answer:**  
Standard ML me models scratch se train hote hain isliye badi learning rate (`0.01`) chahiye hoti hai. LLMs me model pehle se trillions of tokens par pre-trained hota hai. Badi learning rate puraane calibrated weights ko destroy kar degi jisse **Catastrophic Forgetting** ho jayegi. Isliye `2e-5` (0.00002) jaisi gentle learning rate use hoti hai.

---

### Q8: Chote GPU (Colab T4 16GB) par bada batch size kaise simulate karte hain?
**Answer:**  
Hum **Gradient Accumulation** use karte hain. Agar hum `per_device_train_batch_size = 4` aur `gradient_accumulation_steps = 4` set karein, toh GPU ek baar me sirf 4 samples process karega (OOM avoid hoga), lekin weights 16 samples accumulate hone ke baad update honge. Isse **Effective Batch Size = 16** milta hai bina extra VRAM ke.

---

### Q9: Fine-tuning ke time `Loss = NaN` kyun aata hai aur ise kaise solve karte hain?
**Answer:**  
`Loss = NaN` (Not a Number) tab aata hai jab gradients explode ho jaate hain (Exploding Gradients), aamtaur par bohot high `learning_rate` ya numerical instability ki wajah se. Iska fix:  
1. Learning rate ko chota karna (`2e-5`).  
2. Gradient clipping lagana (`max_grad_norm = 1.0`).  
3. FP16 ke bajaye BF16 use karna (agar hardware support kare).

---

### Q10: LLM fine-tuning me `DataCollatorForLanguageModeling` me `mlm=False` kyun rakhte hain?
**Answer:**  
MLM (Masked Language Modeling) BERT-style models ke liye hota hai jahan beech ke words mask kiye jaate hain. Modern LLMs (Llama, DeepSeek) **Autoregressive / Causal Language Models** hain jo left-to-right agla token predict karte hain. Isliye `mlm=False` batata hai ki collator ko labels ke roop me `input_ids` ko hi use karna hai next-token prediction loss nikalne ke liye.

---