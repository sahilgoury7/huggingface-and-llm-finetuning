# 📘 Day 5: Supervised Fine-Tuning (SFT) & Unsloth GPU Acceleration

> **Track:** Gen AI Batch 9 | LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Target:** Model ko ChatGPT jaisa Q&A assistant banana aur Unsloth se 5x fast train karna!  

---

# 📖 Part 1: Glossary (Sabhi Terms Ka Seedha Matlab)

| Term | Ek Line Me Asli Matlab | Asli Zindagi Ka Example |
| :--- | :--- | :--- |
| **Pre-training** | Internet ke trillions of words padha kar model ko sirf sentence complete karna sikhana. | Bachhe ko library me baithakar 10,000 kitabein padhana (gyaan bohot hai, par baat karni nahi aati). |
| **SFT (Supervised Fine-Tuning)** | Model ko Instruction aur Response ke pairs dikha kar ek helpful, polite assistant banana. | Bachhe ko viva/interview table par baitha kar sawaal ka seedha jawab dena sikhana. |
| **Instruction $\to$ Response** | SFT data ka standard format: Ek sawaal (Instruction) aur uska ideal jawab (Response). | Teacher ka question paper aur uski ideal answer sheet. |
| **Prompt Template** | CSV ke alag-alag columns (Question, Answer) ko ek single formatted string me convert karna. | Chithhi ko lifafe me daal kar uspe "From" aur "To" ka thappa lagana. |
| **Alpaca Format** | Stanford ka banaya gaya standard format jisme `### Instruction:` aur `### Response:` ke tags hote hain. | Exam paper ka standard format: "Sawaal yahan hai" aur "Jawab yahan likho". |
| **ChatML Format** | OpenAI ka standard format jisme `<\|im_start\|>user` aur `<\|im_start\|>assistant` tags hote hain. | WhatsApp chat bubbles (Left = User, Right = Assistant). |
| **Llama-3 Format** | Meta LLaMA-3 ka official format jisme `<\|start_header_id\|>user<\|end_header_id\|>` hota hai. | Llama-3 ka official ID card jisse wo pehchanta hai ki kaun bol raha hai. |
| **Response-Only Loss** | Model ki galti (Loss) sirf uske diye gaye JAWAB par calculate karna, sawaal par nahi. | Exam me teacher sirf Answer sheet par marks deta hai, question paper copy karne par nahi. |
| **`-100` (Ignore Index)** | PyTorch ka secret "Skip Code" ya Kaali Parchi! Jis token ka label `-100` hota hai, computer usko check nahi karta. | Exam question paper ke upar chipkai gayi kaali parchi jo examiner ko bolti hai: "Isko skip karo!" |
| **Unsloth** | Ek open-source GPU acceleration library jo fine-tuning ko 5x fast karti hai aur 70% VRAM bachati hai. | Aam family car ke badle Formula 1 (F1) racing car ka engine lagana! |
| **Triton GPU Kernels** | Low-level fast code jo seedha GPU chip se baat karta hai bina kisi beech ke slow software ke. | Train me beech ke saare stations skip karke direct superfast express chalana. |
| **TRL Library** | Hugging Face ki special library jo Post-Training (SFT, DPO, RLHF) ke liye bani hai. | Fine-tuning ke liye bana advanced toolkit box. |
| **SFTTrainer** | TRL library ka ready-made trainer jo prompt formatting aur `-100` loss masking apne aap sambhalta hai. | Ek smart automatic checking machine jo sirf student ke answers check karti hai! |

---

# 💡 Part 2: Important Things (Story-by-Story Concepts)

### Kahani 1: Base Model vs SFT Model (Sentence Completer vs Helpful Assistant)

#### 1. Base Model Kyun Pagal Hota Hai?
Jab LLaMA ya Mistral internet padh kar nikalta hai, toh wo koi Assistant nahi hota! Wo sirf ek **Sentence Completer** hota hai.
* Agar aap Base Model se bologe: *"What is the capital of France?"*
* Wo sochega koi exam paper chal raha hai, aur aage naye sawal generate kar dega:  
  *"What is the capital of Germany? What is the capital of Italy?"* 🤦‍♂️

#### 2. SFT Usse Kya Banata Hai?
SFT me hum model ko hazaron aise examples dikhate hain:
```text
┌────────────────────────────────────────────────────────┐
│ INSTRUCTION : What is the capital of France?           │
│ RESPONSE    : The capital of France is Paris.          │
└────────────────────────────────────────────────────────┘
```
Isse model ka dimaag train ho jata hai ki:  
**"Jab bhi koi mujhse sawaal pooche, toh aage ka sentence complete karne ke bajaye seedha uska helpful answer dena hai!"**

---

### Kahani 2: Prompt Templates (CSV Rows se AI ka Text Banana)

#### 1. Problem:
AI model ko CSV files ya Excel sheet samajh nahi aati! AI model ko sirf **ek lamba text (string)** padhna aata hai.

#### 2. Solution (Alpaca Template):
Hum Python ke ek simple f-string se Excel ke har row ko aise format kar dete hain:

```python
formatted_prompt = f"""### Instruction:
{row['Question']}

### Response:
{row['Answer']}"""
```

* **Fayda:** Model training me hazaron baar `### Instruction:` aur `### Response:` dekh kar samajh jata hai ki user ka sawaal kahan khatam hua aur uska jawab kahan se shuru karna hai!

---

### Kahani 3: The 3 Big Prompt Formats (Alpaca vs ChatML vs Llama-3)

Alag-alag models alag-alag formats pasand karte hain. Interview me yeh 3 formats standard hain:

```text
1. Stanford Alpaca Format (Hamein Capstone me yahi use karna hai):
### Instruction:
What is Machine Learning?

### Response:
Machine Learning is a subset of AI that learns from data.
─────────────────────────────────────────────────────────────────
2. OpenAI ChatML Format:
<|im_start|>user
What is Machine Learning?<|im_end|>
<|im_start|>assistant
Machine Learning is a subset of AI that learns from data.<|im_end|>
─────────────────────────────────────────────────────────────────
3. Meta LLaMA-3 Official Format:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
What is Machine Learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Machine Learning is a subset of AI that learns from data.<|eot_id|>
```

---

### Kahani 4: Response-Only Loss & `-100` (Kaali Parchi)

#### 1. Sawaal par Marks Kyun Nahi Katne Chahiye?
Agar user poochta hai: *"2 + 2 kitna hota hai?"* aur AI jawab deta hai *"5 hota hai"*:
* Galti kisne ki? **AI ne ki!**
* Sawaal toh user ne type kiya tha, usme AI ki koi galti nahi hai.
* Isliye training me model ki galti (Loss) **sirf JAWAB par calculate hoti hai, sawaal par nahi!**

#### 2. Computer ko kaise pata chalta hai? (`-100` Magic Number):
Computer ke liye saare words numbers (Token IDs) hote hain, aur saare real token IDs **positive numbers** hote hain (`0, 1, 2, 3...`).

PyTorch me rule bana hai:
> **"Agar kisi token ke label me `-100` likha ho, toh computer usko CHECK NAHI KAREGA (Ignore kar dega)!"**

```text
Tokens:   [###] [Instruction:] [What] [is] [AI?]   [###] [Response:] [AI] [is] [smart.]
Labels:    -100      -100       -100  -100  -100    -100     -100     45   89    102
           └──────────────────┬─────────────────┘                   └────────┬────────┘
             KAALI PARCHI: IGNORE! (Zero Loss)                   MARKS YAHAN MILENGE!
```

* Saare Instruction tokens ko label milta hai: **`-100`**
* Saare Response tokens ko label milta hai: **Asli Token IDs**
* Model apna poora dimaag sirf sahi jawab generate karne me lagata hai!

---

### Kahani 5: Unsloth GPU Acceleration (Colab T4 Ki Supercar)

#### 1. Problem:
Normal Hugging Face se Colab T4 par model train karne me **2 se 3 ghante** lagte hain aur VRAM bharne ka darr rehta hai.

#### 2. Unsloth Ka Jadu (3 Numbers):
1. **5x Faster Training:** 2 ghante ka kaam sirf **25 se 30 minute** me!
2. **70% VRAM Free:** GPU par load bohot kam padta hai, kabhi crash nahi hota.
3. **0% Accuracy Loss:** Model ki samajh me koi kami nahi aati.

#### 3. Yeh Itna Fast Kyun Hai?
Normal PyTorch har kaam ke liye bana hai, isliye thoda slow chalta hai. Unsloth ke developers ne transformer ke core math ko **Triton GPU kernels** me manually rewrite kiya, jo seedha GPU chip se baat karta hai!

---

### Kahani 6: The 5-Step SFT End-to-End Execution Blueprint

```python
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Step 1: Model & Tokenizer Load Karo (Unsloth 4-bit)
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True
)

# Step 2: LoRA Adapter Attach Karo
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none"
)

# Step 3: Dataset Format Function (Alpaca)
alpaca_prompt = """### Instruction:
{}

### Response:
{}"""

def format_prompts(examples):
    instructions = examples["Question"]
    outputs      = examples["Answer"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = alpaca_prompt.format(instruction, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(format_prompts, batched=True)

# Step 4: SFTTrainer Configure Karo
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs"
    )
)

# Step 5: Train & Save Adapters!
trainer.train()
model.save_pretrained("my_lora_adapter")
```

---

# 🎯 Part 3: Interview Q&A (Seedha Muh Se Bolne Wala Format)

### Sawaal 1: Pre-trained Base Model aur SFT Fine-Tuned Model me kya farq hai?
* **Aapka Jawab (Hinglish):**  
  *"Sir, pre-trained base model sirf text completion janta hai (next-token prediction). Agar hum usse koi sawaal poochenge toh wo jawab dene ke bajaye aage naye sawaal likh dega. SFT (Supervised Fine-Tuning) me hum model ko Instruction aur Response ke pairs dikha kar train karte hain, jisse wo ek helpful conversational assistant (jaise ChatGPT) ban jata hai aur user ke sawaal ka seedha, accurate jawab deta hai."*
* **Aapka Jawab (English):**  
  *"A pre-trained base model is fundamentally a raw next-token predictor. When prompted with a question, it might simply continue listing more questions rather than answering. Supervised Fine-Tuning (SFT) trains the model on curated Instruction-Response pairs, teaching it conversational structure and turning it into a helpful, instruction-following AI assistant."*

---

### Sawaal 2: SFT me Prompt Template (jaise Alpaca format) kyun zaroori hota hai?
* **Aapka Jawab (Hinglish):**  
  *"Model ko raw text dene ke bajaye hum prompt template use karte hain taaki model ko clear boundaries pata chalein. Alpaca format me hum `### Instruction:` me user ka sawaal aur `### Response:` me model ka jawab daalte hain. Isse model train ho jata hai ki inference ke time use sirf `### Response:` ke aage ka text generate karna hai."*
* **Aapka Jawab (English):**  
  *"A prompt template provides structured delimiters (such as `### Instruction:` and `### Response:` in Alpaca format) to separate user prompts from assistant completions. Without templates, the model cannot distinguish between context and response. Templates condition the model to reliably trigger text generation immediately following the response delimiter."*

---

### Sawaal 3: Response-Only Loss Masking kya hai, aur `-100` ka kya role hai?
* **Aapka Jawab (Hinglish):**  
  *"SFT me hum model ko sirf sahi response generate karna sikhana chahte hain, na ki user ka prompt yaad karna. Isliye prompt ke saare tokens ke label ko `-100` set kar diya jata hai. PyTorch ke loss calculation me `-100` ignore index hota hai, jisse prompt par koi loss calculate nahi hota aur gradient sirf response tokens se flow hota hai."*
* **Aapka Jawab (English):**  
  *"In SFT, the objective is to optimize response generation without penalizing the model for user-provided prompts. We mask all prompt tokens by assigning them a label of `-100`, which is the default ignore index in PyTorch's CrossEntropyLoss. As a result, loss and backpropagation occur exclusively over the completion tokens, drastically improving training efficiency."*

---

### Sawaal 4: Unsloth kya hai aur hum isse standard training ke mukable kyun prefer karte hain?
* **Aapka Jawab (Hinglish):**  
  *"Sir, Unsloth ek GPU acceleration library hai jo LLMs ki fine-tuning ko 2x se 5x fast banati hai aur 70% VRAM memory bachati hai. Isne transformer ke core math ko custom Triton GPU kernels me rewrite kiya hai. Iska fayda yeh hota hai ki Colab ke free 16GB T4 GPU par jo training 2 ghante leti, wo Unsloth se sirf 25-30 minute me bina kisi accuracy loss ke complete ho jaati hai."*
* **Aapka Jawab (English):**  
  *"Unsloth is an open-source library that accelerates LLM fine-tuning by 2x to 5x while cutting VRAM consumption by up to 70%, with zero accuracy loss. It achieves this by rewriting core transformer operations directly in manual OpenAI Triton GPU kernels. This allows us to train 8B models on a free 16GB Google Colab T4 GPU in just 25 to 30 minutes instead of hours."*

---

### Sawaal 5: Normal `Trainer` aur `SFTTrainer` me kya farq hai?
* **Aapka Jawab (Hinglish):**  
  *"Sir, normal `Trainer` general-purpose hota hai jisme tokenization aur custom masking manually karni padti hai. Jabki `SFTTrainer` (TRL library) specifically Instruction tuning ke liye bana hai. Yeh Prompt formatting (Alpaca format) aur Response-Only Loss Masking (`-100`) ko automatically background me handle kar leta hai, jisse hume custom collators nahi likhne padte."*
* **Aapka Jawab (English):**  
  *"The standard Hugging Face `Trainer` requires manual tokenization and custom collators for token masking. In contrast, `SFTTrainer` from the `trl` library is built specifically for Supervised Fine-Tuning. It natively supports prompt formatting functions and automatically applies completion-only loss masking, making instruction tuning seamless and less prone to manual preprocessing errors."*
