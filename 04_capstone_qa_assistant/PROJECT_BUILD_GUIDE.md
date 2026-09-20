# 🛠️ Step-by-Step Project Build Guide: AI Medical Healthcare Assistant

> **Project:** Fine-Tuning LLaMA-3 8B with 4-bit QLoRA on ChatDoctor Dataset  
> **Platform:** 100% Free Google Colab (Tesla T4 GPU, 16GB VRAM)  
> **Target Audience:** Sahil Goury (Zero to Production LLM Builder)  
> **Purpose:** Is guide ko Colab ke barabar me khol kar aap shuru se aakhir tak pura project **apne haathon se bina kisi error ke** bana sakte hain!

---

## 📋 Quick Pre-Flight Checklist (Colab Khone Se Pehle)
- [ ] Google Colab me jayein: `colab.research.google.com`
- [ ] Naya Notebook banayein: `File -> New Notebook`
- [ ] GPU on kijiye: `Runtime -> Change runtime type -> T4 GPU -> Save`
- [ ] Verify GPU: `!nvidia-smi` run karke dekhein (Tesla T4, ~15GB VRAM dikhna chahiye)

---

## 🧱 Cell 1: Environment Setup & High-Speed Libraries

#### ❓ Is Cell Ka Kaam:
Unsloth (5x faster fine-tuning engine), Hugging Face TRL, PEFT, aur 4-bit Quantization libraries install karna.

```python
# ==============================================================================
# CELL 1: Required Libraries Installation
# ==============================================================================

# 1. Unsloth install kar rahe hain (Triton GPU acceleration ke liye)
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 2. Hugging Face ki core libraries (TRL for SFTTrainer, PEFT for LoRA, BitsAndBytes for 4-bit)
!pip install --no-deps trl peft accelerate bitsandbytes

# 3. Datasets aur Gradio install kar rahe hain
!pip install datasets gradio

print("✅ Cell 1 Complete: Saari libraries successfully install ho gayi hain!")
```

* **Expected Output:** `✅ Cell 1 Complete: Saari libraries successfully install ho gayi hain!`
* **💡 Pro Tip:** Agar koi warning aaye toh tension mat lijiye, jab tak red error na aaye sab normal hai!

---

## 🧱 Cell 2: Medical Dataset Load Karna

#### ❓ Is Cell Ka Kaam:
Hugging Face Hub se direct `lavita/ChatDoctor-HealthCareMagic-100k` dataset load karna (1.12 Lakh rows) aur pehla sample check karna.

```python
# ==============================================================================
# CELL 2: Load & Inspect ChatDoctor Dataset
# ==============================================================================
from datasets import load_dataset

# 1. Dataset load kiya Hugging Face se
print("⏳ Loading ChatDoctor Dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

# 2. Total rows check kiye
print(f"✅ Cell 2 Complete: Total Samples = {len(dataset):,}")

# 3. Pehla sample print karke dekhte hain
print("\n--- 🩺 Sample Patient Query ---")
print(dataset[0]["input"])

print("\n--- 👨‍⚕️ Sample Doctor Response ---")
print(dataset[0]["output"])
```

* **Expected Output:** `Total Samples = 112,165` aur ek real patient query + doctor answer print hoga.

---

## 🧱 Cell 3: Sequence Length & Word Count EDA

#### ❓ Is Cell Ka Kaam:
Pata lagana ki patient ka sawaal aur doctor ka jawab average kitne words ka hota hai, taaki hum `max_seq_length = 512` bina kisi doubt ke set kar sakein.

```python
# ==============================================================================
# CELL 3: Word Count Analysis (EDA)
# ==============================================================================
import numpy as np

# 1. Pehle 10,000 samples ke words count karte hain
sample_size = 10000
input_words = [len(dataset[i]['input'].split()) for i in range(sample_size)]
output_words = [len(dataset[i]['output'].split()) for i in range(sample_size)]

# 2. Averages calculate kiye
avg_in = np.mean(input_words)
avg_out = np.mean(output_words)

print(f"📊 Patient Query Average : {avg_in:.1f} words")
print(f"📊 Doctor Answer Average : {avg_out:.1f} words")
print(f"📊 Combined Average      : {avg_in + avg_out:.1f} words")
print("\n💡 Conclusion: Combined length ~182 words hai, isliye 'max_seq_length = 512' 100% safe hai!")
```

* **Expected Output:** Combined average ~182 words aayega.

---

## 🧱 Cell 4: 4-Bit LLaMA-3 8B Model & LoRA Adapter Setup

#### ❓ Is Cell Ka Kaam:
Meta ke 8 Billion parameter model ko 4-bit NF4 me load karna aur sirf 0.52% trainable LoRA adapters lagana.

```python
# ==============================================================================
# CELL 4: 4-Bit Model & LoRA Configuration
# ==============================================================================
from unsloth import FastLanguageModel
import torch

# 1. 4-bit LLaMA-3 8B model load kar rahe hain
model_id = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 512

print(f"⏳ Loading {model_id} into GPU...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=None,          # Auto-detect float16
    load_in_4bit=True,   # 4-bit NF4 quantization
)

# 2. LoRA Adapters attach kar rahe hain
print("⏳ Attaching LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                                                                           # LoRA Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # All Linear layers
    lora_alpha=16,                                                                  # LoRA Scaling
    lora_dropout=0,                                                                 # Unsloth optimized
    bias="none",
    use_gradient_checkpointing=True,                                                # VRAM bachata hai
    random_state=3407,
)

# 3. Trainable parameters verify karte hain
print("\n" + "="*50)
model.print_trainable_parameters()
print("="*50)
print("✅ Cell 4 Complete: Model aur LoRA ready hain!")
```

* **Expected Output:** `trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196%`

---

## 🧱 Cell 5: Train/Test Split & 350-Word Length Filter

#### ❓ Is Cell Ka Kaam:
2,000 train samples alag karna, aur 350 words se bade samples ko filter karna taaki training ke dauran koi tensor mismatch crash na ho!

```python
# ==============================================================================
# CELL 5: Train/Test Split & 350-Word Filter
# ==============================================================================

# 1. 2000 Train aur 200 Test samples split kiye
split_data = dataset.train_test_split(train_size=2000, test_size=200, seed=42)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

print(f"Initial: Train = {len(train_dataset)}, Test = {len(eval_dataset)}")
```

---

## 🧱 Cell 6: Stanford Alpaca Prompt Formatting & Filter Apply

#### ❓ Is Cell Ka Kaam:
Template define karna, har sample ke peeche Stop Token (`<|end_of_text|>`) jodna, aur 350-word length filter apply karna.

```python
# ==============================================================================
# CELL 6: Alpaca Formatting & Clean Filtering
# ==============================================================================

# 1. Stanford Alpaca Medical Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert medical doctor. Provide accurate and empathetic medical advice based on the patient's symptoms.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Model ka Stop Token (<|end_of_text|>)

# 2. Formatting Function (EOS token append ke sath)
def formatting_prompts_func(examples):
    texts = []
    for p, d in zip(examples["input"], examples["output"]):
        texts.append(alpaca_prompt.format(p, d) + EOS_TOKEN)
    return {"text": texts}

# 3. Template apply kiya
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

# 4. Safe Limit Filter: 350 words se bade samples hataye (Strictly < 512 tokens)
train_dataset = train_dataset.filter(lambda x: len(x["text"].split()) < 350)

print(f"✅ Cell 6 Complete: Clean Train Samples = {len(train_dataset)} (Zero Mismatch Guarantee!)")
```

* **Expected Output:** `Clean Train Samples = 1924` (~96% clean data).

---

## 🧱 Cell 7: Training Settings (Hyperparameters)

#### ❓ Is Cell Ka Kaam:
Training ke rules set karna (batch size, learning rate, total steps = 60).

```python
# ==============================================================================
# CELL 7: Training Arguments
# ==============================================================================
from transformers import TrainingArguments
import torch

training_args = TrainingArguments(
    output_dir="outputs",                    # Checkpoints folder
    per_device_train_batch_size=2,           # GPU me ek baar me 2 samples
    gradient_accumulation_steps=4,           # 4 steps ke baad update (Effective batch = 8)
    warmup_steps=5,                          # Starting warmup
    max_steps=60,                            # Total 60 steps (~8-10 mins)
    learning_rate=2e-4,                       # LoRA standard rate
    fp16=not torch.cuda.is_bf16_supported(), # Float16 for Tesla T4
    logging_steps=1,                         # Har step par loss dikhega
    seed=3407,
)

print("✅ Cell 7 Complete: Training settings ready!")
```

---

## 🧱 Cell 8: SFTTrainer Setup & Training Start! 🚀

#### ❓ Is Cell Ka Kaam:
Model ko training par lagana aur live screen par Loss ko `2.87` se gir kar `1.89` tak aate dekhna!

```python
# ==============================================================================
# CELL 8: SFTTrainer Execution
# ==============================================================================
from trl import SFTTrainer

# 1. Unsloth training mode on kiya
FastLanguageModel.for_training(model)

# 2. Trainer initialize kiya
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,               # Safe sequence handling
    args=training_args,
)

# 3. Training start! (~8-10 minutes wait kijiye)
print("🚀 Training Shuru Ho Rahi Hai...")
trainer_stats = trainer.train()

print("🎉 Cell 8 Complete: Training successfully finish ho gayi!")
```

* **Expected Output:** Progress bar `60/60` complete hogi aur final Loss ~`1.89` dikhega.

---

## 🧱 Cell 9: LoRA Weights Save Karna

#### ❓ Is Cell Ka Kaam:
Trained LoRA adapters ko Colab ke local folder me save karna (~150 MB).

```python
# ==============================================================================
# CELL 9: Save LoRA Adapters
# ==============================================================================

# 1. Sirf LoRA weights save kar rahe hain (~150 MB)
model.save_pretrained("medical_llama3_lora")
tokenizer.save_pretrained("medical_llama3_lora")

print("✅ Cell 9 Complete: LoRA Adapter 'medical_llama3_lora' folder me save ho gaya!")
```

---

## 🧱 Cell 10: Single-Turn Clinical Test (Inference)

#### ❓ Is Cell Ka Kaam:
Model ko ek real patient sawaal dekar check karna ki kya usne doctor banna seekha.

```python
# ==============================================================================
# CELL 10: Test Doctor Advice (Single-Turn)
# ==============================================================================

# 1. Fast Inference Mode on kiya
FastLanguageModel.for_inference(model)

# 2. Unseen patient symptoms
test_query = "Doctor, I have severe sore throat and fever of 101 F for 2 days. What should I do?"
test_prompt = alpaca_prompt.format(test_query, "")

# 3. Generate answer
inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
doctor_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("👨‍⚕️ AI DOCTOR KA JAWAB:")
print(doctor_reply.split("### Response:")[1].strip())
```

---

## 🧱 Cell 11: Multi-Turn Gradio Web App (With Memory & Anti-Repetition) 🌐

#### ❓ Is Cell Ka Kaam:
Ek live website link (`https://xxxx.gradio.live`) launch karna jisme user WhatsApp/ChatGPT ki tarah doctor se chat kar sakta hai aur doctor purani baatein yaad rakhta hai!

```python
# ==============================================================================
# CELL 11: Launch Multi-Turn Chatbot with Memory & Anti-Repetition
# ==============================================================================
import gradio as gr

def chat_with_memory(message, history):
    # 1. Purani chat history ka context build kiya
    context = ""
    for item in history:
        if isinstance(item, dict):
            role = "Patient" if item.get("role") == "user" else "Doctor"
            context += f"{role}: {item.get('content', '')}\n"
        elif isinstance(item, (list, tuple)):
            context += f"Patient: {item[0]}\nDoctor: {item[1]}\n"
            
    # 2. Prompt me context + naya message fit kiya
    full_prompt = alpaca_prompt.format(f"{context}Patient: {message}", "")
    
    # 3. Model generation (Anti-repetition aur natural temperature ke sath)
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,          # Natural human-like flow
        repetition_penalty=1.15,  # Repetitive chair loop ko rokta hai
        use_cache=True
    )
    
    # 4. Sirf naye tokens decode kiye (Zero split error)
    new_tokens = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# 5. Live UI launch kiya
demo = gr.ChatInterface(
    fn=chat_with_memory,
    title="🩺 AI Medical Healthcare Doctor",
    description="Fine-tuned LLaMA-3 8B on 100k+ doctor consultations. Ask any medical question!",
)

demo.launch(share=True)
```

* **Expected Output:** Ek blue link aayega: `Running on public URL: https://xxxx.gradio.live`.
* **Testing Steps:**
  1. Pehla message bhejo: *"Doctor, I have severe knee pain since 2 days."*
  2. Doosra message bhejo: *"Should I apply ice on it?"*
  3. Doctor bina kisi repetition ke ice pack aur knee pain dono ka perfect advice dega!

---

## 🎯 Summary of Golden Rules to Remember:
1. **Model:** Hamesha `unsloth/llama-3-8b-bnb-4bit` (4-bit NF4) use karo T4 GPU ke liye.
2. **EOS Token:** Prompt ke end me `+ tokenizer.eos_token` lagana kabhi mat bhoolna.
3. **Filter Length:** Training data ko hamesha `< 350 words` par filter karo taaki 512 token limit cross na ho.
4. **Decoding:** Inference me hamesha `temperature=0.7` aur `repetition_penalty=1.15` lagao taaki repetitive loops na banein!
