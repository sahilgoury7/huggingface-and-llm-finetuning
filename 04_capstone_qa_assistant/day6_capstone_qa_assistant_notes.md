# 📘 04: Capstone Project — AI Medical Healthcare Assistant (ChatDoctor QA)

> **Track:** Gen AI Batch 9 | LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Project Goal:** LLaMA-3 8B ko ek Specialized Medical Healthcare Assistant me fine-tune karna, multi-turn memory dalna, aur 100% FREE Cloud par live web link ke sath deploy karna!  

---

# 📖 Part 1: Glossary (Sabhi Terms Ka Asli Matlab)

| Term | Ek Line Me Asli Matlab | Asli Zindagi Ka Example |
| :--- | :--- | :--- |
| **Domain-Specific Assistant** | Ek aam general AI ko kisi specific field (jaise Healthcare) ka verified specialist doctor banana. | Aam college pass-out ko hospital ke OPD me baithne wala senior doctor banana. |
| **ChatDoctor Dataset** | 1,12,000 real patient symptoms (`input`) aur verified doctor diagnoses (`output`) ka curated dataset. | Ek bade hospital ki 10 saal ki patient case history aur doctor prescription diary. |
| **Alpaca Formatting** | Patient query ko `### Instruction:`, `### Input:` aur doctor answer ko `### Response:` me wrap karna. | Patient ki parchi ko hospital ke official OPD prescription format me jodna. |
| **EOS Token (`<\|end_of_text\|>`)** | Model ko batane wala "Full Stop" ya "Stop Signal" ki doctor ka jawab yahan khatam ho gaya. | Doctor ka prescription likh kar neeche signature markar pen rakh dena. |
| **4-Bit NF4 Quantization** | 16-bit ke massive model weights ko 4-bit NormalFloat me compress karna (~16GB $\to$ ~5.5GB). | Ek bhaari suitcase ke kapdon ko vacuum bag me compress karke chote bag me fit karna. |
| **LoRA Decomposition ($B \times A$)** | Original 8B weights ko freeze karke sirf do choti matrices ($16 \times d$) train karna. | Poori moti textbook rewrite karne ke bajaye margin me chote sticky notes chipkana. |
| **Trainable Ratio (0.52%)** | 8.07 Billion parameters me se sirf 4.19 Crore parameters train hona (< 1%). | 800-page ki encyclopedia me se sirf 4 page par correction pencil chalana. |
| **Response-Only Loss** | Loss sirf doctor ke diye gaye advice par nikalna, patient ke sawaal par `-100` kaali parchi lagana. | Medical exam me examiner sirf student ke likhe answer par marks deta hai, question paper copy karne par nahi. |
| **Stateless LLM** | LLM ke paas koi internal hard disk ya memory nahi hoti; har naye run par wo sab bhool jata hai. | *Ghajini* ka Aamir Khan ya *Finding Nemo* ki Dory — jise har 15 minute me sab naya lagta hai! |
| **History Buffer (Chat Memory)** | Purani chat history ko naye sawaal ke aage append karke model ko bhejna taaki usse context yaad rahe. | Patient ki purani OPD file doctor ke samne rakhna taaki doctor ko pichli visit yaad aa jaye. |
| **Temperature ($0.7$)** | Model ke words chunne me thodi creativity aur natural flow lana (taaki boring ya repetitive na ho). | Khane me thoda sa namak-mirch dalna taaki swad natural aur mazedaar aaye. |
| **Repetition Penalty ($1.15$)** | Model ko ek hi sentence ya word baar-baar ghumane (loop me phansne) se rokna. | Stage speaker ko ek hi line 10 baar bolne par tok kar aage badhana. |
| **Gradio Live Share (`share=True`)** | Colab ke GPU model ko ek public URL (`https://xxxx.gradio.live`) par live web app bana dena. | Hospital ke bahar ek live digital help-desk laga dena jise koi bhi phone se access kar sake. |

---

# 💡 Part 2: Important Things (Story-by-Story Concepts)

### Kahani 1: Doctor Ki Parchi Aur Alpaca Template (Universal 3-Pillars Formula)

#### 1. AI Model Ko Direct CSV Kyun Samajh Nahi Aati?
Agar aap LLaMA-3 ke aage direct Excel sheet ya Pandas DataFrame fek doge, toh wo confuse ho jayega! AI model koi database software nahi hai, wo ek **Language Model** hai jise ek lamba, structured text (string) padhna pasand hai.

#### 2. The 3-Pillars Formula (Har Project Ke Liye):
Har instruction prompt template me 3 pillars hote hain:
1. **Instruction (Role):** Model ko uska role aur persona batana.
   - Medical: *"You are an expert medical doctor. Provide accurate and empathetic medical advice."*
   - SQL: *"You are an expert SQL engineer. Write valid SQL queries for the given schema."*
   - Support: *"You are a polite Swiggy support executive. Resolve customer complaints."*
2. **Input (Context):** User ne jo bataya ya pucha (Symptoms, bug code, question).
3. **Response (Target Output):** Jo model ko seekhna aur bolna hai (Prescription, fixed code, answer).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ ### Instruction:                                                        │
│ You are an expert medical doctor...                                     │
│                                                                         │
│ ### Input:                                                              │
│ {patient_symptoms}                                                      │
│                                                                         │
│ ### Response:                                                           │
│ {doctor_advice} <|end_of_text|>                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Kahani 2: The EOS Token Secret (`<|end_of_text|>`)

#### 1. Agar Stop Token Na Lagayein Toh Kya Hoga?
Sochiye ek doctor ne patient ko bola: *"Aap Paracetamol le lijiye."*  
Aur uske baad doctor chup hi nahi ho raha! Wo bolta ja raha hai: *"Aur suno, kal mausam accha hoga, mere chacha ke bete ki shaadi hai, hospital ki deewar peeli hai..."* 🤦‍♂️

Isi ko AI me **Endless Babbling / Hallucination Loop** kehte hain!

#### 2. Solution: `tokenizer.eos_token`
Har model ka apna ek secret Stop Word hota hai:
* **LLaMA-3:** `<|end_of_text|>` ya `<|eot_id|>`
* **Mistral:** `</s>`

Jab hum formatting function me har doctor answer ke aakhir me `+ tokenizer.eos_token` lagate hain:
```python
text = alpaca_prompt.format(patient, doctor) + tokenizer.eos_token
```
Toh model training ke dauran seekh leta hai:  
👉 **"Jaise hi mera medical advice complete ho, mujhe turant yeh token produce karke CHUP ho jana hai!"**

---

### Kahani 3: The 772 vs 512 Token Crisis (Live Colab Debugging Story)

#### 1. Error Kyun Aaya Tha?
Training ke Step 1 ke baad hamara Colab achanak crash hua:
> `Unsloth: Input IDs shape [2, 772] with length 772 > max sequence length of 512. We shall truncate it ourselves.`  
> `ValueError: Expected input batch_size (1024) to match target batch_size (1544).`

#### 2. Iske Peeche Ka Asli Drama:
- Humne model ki limit rakhi thi **512 tokens**.
- Dataset me ek patient ne apni 10 saal purani poori kahani likh di thi, jisse total length **772 tokens** ho gayi!
- Unsloth ne emergency me input ko 512 par kaat diya ($2 \times 512 = 1024$).
- Lekin target labels ko kaatna bhool gaya ($2 \times 772 = 1544$).
- PyTorch Dynamo compiler ne dekha ki Input aur Target ka size match nahi kar raha ($1024 \neq 1544$), aur usne fatal error fek diya!

#### 3. Hamara Clean 2-Line Fix:
Humne dataset ko train karne se pehle filter kar diya:
```python
train_dataset = train_dataset.filter(lambda x: len(x["text"].split()) < 350)
```
- 2,000 me se **1,924 high-quality samples** bach gaye (saare strictly < 512 tokens).
- Iske baad training bina kisi rukawat ke rocket ki tarah 60 steps tak complete hui!

---

### Kahani 4: 8 Billion Parameters vs 0.52% LoRA Math

#### 1. Free Colab T4 GPU Par 8B Model Kaise Chala?
- **Full Model Size:** LLaMA-3 me **8,072,204,288 parameters** hote hain.
- FP16 me isko train karne ke liye **~64 GB VRAM** chahiye (jo $10,000 ke A100 GPU me hi hoti hai).
- Hamare paas Google Colab ka free **Tesla T4 GPU (14.56 GB VRAM)** tha.

#### 2. The LoRA + NF4 Miracle:
1. **4-Bit NF4 Quantization:** 16GB ke base model ko compress karke **~5.5 GB VRAM** par le aaye.
2. **LoRA Rank $r=16$:** Original 8 Billion parameters ko freeze kar diya (Unhe chhua tak nahi!).
3. Sirf 7 linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) par chote adapter matrices lagaye.
4. **Result:** Sirf **41,943,040 parameters (0.52%)** train hue!
5. **Loss Curve:** Training loss **2.876** se smoothly gir kar **1.891** par aa gaya!

---

### Kahani 5: The Stateless LLM & The Chatbot Memory Secret

#### 1. Model Ke Paas Memory Kyun Nahi Hoti?
LLMs pure mathematical functions hote hain: $y = f(x)$.  
Jab aapne pucha: *"Doctor, I have severe knee pain."*, model ne answer diya.  
Lekin jab aapne agla sawaal pucha: *"Should I apply ice on it?"*, toh model ke liye aap ek bilkul naye insaan ho! Usse pichli baat ka 0% pata hota hai.

#### 2. ChatGPT Memory Kaise Banata Hai? (The Context Buffer):
ChatGPT koi magic memory chip use nahi karta. Wo naye sawaal ke upar **purani chat history ka thappa laga kar** model ko bhejta hai:

```text
Prompt Sent to Model on Turn 2:
### Input:
Patient: Doctor, I have severe knee pain since 2 days.
Doctor: Consult an orthopedic surgeon, take rest, avoid heavy lifting.
Patient: Should I apply ice on it?

### Response:
[Doctor writes answer with full context!]
```

Humne hamare Gradio `chat_with_memory` function me yahi context buffer implement kiya!

---

### Kahani 6: The "Chair Loop" Mystery & The Two Magic Knobs

#### 1. Model Ek Hi Line Kyun Repeat Kar Raha Tha?
Shuru me model ne knee pain ke jawab me yeh line 10 baar bol di:
> *"avoid sitting on a chair with no back support... avoid sitting on a chair with low back rest... avoid sitting on a chair with high back rest..."* 🪑

Yeh isliye hua kyunki model **Greedy Decoding (Temperature = 0)** par chal raha tha. Jab model ko lagta hai ki "chair" sabse safe word hai, toh wo usi ke loop me phans jata hai.

#### 2. The 2 Magic Knobs:
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,         # 1. Natural flow aur empathy deta hai
    repetition_penalty=1.15, # 2. Ek hi phrase repeat karne par bhaari penalty lagata hai!
    use_cache=True
)
```

**Result Dekhiye:**  
Model ne turant chair loop band kar diya aur seedha point-to-point clinic advice di:
> *"First thing to do is rest... Ice pack should be applied at regular intervals of time... Hope this helps you."* 🧊✅

---

# 💻 Part 3: Complete Capstone Production Blueprint

Yeh hai hamara end-to-end production script jo humne Colab me step-by-step execute kiya:

```python
# ==============================================================================
# 🩺 END-TO-END MEDICAL HEALTHCARE ASSISTANT (LLaMA-3 8B + 4-bit QLoRA + GRADIO)
# ==============================================================================

# Step 1: 4-Bit LLaMA-3 aur LoRA Adapters Load Karna
from unsloth import FastLanguageModel
import torch

model_id = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# Step 2: Dataset Load, Train/Test Split & Filter (< 350 words)
from datasets import load_dataset

dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
split_data = dataset.train_test_split(train_size=2000, test_size=200, seed=42)
train_dataset = split_data["train"]

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert medical doctor. Provide accurate and empathetic medical advice based on the patient's symptoms.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_func(examples):
    texts = [alpaca_prompt.format(p, d) + EOS_TOKEN for p, d in zip(examples["input"], examples["output"])]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_func, batched=True)
train_dataset = train_dataset.filter(lambda x: len(x["text"].split()) < 350)

# Step 3: SFTTrainer Setup & Training Execution
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=1,
    seed=3407,
)

FastLanguageModel.for_training(model)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
    args=training_args,
)

trainer.train()

# Step 4: LoRA Weights Save Karna Locally
model.save_pretrained("medical_llama3_lora")
tokenizer.save_pretrained("medical_llama3_lora")

# Step 5: Multi-Turn Gradio Web Interface (With Memory & Anti-Repetition)
import gradio as gr

FastLanguageModel.for_inference(model)

def chat_with_memory(message, history):
    context = ""
    for item in history:
        if isinstance(item, dict):
            role = "Patient" if item.get("role") == "user" else "Doctor"
            context += f"{role}: {item.get('content', '')}\n"
        elif isinstance(item, (list, tuple)):
            context += f"Patient: {item[0]}\nDoctor: {item[1]}\n"
            
    full_prompt = alpaca_prompt.format(f"{context}Patient: {message}", "")
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.15,
        use_cache=True
    )
    new_tokens = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

demo = gr.ChatInterface(
    fn=chat_with_memory,
    title="🩺 AI Medical Doctor (With Memory)",
    description="Fine-tuned LLaMA-3 8B with 4-bit QLoRA on 100k+ doctor consultations."
)
demo.launch(share=True)
```

---

# 🎯 Part 4: Top 5 Interview Questions & Answers

### Q1: "Aapne 8 Billion parameter ke LLaMA-3 model ko Colab ke free 15GB T4 GPU par bina OOM crash ke kaise train kiya?"
* **Candidate Answer (Hinglish):**  
  *"Sir, 8B model ko FP16 me train karne ke liye 64GB+ VRAM chahiye hoti hai. Maine 4-bit NormalFloat (NF4) quantization use kiya through `bitsandbytes`, jisse base model weights ~5.5GB me load ho gaye. Training ke liye maine QLoRA use kiya with Rank $r=16$ aur Alpha $\alpha=16$ on all linear projection layers (`q, k, v, o, gate, up, down`). Isse hamare trainable parameters 8 Billion se ghat kar sirf 41.9 Million (0.52%) reh gaye. Sath hi gradient accumulation steps=4 aur batch size=2 rakhne se peak VRAM 10GB ke andar rahi aur training 10 minute me smoothly complete ho gayi."*
* **English:**  
  *"We leveraged 4-bit NormalFloat (NF4) quantization via bitsandbytes, reducing base model memory to ~5.5GB. We applied QLoRA with rank $r=16$ and $\alpha=16$ across all linear projection layers, reducing trainable parameters to just 0.52% (41.9M out of 8B). Combined with a micro-batch size of 2 and gradient accumulation steps of 4, peak VRAM stayed well below Colab's 14.5GB limit."*
* **💡 Golden Tip:** Hamesha **0.52%** aur **41.9M parameters** ka exact number quote kijiye — interviewer turant impress hota hai!

---

### Q2: "Prompt formatting ke waqt `tokenizer.eos_token` append karna kyun mandatory hai?"
* **Candidate Answer (Hinglish):**  
  *"Sir, LLM naturally ek text generation engine hai. Agar hum training prompts ke end me EOS token (jaise LLaMA-3 ka `<|end_of_text|>`) nahi lagayenge, toh model ko pata hi nahi chalega ki doctor ka answer kahan khatam hua. Inference ke time model answer complete hone ke baad bhi endless hallucination aur irrelevant sentences generate karta rahega. EOS token model ko sikhata hai ki jawab complete hote hi generation stop karni hai."*
* **English:**  
  *"The EOS token acts as the model's explicit stopping boundary. Without appending `tokenizer.eos_token` at the end of each training sample, the model never learns when a clinical response concludes, resulting in endless babbling and hallucinations during inference."*

---

### Q3: "Single-turn QA model ko multi-turn conversational chatbot kaise banaya ja sakta hai jabki LLMs stateless hote hain?"
* **Candidate Answer (Hinglish):**  
  *"Sir, LLMs naturally stateless hote hain aur unke paas koi physical memory nahi hoti. Multi-turn conversation banane ke liye hum **Chat History Buffer** use karte hain. Har naye turn par hum purane turns ke User aur Assistant messages ko format karke naye prompt ke aage append kar dete hain. Jab model full conversation context dekhta hai, toh wo follow-up questions (jaise 'Should I apply ice on it') ko pichle context (knee pain) ke sath correlate karke accurate answer deta hai."*
* **English:**  
  *"Since LLMs are inherently stateless, conversational memory is achieved by appending historical user and assistant turns into the current prompt's context window. This allows the model to resolve coreferences and follow-up queries seamlessly based on previous dialogue state."*

---

### Q4: "Training ke dauran `cross_entropy` loss me input batch size aur target batch size mismatch error kyun aaya tha, aur aapne use kaise fix kiya?"
* **Candidate Answer (Hinglish):**  
  *"Sir, humne `max_seq_length = 512` set kiya tha. Dataset me kuch samples 770+ tokens ke the. Unsloth ne on-the-fly inputs ko 512 par truncate kiya lekin target labels untruncated reh gaye, jisse PyTorch Dynamo compiler me tensor shape mismatch ($1024 \neq 1544$) ho gaya. Humne dataset ko pre-filter kiya where `len(text.split()) < 350`, jisse saare 1,924 samples strictly 512 tokens ke andar aa gaye aur batch mismatch 100% eliminate ho gaya."*
* **English:**  
  *"The mismatch occurred when sequences exceeding 512 tokens were dynamically truncated on the input dimension while target labels remained at 772 tokens, triggering a shape mismatch in PyTorch Dynamo's fused cross-entropy kernel. We resolved it by pre-filtering samples to under 350 words, ensuring every training sequence strictly satisfied the 512-token constraint."*

---

### Q5: "Model generation ke dauran repetitive n-gram loops (jaise baar-baar chair advice repeat hona) ko kaise roka gaya?"
* **Candidate Answer (Hinglish):**  
  *"Sir, greedy decoding (temperature=0) me model highest-probability tokens ke loop me phans jata hai. Humne do decoding hyperparameters tune kiye: `temperature=0.7` (jisse token selection me natural diversity aayi) aur `repetition_penalty=1.15` (jo already generated tokens ki probability par penalty lagata hai). Isse repetitive loops 100% khatam ho gaye aur model ne concise, point-to-point advice di."*
* **English:**  
  *"Greedy decoding often suffers from degenerative repetition loops. We mitigated this by setting `temperature=0.7` to inject controlled stochasticity and introducing a `repetition_penalty=1.15`, which discounts the logits of previously generated tokens to enforce concise, non-redundant completions."*

---

# 🎙️ Part 5: Interview Spoken Pitches (Resume / Portfolio)

### 🗣️ Hinglish Pitch (Networking / Interview intro):
*"Maine Meta ke LLaMA-3 8B model ko ek Specialized Medical Healthcare Assistant ke roop me fine-tune kiya using ChatDoctor dataset (100k+ doctor-patient interactions). Free Google Colab T4 GPU par training ko feasible banane ke liye maine 4-bit NF4 Quantization aur Unsloth GPU acceleration ka use kiya, jisse trainable parameters 8 Billion se ghat kar sirf 41.9 Million (0.52%) reh gaye aur training loss 2.87 se 1.89 tak converge hua. Model ko Stanford Alpaca format me SFTTrainer se train kiya gaya with response-only loss masking. Inference ke liye maine context-buffer memory aur anti-repetition decoding (`temperature=0.7`, `repetition_penalty=1.15`) lagayi, aur Gradio ke zariye ek live public web interface deploy kiya jo complex medical queries aur follow-ups ko human doctor jaisi empathy ke sath handle karta hai."*

### 🗣️ English Pitch:
*"I fine-tuned Meta's LLaMA-3 8B into a domain-specific Medical Healthcare Assistant using the ChatDoctor dataset comprising over 100k real clinical consultations. To train efficiently on a single consumer-grade T4 GPU (16GB VRAM), I implemented 4-bit NF4 Quantization and QLoRA via Unsloth, reducing trainable parameters to just 0.52% (41.9M). The model was trained using TRL's SFTTrainer with Alpaca prompt formatting and response-only loss masking, achieving a loss reduction from 2.87 to 1.89 in under 10 minutes. For production deployment, I built a multi-turn Gradio interface featuring conversational context buffers and anti-repetition decoding penalties, delivering empathetic, clinically grounded medical guidance."*
