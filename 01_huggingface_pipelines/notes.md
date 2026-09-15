# 01: Hugging Face Basics & Pipelines

> **Session 09 | Gen AI Batch 9**  
> **Instructor:** Ashish Jangra  
> **Learner:** Sahil Goury  

---

# 📖 Part 1: Glossary (Key Terms & Definitions)

| Term | Simple Meaning | Why It Matters |
| :--- | :--- | :--- |
| **Hugging Face** | AI ka open-source ecosystem (GitHub + Linux of AI). | Yahan 10 Lakh+ models, datasets aur web spaces free milte hain. |
| **Transformers** | Hugging Face ki core Python library. | PyTorch/TensorFlow ke complex code ko 1-line simple interface deti hai. |
| **Model Card** | Model ka 'Aadhaar Card' ya Product Manual. | Model ka size, architecture, limitations, training data aur license batata hai. |
| **Inference API** | Hugging Face ka cloud-based serverless test system. | Model ko bina laptop me download kiye browser ya code se test karne deta hai. |
| **pipeline()** | Transformers library ka sabse high-level function. | 1 line code me text preprocessing, model prediction, aur postprocessing karta hai. |
| **Tokenizer** | Text ko numbers/vectors me todne wala tool. | AI models English nahi samajhte, sirf numbers (Tokens) samajhte hain. |
| **Sub-words** | Bado shabdon ke chote tukde (e.g. Bangalore -> ['Bang', '##alore']). | Vocabulary size ko chota aur spelling mistakes ko handle karne ke liye use hota hai. |
| **input_ids** | Har token ka unique integer ID (e.g. [101, 2054, 2003]). | Model isi array ko input ke roop me leta hai. |
| **attention_mask** | 0 aur 1 ki list (1 = asli word, 0 = padding/khali jagah). | Model ko batata hai ki kin tokens par dhyan dena hai aur kise ignore karna hai. |
| **Logits** | Model ke raw, unnormalized mathematical scores. | Model ka kacha output hota hai, jise baad me probabilities me badla jaata hai. |
| **Softmax** | Ek mathematical function. | Raw Logits ko 0%% se 100%% (0.0 to 1.0) ki probabilities me convert karta hai. |
| **torch.argmax()** | Array me sabse badi value ka Index (Position) nikalne wala function. | QA me answer kahan se shuru aur kahan khatam hua, uska exact token index nikalta hai. |
| **AutoClasses** | AutoTokenizer, AutoModel family classes. | Model name dekh kar backend architecture khud configure karti hain; Fine-tuning ka base hain. |
| **aggregation_strategy='simple'** | NER pipeline ka ek parameter. | Toote hue sub-words ko jodkar poora single word banata hai. |
| **Extractive QA** | Context paragraph me se exact answer dhoondhna. | Zero hallucination hoti hai; answer wahi milega jo paragraph me likha hai. |
| **Abstractive Summarization** | Bada article padhkar apne naye words me summary banana. | Insaan ki tarah paraphrasing karke short concise points generate karta hai. |
| **Beam Search (num_beams)** | Text generation ke time multiple words path explore karna. | Greedy search se behtar quality aur natural text output produce karta hai. |
| **Zero-Shot Classification** | Bina kisi model training ke text ko custom labels me classify karna. | Natural Language Inference (NLI) se test karta hai ki text aur label ka rishta kitna sach hai. |
| **Apache 2.0 / MIT** | Open-source software licenses. | Commercial products aur startups me use karne ke liye 100%% free hote hain. |
| **CC-BY-NC** | Creative Commons Non-Commercial License. | Commercial use strictly band hai; sirf research/learning ke liye allowed hai. |

---

# 💡 Part 2: Important Things (Rules, Formulas & Practical Gotchas)

### 1. The 3-Stage Pipeline Lifecycle (Under the Hood)
Jab aap classifier('text') call karte hain:

```text
Raw Text ──▶ 1. Tokenizer (input_ids + attention_mask) ──▶ 2. Model Forward Pass (Raw Logits) ──▶ 3. Post-Processing (Softmax) ──▶ Label + Score
```

---

### 2. pipeline() vs AutoTokenizer + AutoModel (Golden Rule)

| Feature | pipeline() | AutoTokenizer + AutoModel |
| :--- | :--- | :--- |
| **Analogy** | Automatic Car | Manual / Tuned Car |
| **Code Size** | Sirf 1-2 lines | 6-10 lines |
| **Control** | Bahut kam | Full control (Tensors, Logits, Hidden layers) |
| **Fine-Tuning / Training** | Nahi ho sakti | **100%% Mandatory for Fine-Tuning** |
| **Kab use karein?** | Quick demo, hackathon, simple prediction | Custom LLM training, LoRA/QLoRA, production backend |

---

### 3. 5-Step Model Selection Formula (10 Lakh+ models me se kaise chunein?)

1. **Step 1: Task Filter** - Left sidebar me apna task chuno (`text-classification`, `text-generation`, etc.).
2. **Step 2: Downloads & Likes** - 500k+ downloads wale models battle-tested aur stable hote hain.
3. **Step 3: Hardware & VRAM Check** - `<500M` CPU par chalega, `1B-8B` Colab 16GB T4 GPU par, `70B+` multi-GPU cloud par.
4. **Step 4: Base Model vs Instruct Model** - Fine-tuning ke liye Base model; Chat/QA ke liye Instruct model.
5. **Step 5: License Check** - Commercial startup ke liye Apache 2.0 / MIT.

---

### 4. Core Tasks Syntax Cheatsheet

#### A. Sentiment Analysis (Batching ke saath)
`python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
results = classifier(['I love learning!', 'This is frustrating.'])
# Output: [{'label': 'POSITIVE', 'score': 0.99}, {'label': 'NEGATIVE', 'score': 0.98}]
`

#### B. NER (Subwords fix ke saath)
`python
ner = pipeline('ner', aggregation_strategy='simple')
entities = ner('My name is Sahil, I work at Google in Bangalore.')
# aggregation_strategy='simple' ensures 'Bangalore' doesn't split into 'Bang' and '##alore'
`

#### C. Question Answering (AutoClasses ke saath)
`python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-cased-distilled-squad')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert/distilbert-base-cased-distilled-squad')

inputs = tokenizer('Who founded HF?', 'Hugging Face was founded by Clement in NYC.', return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

start = torch.argmax(outputs.start_logits)
end = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs['input_ids'][0][start:end])
# Output: 'Clement'
`

#### D. Summarization (Seq2Seq ke saath)
`python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')

inputs = tokenizer(article, return_tensors='pt', max_length=1024, truncation=True)
summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=20, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
`

#### E. Zero-Shot Classification (NLI ke saath)
`python
classifier = pipeline('zero-shot-classification')
result = classifier('who is Ai Engineer', candidate_labels=['ai', 'dsa', 'cyber Security'])
# Output: 'ai' scores ~75%% because of Premise-Hypothesis NLI logic!
`

---

### 5. Practical Gotchas & Errors Solved

1. **TypeError: missing 1 required argument: 'image':**
   - Reason: document-question-answering scanned PDF/bills (LayoutLM) ke liye multimodal pipeline hai.
   - Fix: Plain text ke liye question-answering ya AutoModelForQuestionAnswering use karein.
2. **KeyError: Unknown task summarization / question-answering:**
   - Reason: Modern transformers (v4.49+ Python 3.13) me task strings ke bajaye auto-detect ya AutoModelForSeq2SeqLM standard hai.
   - Fix: Direct pipeline(model='...') ya AutoModel classes use karein.
3. **with torch.no_grad():**
   - Inference ke dauran PyTorch ko gradient calculate karne se rokta hai, jisse memory 50%% bachti hai aur speed fast ho jaati hai.

---

# 🎯 Part 3: Q&A (Interview & Concept Mastery)

### Q1: pipeline() aur AutoTokenizer + AutoModel me kya antar hai? Fine-Tuning me kaunsa use hota hai?
**Answer:**  
- pipeline() ek high-level wrapper hai jo quick testing aur inference ke liye bana hai. Isme internal tensors aur loss function ka control nahi milta.
- AutoTokenizer + AutoModel low-level core classes hain jo input IDs, attention masks, logits, aur gradients ka pura control deti hain.
- **Fine-Tuning ke liye hamesha AutoTokenizer + AutoModel use hota hai.**

---

### Q2: NER pipeline me aggregation_strategy='simple' lagana kyun zaroori hai?
**Answer:**  
Transformer tokenizers words ko sub-words me todte hain (e.g. Bangalore -> ['Bang', '##alore']). Agar hum ggregation_strategy='simple' nahi lagayenge, toh model aadhe-aadhe tukdo ko alag entity dikhayega. Yeh parameter un tukdon ko jodkar complete word banata hai.

---

### Q3: torch.max() aur torch.argmax() me kya difference hai? Question Answering me kaunsa use hota hai?
**Answer:**  
- 	orch.max() array me sabse **badi value (Score)** return karta hai.
- 	orch.argmax() sabse badi value ka **Index / Position** return karta hai.
- Question Answering me hume answer ke start aur end token ki position chahiye hoti hai, isliye hum 	orch.argmax(outputs.start_logits) aur 	orch.argmax(outputs.end_logits) use karte hain.

---

### Q4: Zero-Shot Classification bina kisi training data ke kaise kaam karta hai?
**Answer:**  
Yeh backend par **Natural Language Inference (NLI)** use karta hai. 
- Input sentence ko **Premise** maana jaata hai: 'who is Ai Engineer'.
- Har label ka sentence banakar use **Hypothesis** maana jaata hai: 'This text is about ai'.
- Model check karta hai ki kya Premise Hypothesis ko support (Entailment) karta hai. Jiska support score sabse zyada hota hai, use top rank milti hai.

---

### Q5: Extractive QA aur Generative QA me kya fark hai?
**Answer:**  
- **Extractive QA:** Model context paragraph me se hi exact word/sentence span nikalta hai (Zero Hallucination).
- **Generative QA (e.g. ChatGPT):** Model context padhkar naye words generate karta hai, jisme hallucination ka risk hota hai.

---

### Q6: Startup ya Commercial product me CC-BY-NC license wala model kyun nahi use kar sakte?
**Answer:**  
NC ka matlab hota hai **Non-Commercial**. Yeh legal violation hoga. Commercial products ke liye hamesha **Apache 2.0**, **MIT**, ya commercial-friendly open licenses (jaise Llama 3 Community License) wale models hi choose karne chahiye.

---

### Q7: Logits kya hote hain aur unhe confidence percentage me kaise badla jaata hai?
**Answer:**  
Logits transformer neural network ke aakhri layer ke raw unnormalized scores hote hain (jo positive ya negative koi bhi number ho sakte hain). Unpar **Softmax** mathematical function apply karke 0.0 se 1.0 (0% to 100%) ki probabilities banayi jaati hain.

---

### Q8: Inference run karte waqt with torch.no_grad(): kyun lagate hain?
**Answer:**  
Model inference (sirf prediction) ke dauran hume weights update nahi karne hote. `torch.no_grad()` PyTorch ko memory me computational graph aur gradients store karne se rok deta hai, jisse RAM/VRAM save hoti hai aur prediction fast chalta hai.
