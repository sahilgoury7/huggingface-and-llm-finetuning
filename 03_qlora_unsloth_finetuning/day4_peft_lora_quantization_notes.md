# 📘 Day 4: PEFT, LoRA & 4-Bit Quantization (Aasan Bhasha Me Notes)

> **Track:** Gen AI Batch 9 | LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Target:** 8B model ko free Colab T4 (16GB) par train karne ka poora funda!  

---

# 📖 Part 1: Glossary (Sabhi Terms Ka Seedha Matlab)

| Term | Ek Line Me Asli Matlab | Asli Zindagi Ka Example |
| :--- | :--- | :--- |
| **Full Fine-Tuning (FFT)** | Model ke 100% dabbo (parameters) ko zabardasti badalna. | 1000-page ki mehengi kitab ke har page par pen chala ke ganda karna. |
| **AdamW Optimizer** | Model ko sikhane wala engine jo har number ke peeche 3 extra copies (12 bytes) yaad rakhne ke liye mangta hai. | Ek aisa munshi jo har 1 hisaab likhne ke liye 3 nayi diary bhar deta hai! |
| **PEFT** | Bade model ko lock (freeze) karke sirf ek chota sa naya hissa (< 1%) train karna. | Poori car badalne ke bajaye sirf steering wheel badalna. |
| **LoRA** | Ek badi matrix ke bajaye do bohot patli pattiyaan (strips $B \times A$) use karna. | 10,000 dabbe bharne ke bajaye sirf 400 dabbe bharna! |
| **Frozen Base Model ($W_0$)** | Original model jiske weights ko hum lock kar dete hain (Gradients = 0). | Original kitab jisko hum chhu bhi nahi rahe, taaki purani English na bhoole. |
| **LoRA Adapter** | Choti si patli layer jo base model ke sath chipakti hai (size: sirf 50MB se 200MB). | Kitab ke upar chipkaya gaya transparent butter paper / sticky note. |
| **Rank ($r$)** | LoRA ki patti (strip) kitni patli ya moti hogi (jaise 8 ya 16). | Butter paper par notes likhne ke liye kitni jagah di gayi hai. |
| **Alpha ($\alpha$)** | LoRA adapter ka Volume Knob! Yeh decide karta hai ki naye notes ki awaaz kitni tez hogi. | Radio ka volume knob (Hamesha Rank ka double rakhte hain: $\alpha = 2 \times r$). |
| **Zero-Initialization** | Shuru me patti $B$ ko bilkul `0` rakhna taaki shuruwat me model koi ulti-seedhi baat na bole. | Butter paper ko shuru me ekdum saaf (blank) rakhna, uspe koi pehle se scribble na ho. |
| **Target Modules** | Model ke kis kamre me adapter chipkana hai. | Kitab ke sabse important chapters (Saare Linear Layers: `q, k, v, o, gate, up, down`). |
| **Merge & Unload** | Training ke baad butter paper ko kitab ke andar permanently jod dena. | Butter paper ke notes ko kitab me print kar dena taaki padhte waqt koi delay (latency) na ho. |
| **Quantization** | Badi precision (16-bit) wale numbers ko compress karke 4-bit banana. | 16 GB ke model ko nichod kar sirf 4 GB ka bana dena! |
| **NF4 (NormalFloat 4)** | 4-bit ka aisa smart tarika jo model ko 4-bit me bhi 16-bit jitna hoshiyar rakhta hai. | Dabbe barabar na banakar wahan zyada dabbe banana jahan asali numbers jama hain. |
| **Double Quantization** | Chote scaling numbers ko bhi dobara chota karna. | Extra 400MB VRAM bachane ka jugad. |
| **Paged Optimizers** | GPU ka Airbag! Jab VRAM bharne lage toh thoda data CPU RAM me bhej deta hai taaki crash na ho. | Car ka airbag jo crash hone se bacha leta hai. |
| **QLoRA** | 4-bit Base Model + LoRA Adapter + Paged Optimizer = Colab T4 par 8B model train karne ki recipe! | Saste laptop/GPU par sabse bada model chalane ka superpower! |
| **BitsAndBytesConfig** | 4-bit quantization ki settings define karne wala remote control. | Compressor machine ki settings set karna. |
| **peft** | Parameter-Efficient Fine-Tuning library jo base model ko freeze karti hai. | Adapter banane aur chipkane wali toolkit. |
| **LoraConfig** | LoRA adapter ke niyam (Rank, Alpha, Target modules) set karna. | Butter paper ka size aur volume knob set karne ka form. |
| **get_peft_model** | Base model ke sath LoRA adapter ko physically jodna/attach karna. | Kitab ke upar butter paper ko physically chipka dena! |

---

# 💡 Part 2: Important Things (Story-by-Story Concepts)

### Kahani 1: Full Fine-Tuning Kyun Fail Hoti Hai? (The 16-18 Bytes Math)

Jab hum kisi **8 Billion (8B)** model ko sirf chalate hain (inference):
* Wo har parameter ke liye **2 Bytes** leta hai $\to 8 \times 2 = \mathbf{16\,\text{GB VRAM}}$ (Colab T4 par fit ho jata hai).

Lekin jab hum **Train** ka button dabate hain, toh GPU me 4 cheezein aati hain:
1. **Model Weights:** `2 Bytes`
2. **Gradients (Slopes):** `2 Bytes`
3. **AdamW Optimizer:** `12 Bytes` (Yeh akela 3 copies rakhta hai!)
4. **Activations:** `~2 Bytes`

$$\text{Total VRAM} = 2 + 2 + 12 + 2 = \mathbf{16 \text{ se } 18 \text{ Bytes per parameter!}}$$

* **8B Model ka Kharcha:** $8\,\text{Billion} \times 16\,\text{Bytes} = \mathbf{128\,\text{GB VRAM}}$!
* Colab T4 ke paas sirf **16 GB** hai $\to$ **Turant Crash!**

---

### Kahani 2: LoRA Kya Hai? (Kitab aur Butter Paper)

LoRA ka kehna hai:
> *"Bhai, 1000-page ki original kitab (Base Model) ko bilkul mat chhedo (FREEZE). Uske upar ek patla sa transparent butter paper (LoRA Adapter) chipka do!"*

* Hum naye notes sirf butter paper par likhenge.
* Original kitab 16 GB ki hai, par butter paper sirf **50 MB se 200 MB** ka hota hai!
* Train sirf **0.5% parameters** hote hain, baaki 99.5% model lock rehta hai.
* Training VRAM 128 GB se gir kar seedha **10-12 GB** par aa jaati hai!

---

### Kahani 3: Do Patli Pattiyaan ($B \times A$) aur Rank ($r$)

Socho ek Excel sheet hai jisme **100 Rows** aur **100 Columns** hain:
* Total dabbe = $100 \times 100 = \mathbf{10{,}000 \text{ numbers}}$.
* Full Fine-Tuning me 10,000 numbers badalne padte the.

**LoRA ne bola: Do patli pattiyaan banao!**
* **Patti A:** Size $2 \times 100 = 200$ numbers
* **Patti B:** Size $100 \times 2 = 200$ numbers
* Dono milakar = sirf **400 numbers**! (10,000 ke mukable sirf 4% mehnat!)

Jab Patti B aur Patti A multiply hoti hain, toh poori $100 \times 100$ ki sheet wapas ban jaati hai!
* Yeh beech ka chota number **`2`** hi **Rank ($r$)** hota hai.
* LLMs me hum $r = 8$ ya $r = 16$ use karte hain.

---

### Kahani 4: Do Zaroori Niyam (Zero-Init aur Alpha)

#### Niyam 1: Patti B ko shuru me ZERO (`0`) kyun rakhte hain?
* Agar butter paper par shuru se hi kisi ne gande pen se scribble kiya hota, toh kitab padhte hi kachra dikhta.
* Patti B ko `0` rakhne se:
  $$\Delta W = B \times A = 0 \times A = \mathbf{0}$$
* Training ke pehle second par model bilkul original base model jaisa rehta hai, koi random kachra nahi bolta. Dheere-dheere wo naye numbers seekhta hai!

#### Niyam 2: Alpha ($\alpha$) — Volume Knob
* Naye notes ki awaaz base model ke mukable kitni tez honi chahiye?
* **Rule:** Hamesha Rank ka double rakho: $\mathbf{\alpha = 2 \times r}$  
  *(Agar $r=8$ hai, toh $\alpha=16$ rakh do).*

---

### Kahani 5: Kahan Chipkana Hai? (Target Modules)

Model ke do main engine hote hain:
1. **Attention:** `q_proj` (Sawaal), `k_proj` (Match), `v_proj` (Meaning), `o_proj` (Result).
2. **MLP / Reasoning:** `gate_proj`, `up_proj`, `down_proj`.

* **Best Practice:** Saare ke saare linear layers (`q, k, v, o, gate, up, down`) par LoRA lagao! Isse model ki accuracy poori 100% Full Fine-Tuning jaisi aati hai, aur memory tab bhi 1% hi lagti hai!

---

### Kahani 6: Merge & Unload (Deployment ke time zero delay)

* Jab training complete ho jaye, toh deployment ke time hum ek line ka code chalate hain:
  ```python
  merged_model = model.merge_and_unload()
  ```
* Yeh butter paper ke numbers ko permanently kitab ke andar jod (add) deta hai.
* Iske baad model normal transformer ban jata hai — ZERO extra delay, aur kisi bhi server (vLLM, Ollama) par super-fast chalta hai!

---

### Kahani 7: 4-Bit Quantization (Bits-to-Bytes Math)

Sabse simple hisaab ($1 \text{ Byte} = 8 \text{ Bits}$):
* **16-bit (FP16):** $16 \div 8 = 2 \text{ Bytes} \implies 8\text{B Model} = \mathbf{16\,\text{GB}}$
* **8-bit (INT8):** $8 \div 8 = 1 \text{ Byte} \implies 8\text{B Model} = \mathbf{8\,\text{GB}}$
* **4-bit (NF4):** $4 \div 8 = 0.5 \text{ Byte (Aadha Byte!)} \implies 8\text{B Model} = \mathbf{4\,\text{GB!}}$

* **Colab T4 (16GB) me kya hota hai?**
  * 4-bit model akela sirf **4 GB** leta hai.
  * Baaki bachi **12 GB VRAM** me LoRA aaram se train hota hai bina kisi crash ke!

---

### Kahani 8: NormalFloat4 (NF4) Kyun Jeeta?

* **Standard INT4 (Pagal hone ka karan):** Yeh saare dabbo ko barabar doori par baantta hai. Lekin model ke 90% numbers `0` ke paas jama hote hain. INT4 ne un sabko mix karke kachra bana diya.
* **NF4 (Smart Tarika):** Isne `0` ke paas bohot saare chote-chote barik dabbe banaye. Har dabbe me barabar information aayi, aur 4-bit model bhi 16-bit model jitna smart raha!

---

### Kahani 9: Paged Optimizer (GPU ka Airbag)

* Jab kabhi lamba sentence aata hai aur GPU VRAM achanak bharne lagti hai, toh Paged Optimizer us extra bojh ko temporary **CPU RAM** me bhej deta hai.
* Jaise hi GPU halka hota hai, wapas le aata hai. Model kabhi `Out of Memory` crash nahi hota!

---

### 💻 Hugging Face Me 2 Line Ka Asli Setup:

```python
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 1. 4-bit NF4 Quantization (Model ko 16GB se 4GB banana)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # Smart 4-bit
    bnb_4bit_compute_dtype=torch.float16,    # Math FP16 me hoga
    bnb_4bit_use_double_quant=True           # Thodi aur VRAM bachegi
)

# 2. LoRA Config (Butter Paper setup)
lora_config = LoraConfig(
    r=16,                                    # Patti ki motai
    lora_alpha=32,                           # Volume knob (2 * r)
    target_modules=[                         # Saare linear chapters
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

# 🎯 Part 3: Interview Q&A (Seedha Muh Se Bolne Wala Format)

### Sawaal 1: 8B model ko normal GPU par Full Fine-Tune kyun nahi kar sakte?
* **Aapka Jawab (Hinglish):**  
  *"Sir, training ke waqt sirf model ke weights nahi aate. AdamW optimizer har parameter ke liye 12 bytes ka extra state leta hai (master weights, momentum 1 aur momentum 2). Gradients aur weights milakar total 16 se 18 bytes per parameter lagta hai. Is hisaab se 8B model ke liye 128GB+ VRAM chahiye hoti hai, jabki Colab T4 me sirf 16GB aur RTX 3090 me 24GB hoti hai. Isliye bina LoRA/QLoRA ke training crash ho jayegi."*
* **Aapka Jawab (English):**  
  *"Full Fine-Tuning an 8B model requires around 16 to 18 bytes of VRAM per parameter with mixed-precision AdamW. Beyond the 2 bytes for FP16 weights, AdamW needs 12 bytes for optimizer states (FP32 master weights, 1st momentum, and 2nd momentum) plus 2 bytes for gradients. For an 8B model, this demands over 128GB VRAM, which immediately crashes a 16GB T4 or 24GB RTX GPU."*

---

### Sawaal 2: LoRA kya hai aur yeh memory kaise bachata hai?
* **Aapka Jawab (Hinglish):**  
  *"LoRA ek PEFT technique hai. Iska basic funda hai ki base model ke 99% weights ko freeze kar do aur sirf do patli low-rank matrices ($B \times A$) ko train karo. Agar layer $4096 \times 4096$ ka hai, toh 1.67 crore numbers train karne ke bajaye hum rank $r=8$ ki do patli pattiyaan train karte hain jisme sirf 65,000 parameters hote hain (99.6% reduction!). Trainable parameters 1% se kam ho jaate hain aur VRAM 128GB se gir kar 10-12GB par aa jaati hai."*
* **Aapka Jawab (English):**  
  *"LoRA decomposes the weight update matrix $\Delta W$ into two low-rank matrices: $\Delta W = B \times A$. Instead of updating a full $4096 \times 4096$ matrix (16.7M parameters), with rank $r=8$ we only train two thin matrices of shape $4096 \times 8$ and $8 \times 4096$, which is just 65K parameters. This reduces trainable parameters by over 99%, bringing VRAM requirements down from 128GB to 10-12GB."*

---

### Sawaal 3: Matrix B ko shuru me ZERO kyun rakhte hain?
* **Aapka Jawab (Hinglish):**  
  *"Kyunki agar B ko zero rakhenge toh $B \times A = 0$ hoga. Iska matlab training ke step 0 par model bilkul original base model jaisa sahi output dega, koi random noise add nahi hogi. Jaise-jaise training aage badhegi, B dheere-dheere naye numbers seekhega."*
* **Aapka Jawab (English):**  
  *"Matrix B is initialized to zero so that at step zero, $\Delta W = B \times A = 0$. This ensures the model starts with the exact pre-trained base model outputs without injecting any initial random noise. Matrix B then smoothly learns the new task during training."*

---

### Sawaal 4: NF4 standard INT4 se behtar kyun hai?
* **Aapka Jawab (Hinglish):**  
  *"Standard INT4 saare dabbo ko barabar doori par rakhta hai, jabki neural network ke 90% weights zero ke paas hote hain (normal distribution). INT4 me zero ke paas wale zaroori weights mix ho jaate hain jisse model pagal ho jata hai. NF4 zero ke paas barik dabbe banata hai taaki har dabbe me barabar information aaye, isliye 4-bit me bhi model 99% accurate rehta hai."*
* **Aapka Jawab (English):**  
  *"Standard INT4 uses uniform linear bins, which fails because neural network weights follow a normal distribution centered at zero. Uniform bins squash the dense weights near zero and waste bits on empty tails. NF4 uses quantile bins where each 4-bit bin has an equal probability of containing weights, maximizing information entropy and preserving 99%+ of 16-bit accuracy."*
