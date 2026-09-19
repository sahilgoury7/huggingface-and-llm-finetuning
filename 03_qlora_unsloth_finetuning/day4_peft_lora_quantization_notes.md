# 📘 Day 4: PEFT, LoRA Mathematics & 4-Bit Quantization (Step-by-Step Notes)

> **Track:** Gen AI Batch 9 | LLM Fine-Tuning  
> **Instructor:** Ashish Jangra | **Learner:** Sahil Goury  
> **Session:** Phase 4 Deep Dive  

---

# 📖 Part 1: Glossary (Growing As We Learn)

| Term | Simple Meaning | Why It Matters |
| :--- | :--- | :--- |
| **Full Fine-Tuning (FFT)** | Model ke 100% parameters, gradients aur optimizer states ko train karna. | Gigantic memory (16-18 bytes per param) mangta hai; consumer GPUs par crash ho jata hai. |
| **AdamW Optimizer States** | Training ke waqt har parameter ke liye extra 12 bytes memory (FP32 master weight, 1st momentum, 2nd momentum). | Training me 70% VRAM akela AdamW optimizer consume karta hai. |

---

# 💡 Part 2: Important Things (Step-by-Step Concepts)

### Step 1: Why Full Fine-Tuning Fails on Consumer GPUs (The 16-18 Bytes/Param Formula)

#### 1. Inference vs Training VRAM:
* **Inference (FP16):** Model weights ko memory me rakh kar output generate karna.
  * Sirf **2 Bytes** per parameter chahiye.
  * 8B Model = $8 \times 2 = 16\,\text{GB VRAM}$ (Colab T4 par barely fit hota hai).
* **Training (Full Fine-Tuning with AdamW in FP16):**
  * GPU me 4 distinct memory pools baithte hain:
    1. **Model Weights (FP16):** `2 Bytes` / param
    2. **Gradients (FP16):** `2 Bytes` / param (backward pass slopes)
    3. **AdamW Optimizer States (FP32):** `12 Bytes` / param
       - 4 Bytes: Master weights copy in FP32
       - 4 Bytes: 1st Momentum vector (Running mean of gradients)
       - 4 Bytes: 2nd Momentum vector (Running variance of squared gradients)
    4. **Activations & Batch Buffer:** `~2 Bytes` / param
  
$$\mathbf{\text{Total Training VRAM}} = 2 + 2 + 12 + 2 = \mathbf{16 \text{ to } 18 \text{ Bytes per parameter!}}$$

#### 2. The 8B Model Shock:
$$8\,\text{Billion} \times 16\,\text{Bytes} \approx \mathbf{128\,\text{GB VRAM}}$$
* Google Colab T4: **16 GB** $\to$ Immediate `CUDA Out Of Memory`.
* RTX 3090 / 4090: **24 GB** $\to$ Immediate Crash.
* Isliye consumer GPUs par 8B models ko Full Fine-Tune karna mathematically impossible hai.

---

# 🎯 Part 3: Interview Q&A (Spoken Pitch)

### Q1: Why can't we perform Full Fine-Tuning on an 8B model using an RTX 3090 (24GB) or Colab T4 (16GB)?
* **Spoken Pitch (English):**  
  *"Full Fine-Tuning an 8B parameter model requires approximately 16 to 18 bytes of VRAM per parameter when using mixed-precision AdamW. Beyond the 2 bytes for static FP16 weights, we need 2 bytes for gradients, 12 bytes for AdamW optimizer states (FP32 master weights, momentum, and variance), and about 2 bytes for activations. This sums to 128GB+ of VRAM, causing an immediate Out-Of-Memory crash on consumer GPUs like a 16GB T4 or 24GB RTX 3090."*
* **Spoken Pitch (Hinglish):**  
  *"Full Fine-Tuning me sirf model ke weights GPU me nahi aate. AdamW optimizer har parameter ke liye 12 bytes ka state leta hai (FP32 master weights, 1st momentum, 2nd momentum), 2 bytes gradients aur 2 bytes weights milakar total 16-18 bytes per parameter banta hai. 8B model ke liye 128GB+ VRAM chahiye hoti hai, jabki Colab T4 me sirf 16GB hai. Isliye bina PEFT/LoRA ke consumer GPU par training mathematically impossible hai."*
