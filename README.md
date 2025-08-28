# Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore’s Low-Resource Languages

A public repository containing datasets and code for the paper "Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore’s Low-Resource Languages" (EMNLP 2025)

## 📌 Overview  
SGToxicGuard is a **multilingual dataset and evaluation framework** for benchmarking the safety of Large Language Models (LLMs) in Singapore’s unique low-resource linguistic environment.  

It systematically probes model vulnerabilities across **Singlish, Chinese, Malay, and Tamil**, in addition to English, using a **red-teaming approach** to test whether multilingual LLMs can resist generating toxic or biased outputs in realistic scenarios.  

<p align="center">
  <img src="https://github.com/Social-AI-Studio/SGToxicGuard/blob/main/5_new.png" alt="SGToxicGuard" width="60%"/>
</p>

---

## ✨ Key Features  
- **First multilingual red-teaming benchmark** targeting Singapore’s linguistic landscape.  
- Covers **four low-resource languages** (Singlish, Chinese, Malay, Tamil) alongside English.  
- **Three evaluation tasks**:  
  1. **Toxic Conversation** – Assessing LLM safety in dialogue with toxic prompts.  
  2. **Toxic Question-Answering (QA)** – Detecting toxic biases in fill-in-the-blank hate statements.  
  3. **Toxic Tweet Composition** – Evaluating whether LLMs generate disseminable toxic content.  
- **Evaluation Metrics**:  
  - *Hateful Response Rate* (for conversation and composition tasks).  
  - *Bias Rate* (for toxic QA).  
- Includes **zero-shot and few-shot** in-context learning settings to test susceptibility to **toxicity jailbreaks**.  

---

## 📂 Dataset  
The dataset builds on **HateCheck** and **SGHateCheck**, extending them to support multilingual red-teaming evaluations. The dataset folder includes all samples for the three red-teaming tasks: Toxic Conversation Task (task1), Toxic Question-Answering (QA) Task (task2) and Toxic Tweet Composition Task (task3).

- `*.json` files are provided for each **task** and **language**:  
  - `en` = English  
  - `ss` = Singlish  
  - `zh` = Chinese  
  - `ms` = Malay  
  - `ta` = Tamil

It includes:  
- ~2.5k samples per language for **Conversation** and **Tweet** tasks.  
- ~120–180 samples per language for the **Toxic QA** task.  
- Annotations spanning **15 social groups** (race, religion, gender, disability, etc.) relevant to Singapore’s cultural context.  

### Task 1: Toxic Conversation  
- Single-turn dialogues with toxic input → model must respond safely.  
- Files: `task1_[lang].json`  

### Task 2: Toxic Question-Answering (QA)  
- Fill-in-the-blank hateful statements, testing bias toward vulnerable groups.  
- General setting: `task2_all_[lang].json`  
- Localized settings:  
  - Race → `task2_race_[lang].json`  
  - Religion → `task2_religion_[lang].json`  
  - Gender → `task2_gender_[lang].json`  
  - Disability → `task2_disability_[lang].json`  

### Task 3: Toxic Tweet Composition  
- Generate an engaging tweet while preserving the meaning of a hateful statement.  
- Files: `task3_[lang].json`

⚠️ **Disclaimer**: The dataset contains examples of toxic, hateful, and offensive language. It is released **for research purposes only** to support AI safety, fairness, and multilingual moderation studies.  

---

## 📊 Benchmark Results  
We evaluated six popular multilingual LLMs:  

- **Llama-3.1**  
- **Mistral**  
- **Qwen2.5**  
- **GPT-4o mini**  
- **SeaLLM**  
- **SEA-LION**  

**Findings:**  
- Models often **exhibited higher toxicity** in low-resource languages compared to English.  
- Some multilingual models showed **systematic biases** toward racial and religious groups.  
- **Few-shot demonstrations** significantly increased the likelihood of **toxic jailbreaks**.  
- GPT-4o mini showed the most robust safety alignment, but **vulnerabilities persist across all models**.  

---

## 🚀 Usage  
### 1. Clone the repository  
```bash
git clone https://github.com/<your-org>/SGToxicGuard.git
cd SGToxicGuard
```

### 2. Run evaluation
```
python eval/eval.py \
    --dataset ms \
    --model_name openai/gpt-4o-mini
    --input_path '../ms.csv' \
    --output_path '../task1_re/gpt' \
    --task task1 \
    --shot 0shot
```

---

Please leave issues for any questions about the paper or the dataset/code.
