# MISInsight-Pro — MIS Insight Chatbot (Streamlit)

MISInsight-Pro is a Streamlit-based MIS analysis chatbot that helps users extract **key insights** and **actionable business strategies** from MIS reports (CSV exported from Excel).  
It enforces **strict scope behavior**, meaning it answers questions **only based on the uploaded report**, and rejects unrelated queries.

---

## 🌐 Live Demo (Deployed)

✅ Try the deployed app here:  
https://pxdarkshadow-mis-insight-chatbo-misinsight-pro-local-app-xqbyyb.streamlit.app/

---

## 🚀 Features

- 📤 Upload MIS reports as **CSV (Excel export)**
- 👀 Dataset preview + summary (shape, columns, datatypes, numeric stats)
- 🧠 Multiple analysis backends:
  - ✅ **Rule-based (Offline)**
  - ✅ **Ollama (Local LLM)**
  - ✅ **Groq Cloud Llama**
- 🎯 Always generates at least **5 actionable strategies**
- 🔒 Strict scope control:
  - Only answers questions related to the uploaded MIS report
  - Blocks unrelated questions with fallback message
- 📥 Download analysis results as:
  - `.txt`
  - `.json`

---

## 🧠 Backends Supported

### 1) Rule-based (Offline)
No internet, no API key needed.

### 2) Ollama (Local LLM)
Requires Ollama running locally:
- Default URL: `http://localhost:11434/api/generate`

### 3) Groq (Cloud Llama)
Fast cloud LLM backend (ideal for deployment).  
Requires Groq API key (see setup below).

---

## 🛠️ Installation (Local Setup)

### 1) Clone the repository
```bash
git clone https://github.com/Pxdarkshadow/MIS_Insight_chatbot.git
cd MIS_Insight_chatbot
````

### 2) Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the app

```bash
streamlit run MISInsight-Pro_local_app.py
```

---

## 🔐 API Key Setup (Groq)

### ✅ Option A (Recommended): Streamlit Secrets

Create:

```
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_KEY"
```

⚠️ IMPORTANT: Never push this file to GitHub.

---

### ✅ Option B: Environment Variable

**Windows (PowerShell)**

```bash
setx GROQ_API_KEY "YOUR_GROQ_KEY"
```

**Mac/Linux**

```bash
export GROQ_API_KEY="YOUR_GROQ_KEY"
```

---

## 🦙 Ollama Setup (Local Llama)

1. Install Ollama
2. Pull a model:

```bash
ollama pull llama3.1
```

3. Run Ollama (it starts the server automatically)

The app will call:

```
http://localhost:11434/api/generate
```

---

## 📄 Input Format

* Upload a `.csv` file exported from Excel MIS reports
* Works best with structured reports containing columns like:

  * Sales / Revenue / Income
  * Cost / Expense / Spend
  * Region / Area / Location

---

## 📌 Strict Scope Policy

The chatbot only answers based on uploaded MIS report data.

If user asks unrelated questions (weather, politics, movies, coding help, etc.), it responds with:

> I can only help with the MIS report you provided.

---

## 📥 Output

The system generates:

* **Insights** (3–8 bullets)
* **Strategies** (minimum 5 actionable strategies)

Results can be downloaded in `.txt` and `.json` formats.

---

## 📌 Author

Created by **Shaun Mathew**
GitHub: [https://github.com/Pxdarkshadow](https://github.com/Pxdarkshadow)

---

## ⭐ Support

If you find this project useful, consider giving it a ⭐ on GitHub!

```
