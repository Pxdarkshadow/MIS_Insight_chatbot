```
# 📊 MIS Report Strategy Chatbot

A lightweight **offline desktop chatbot** that processes **MIS-generated PDF reports** and suggests at least **5 actionable strategies**.  
If any other type of input is uploaded, the app simply responds with **“Not relevant.”**

This project runs fully **offline**, using **rule-based logic** for strategy generation.  
Optionally, you can integrate **Ollama** for smarter, locally hosted NLP-based insights.

---

## ✨ Features
- ✅ Upload MIS-generated **PDF reports**
- ✅ Extract text locally (no cloud dependencies)
- ✅ Generate **5+ rule-based strategies**
- ✅ Responds *“Not relevant”* to invalid inputs
- ✅ Simple, clean desktop UI
- ✅ Works fully **offline**
- ⚡ Optional: **Ollama integration** for advanced insights

---

## 🛠️ Tech Stack
- **Python** (core logic)
- **Tkinter** (desktop UI)
- **PyMuPDF (fitz)** for PDF text extraction
- **Rule-based strategy engine**
- **[Optional] Ollama** for local NLP

---

## 📂 Project Structure
```

mis-chatbot/
│── app.py               # Main app
│── strategies.py        # Rule-based strategies
│── requirements.txt     # Dependencies
│── README.md            # Documentation

````

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mis-chatbot.git
cd mis-chatbot
````

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

---

## 📝 Usage

1. Open the app.
2. Click **Upload MIS Report** and select a PDF.
3. If valid, the app will extract text and generate strategies.
4. If not, it will display *“Not relevant.”*

---

## 🔧 Optional: Ollama Integration

If you want to use **Ollama** for smarter suggestions:

1. Install Ollama: [https://ollama.ai](https://ollama.ai)
2. Pull a model (`llama3.1`):

   ```bash
   ollama pull llama3.1
   ```
3. Modify `app.py` to send extracted text to Ollama and display responses.

---

## 📌 Example Strategies

* Improve data-driven decision making.
* Optimize resource allocation.
* Identify performance trends.
* Enhance reporting accuracy.
* Automate repetitive tasks.

---

## 📦 Build as Executable

You can package the app as a `.exe` (Windows) or `.app` (Mac) with **PyInstaller**:

```bash
pyinstaller --onefile --noconsole app.py
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

## 📜 License

MIT License – Free to use and modify.

```
