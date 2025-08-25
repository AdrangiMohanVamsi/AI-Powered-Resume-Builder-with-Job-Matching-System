# 📝 Resume Builder  

An **AI-powered Resume Builder with Job Matching System** that helps users create professional resumes and intelligently match them with job descriptions using Natural Language Processing (NLP) and similarity scoring.

---

## 🚀 Features  
- 📄 Build professional resumes dynamically.  
- 🤖 AI-powered job matching with similarity scoring.  
- 🔍 Extracts and analyzes resume/job description content.  
- 🧠 Uses embeddings & similarity analysis for best-fit job matching.  
- 🛠️ Built with **Streamlit**, **PyPDF2**, **Sentence Transformers**, and **Scikit-learn**.  

---

## 📂 Project Structure  
```
demo_resume/
│── Home.py                    # Main Streamlit application
│── requirements.txt           # Project dependencies
│── analysis_history.json      # Stores previous analysis results
│── 01 AI-Powered Resume Builder with Job Matching System.pdf  # Documentation
│── README.md                  # Project documentation
│── .gitignore                 # Files ignored in git
│── .git/                      # Git metadata
```

---

## ⚙️ Installation & Setup  

1. Clone the repository:  
   ```bash
   git clone <repo-url>
   cd demo_resume
   ```

2. Create and activate a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate     # Mac/Linux
   venv\Scripts\activate        # Windows
   ```

3. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:  
   ```bash
   streamlit run Home.py
   ```

---

## 🛠️ Tech Stack  
- **Frontend/UI**: Streamlit  
- **Backend/Logic**: Python (NLP + ML libraries)  
- **PDF Processing**: PyPDF2, pdfplumber  
- **Embeddings & Similarity**: Sentence Transformers, Scikit-learn  
- **Data Storage**: JSON (analysis history)  

---

## 📖 Usage  
1. Launch the app using Streamlit.  
2. Upload your **resume (PDF format)**.  
3. Paste or upload a **job description**.  
4. The system calculates a **similarity score** between your resume and the job description.  
5. Download your generated **AI-enhanced resume**.  

---

## 👨‍💻 Author  
Developed by **Adrangi Mohan Vamsi**  
