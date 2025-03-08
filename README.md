# EasyStep-Math 🪄😊

**EasyStep-Math** is an interactive, enterprise-grade mathematics chatbot designed to solve math problems, generate dynamic visualizations, and even extract questions from images! Enjoy a friendly, step-by-step learning experience for everything from basic algebra to advanced calculus. 

---

## 📬 Contact & Profile Links

- **LinkedIn:** [Nandesh Kalashetti](https://www.linkedin.com/in/nandesh-kalashetti-333a78250)  
- **Portfolio/Resume:** [My Portfolio](https://nandesh-kalashettiportfilio2386.netlify.app)  
- **Email:** [nandeshkalshetti1@gmail.com](mailto:nandeshkalshetti1@gmail.com)  
- **Phone:** 9420732657  

---

## 🌟 Overview

**EasyStep-Math** provides:
- **Step-by-step** math solutions with LaTeX formatting and a friendly, instructive tone.
- **Image OCR** to scan and solve questions directly from images.
- **Dynamic plotting** with Plotly for visualizing equations and data.
- **Handy formula library** for classes 11, 12, and engineering topics.
- **Multi-turn** memory so the chatbot can recall previous queries and context.

If you’re looking for a **user-friendly** way to tackle math problems, visualize concepts, and reference important formulas, **EasyStep-Math** has you covered!

---

## 🏗️ Project Structure

```bash
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── packages.txt         # (Optional) System dependencies for Tesseract on Streamlit Cloud
├── README.md            # This documentation file
└── ...


## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```
> Make sure `requirements.txt` includes `pytesseract`, `pillow`, `streamlit`, and other needed packages.

### 3. Install Tesseract OCR
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```
- **Windows:**
  - [Download Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your system’s PATH.

### 4. Run the App
```bash
streamlit run app.py
```

You can then open the app in your browser (usually at http://localhost:8501).

---

## 🚀 Key Features

1. **🤖 Chat-based Solver**  
   - Ask math questions in a conversational manner.  
   - Receives step-by-step solutions with LaTeX clarity.  
   - Remembers context from previous turns (multi-turn memory).

2. **🖼️ Image OCR**  
   - Upload a PNG/JPG/JPEG of a math problem.  
   - The chatbot extracts text with Tesseract OCR and solves it for you.

3. **📈 Dynamic Plotting**  
   - Type commands like “Plot x^2 from -2 to 2” to generate interactive graphs via Plotly.  
   - Perfect for visualizing functions or data sets.

4. **📚 Formula References**  
   - Handy library of formulas for Class 11, Class 12, and basic engineering.  
   - Quickly accessible from the sidebar for easy reference.

5. **Multi-Turn Memory**  
   - The chatbot keeps track of your past queries, so you can ask follow-up questions or reference previous steps.

---

## 🧩 How the Chatbot Works

1. **User Interaction**  
   - Type a question or greeting in the main chat input.  
   - If you prefer images, use the sidebar to upload a math problem image.

2. **OCR & Parsing**  
   - If an image is uploaded, `pytesseract` extracts text.  
   - The extracted text becomes your query.

3. **LLM Processing**  
   - The system uses **Langchain-Groq** with a multi-turn memory approach.  
   - The **system prompt** ensures consistent LaTeX formatting and step-by-step logic.

4. **Response Generation**  
   - The chatbot provides a friendly, moderate-length answer.  
   - If plotting is requested, it creates an interactive Plotly chart.

5. **Context Memory**  
   - The app keeps a record of the conversation in `st.session_state`.  
   - You can ask “What was my previous question?” or refer to past steps.

---

## ⚠️ Troubleshooting

- **Tesseract Not Found:**  
  - Make sure you have installed Tesseract OCR and added it to your PATH.  
  - On Streamlit Cloud, add `tesseract-ocr` to your `packages.txt`.

- **LaTeX Formatting Issues:**  
  - The chatbot uses $$...$$ for display math.  
  - Avoid raw LaTeX environments like `\begin{align}`.  
  - If formatting is off, check the `SYSTEM_PROMPT` instructions.

- **Plot Not Displaying:**  
  - Ensure you typed something like “Plot x^2 from -2 to 2.”  
  - The chatbot might skip plotting if it doesn’t detect a “plot” command.

---

## 🎉 Enjoy!

**EasyStep-Math** aims to make math more approachable and fun. Whether you’re a high school student, an engineering undergrad, or just curious, we hope this project helps you explore and learn!

If you have any questions, feature requests, or ideas, feel free to [contact me](mailto:nandeshkalshetti1@gmail.com). Happy problem-solving! 🌟
