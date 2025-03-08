```markdown
# EasyStep-Math 😊🪄

Welcome to **EasyStep-Math** – your interactive, enterprise-grade math assistant! This chatbot is designed to help students and professionals alike solve math problems step by step, visualize graphs dynamically, and even extract questions from images with OCR. It's like having a friendly math tutor in your pocket! 🤓✨

---

## 🔗 Contact Information

- **LinkedIn:** [Nandesh Kalashetti](https://www.linkedin.com/in/nandesh-kalashetti-333a78250)
- **Portfolio/Resume:** [View My Portfolio](https://nandesh-kalashettiportfilio2386.netlify.app)
- **Email:** [nandeshkalshetti1@gmail.com](mailto:nandeshkalshetti1@gmail.com)
- **Phone:** 9420732657

---

## 📖 Project Overview

**EasyStep-Math** is a powerful, user-friendly mathematics platform that provides:
- **Step-by-step solutions** with detailed explanations and LaTeX formatting.
- **Dynamic visualizations** using Plotly for interactive graphs.
- **Image OCR capability** to extract and solve math problems from images.
- **Quick access** to essential formulas and concepts for classes 11, 12, and engineering.
- **Multi-turn context-aware conversation** that remembers previous queries to provide better answers.

Whether you need help with algebra, calculus, or advanced topics, EasyStep-Math is here to guide you through every step of your math journey! 🚀

---

## ⚙️ Setup & Usage

### Prerequisites
- **Python 3.8+**
- **Tesseract OCR**:
  - **Linux:** `sudo apt-get update && sudo apt-get install tesseract-ocr`
  - **Windows:** [Download Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your PATH.
- Python packages as listed in `requirements.txt`.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

### Requirements File
Ensure your `requirements.txt` includes:
```
streamlit>=1.43.0
langchain-groq>=0.2.4
nest_asyncio>=1.6.0
plotly>=5.16.0
sympy>=1.9
numpy>=1.23.5
pytesseract
pillow
```

---

## ✨ Key Features

- **Interactive Chatbot 🤖:**  
  Ask math questions and receive clear, step-by-step solutions in real-time, with friendly emojis and a conversational tone.

- **Image OCR 🖼️:**  
  Upload an image containing a math problem, and the app will extract the question using OCR and solve it for you.

- **Dynamic Plotting 📈:**  
  Generate interactive graphs for equations (e.g., "Plot x² from -2 to 2") with dynamic Plotly visualization.

- **Formula Library 📚:**  
  Quickly look up key formulas and concepts for classes 11, 12, and engineering subjects via the sidebar.

- **Multi-turn Memory 🔄:**  
  The chatbot remembers previous questions so you can ask follow-up queries seamlessly.

---

## 💡 How It Works

1. **Chat-Based Interaction:**  
   Type your question or greeting in the chat box. The chatbot responds with detailed explanations and step-by-step solutions using LaTeX for clarity.

2. **Image Upload:**  
   Use the sidebar’s image upload option to submit a photo of your math problem. The app extracts the text and uses it as your query.

3. **Visualize Concepts:**  
   Ask for graphs (e.g., "Plot x² from -2 to 2") and get interactive Plotly visualizations to better understand your problems.

4. **Search Formulas:**  
   Explore essential formulas and mathematical concepts quickly using the search box in the sidebar.

Enjoy your journey through mathematics with **EasyStep-Math** – making math fun, interactive, and accessible! 🎉🧮
```
