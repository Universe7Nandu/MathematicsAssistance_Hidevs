import sys
import os
import re
import json
import asyncio
import nest_asyncio
import streamlit as st
import plotly.graph_objs as go
import sympy
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from langchain_groq import ChatGroq

# -------------- OCR --------------
import pytesseract
from PIL import Image, UnidentifiedImageError

# ==============================
#      CONFIGURATION
# ==============================
GROQ_API_KEY = "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"

# ==============================
#  EXTENDED KNOWLEDGE BASE
# ==============================
MATH_CONCEPTS = {
    "derivative rules": """
**Derivative Rules** (Reference):
1. \\(\\frac{d}{dx}[x^n] = n x^{n-1}\\)
2. \\(\\frac{d}{dx}[\\sin x] = \\cos x\\)
3. \\(\\frac{d}{dx}[\\cos x] = -\\sin x\\)
4. \\(\\frac{d}{dx}[e^x] = e^x\\)
5. \\(\\frac{d}{dx}[\\ln x] = \\frac{1}{x}\\)
""",
    "integration rules": """
**Integration Rules** (Reference):
1. \\(\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C\\)
2. \\(\\int \\sin x \\, dx = -\\cos x + C\\)
3. \\(\\int \\cos x \\, dx = \\sin x + C\\)
4. \\(\\int e^x \\, dx = e^x + C\\)
5. \\(\\int \\frac{1}{x} \\, dx = \\ln|x| + C\\)
""",
    "pythagorean theorem": """
**Pythagorean Theorem** (Reference):
For a right triangle with legs \\(a\\) and \\(b\\) and hypotenuse \\(c\\):
\\[
a^2 + b^2 = c^2.
\\]
""",
    "class 11 formulas": """
**Class 11 Formulas**:
1. **Trigonometry**: \\(\\sin^2 \\theta + \\cos^2 \\theta = 1\\)
2. **Quadratic Equations**: For \\(ax^2 + bx + c = 0\\), \\(x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\\)
3. **Permutations**: Number of permutations of n distinct objects is \\(n!\\)
4. **Combinations**: Number of ways to choose r from n is \\(\\binom{n}{r} = \\frac{n!}{r!(n-r)!}\\)
5. **Binomial Theorem**: \\((1 + x)^n = \\sum_{k=0}^{n} \\binom{n}{k} x^k\\)
""",
    "class 12 formulas": """
**Class 12 Formulas**:
1. **Differentiation**: \\(\\frac{d}{dx}[\\sin x] = \\cos x,\\, \\frac{d}{dx}[\\cos x] = -\\sin x\\)
2. **Integration**: \\(\\int \\sin x\\, dx = -\\cos x + C,\\, \\int \\cos x\\, dx = \\sin x + C\\)
3. **Probability**: For independent events A and B, \\(P(A \\cap B) = P(A)P(B)\\)
4. **Continuity & Differentiability**: A function f(x) is continuous at x=a if \\(\\lim_{x \\to a} f(x) = f(a)\\).
5. **Matrices**: If A is an invertible matrix, \\(A^{-1} A = I\\).
""",
    "engineering formulas": """
**Engineering Formulas** (Reference):
1. **Ohm's Law**: \\(V = IR\\)
2. **Bernoulli's Principle**: \\(p + \\frac{1}{2}\\rho v^2 + \\rho g h = \\text{constant}\\)
3. **Hooke's Law**: \\(F = k x\\)
4. **Stress & Strain**: \\(\\sigma = \\frac{F}{A},\\, \\epsilon = \\frac{\\Delta L}{L}\\)
5. **Kinematics**: \\(v = u + at,\\, s = ut + \\frac{1}{2}at^2\\)
"""
}

# ==============================
#    IMPROVED SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = """
You are a friendly yet knowledgeable mathematics tutor who responds with moderate detail and uses light emojis to keep the conversation engaging.

**Formatting Guidelines**:
1. If the user just greets you (e.g., "hi", "hello"), greet them politely (e.g., "Hello there! 👋") and wait for a math question.
2. When the user asks a math question, provide a clear, step-by-step explanation with LaTeX formatting.
3. Use **$$ ... $$** for display math, and **\\(...\\)** for inline math. 
4. **Avoid** environment commands like \\begin{{align}}, \\begin{{equation}}, etc.
5. If you want to highlight a final result, use $$\\boxed{{...}}$$ or **bold** text. 
6. Keep answers moderately detailed (not too short, not overly lengthy).
7. Mention multiple solution methods only if relevant.
8. Keep the tone friendly and professional, with occasional emojis to add warmth (e.g., "Sure thing! 🤓").
9. If uncertain, say so and suggest possible directions.

Let's begin!
"""

# ==============================
#   SPECIAL QUERY HANDLER
# ==============================
def handle_special_queries(user_text: str, chat_history: list) -> str or None:
    text_lower = user_text.lower()

    # 1. "Who created this chatbot?" (short or long)
    if "who created this chatbot" in text_lower:
        if "short" in text_lower:
            return (
                "🤖 **Short Answer**: This chatbot was created by Nandesh Kalashetti, "
                "a Full-Stack Web/Gen-AI Developer."
            )
        else:
            # Long version
            return (
                "🤖 Nandesh Kalashetti is a Full-Stack Web/Gen-AI Developer. You can reach out to him via email at "
                "nandeshkalshetti1@gmail.com or give him a call at 9420732657. He is located in Samarth Nagar, "
                "Akkalkot. For more information, you can visit his portfolio at "
                "nandesh-kalashettiportfilio2386.netlify.app or check out his GitHub profile at "
                "github.com/Universe7Nandu. He also has a LeetCode profile at leetcode.com/u/Nandesh2386 "
                "and you can connect with him on LinkedIn at linkedin.com/in/nandesh-kalashetti-333a78250."
            )

    # 2. "What's my previous question?" or "What was my previous question?"
    if "what's my previous question" in text_lower or "what was my previous question" in text_lower:
        last_user_query = get_last_user_query(chat_history)
        if last_user_query is None:
            return "You haven't asked any previous question yet. 🤔"
        else:
            return f"Your previous question was: \"{last_user_query}\""

    return None

def get_last_user_query(chat_history: list) -> str or None:
    if len(chat_history) < 2:
        return None
    for i in range(len(chat_history) - 2, -1, -1):
        entry = chat_history[i]
        if entry["role"] == "user":
            return entry["content"]
    return None

# ==============================
#  ASYNC PATCH & APP START
# ==============================
nest_asyncio.apply()

def main():
    st.set_page_config(
        page_title="EasyStep-Math😊",
        layout="wide",
        page_icon="🔮"
    )

    # ===========================
    #         CUSTOM UI
    # ===========================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    body {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        margin: 0;
        padding: 0;
    }
    .chat-container {
        max-width: 950px;
        margin: 40px auto;
        background:black;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in-out;
    }
    .chat-title {
        text-align: center;
        color: #333333;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .chat-subtitle {
        text-align: center;
        color: #555555;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatInput textarea {
        border-radius: 12px !important;
        padding: 15px !important;
        font-size: 1rem !important;
    }
    /* Avatars with emojis for user & assistant */
    .user-avatar::before {
        content: "👤 ";
        font-size: 1.2rem;
        margin-right: 5px;
    }
    .assistant-avatar::before {
        content: "🤖 ";
        font-size: 1.2rem;
        margin-right: 5px;
    }
    .user-bubble {
        background-color:#bd4e1e;
        color: #fff;
        padding: 12px 18px;
        border-radius: 16px;
        margin-left: auto;
        max-width: 80%;
        font-size: 1.05rem;
        line-height: 1.5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .assistant-bubble {
        background-color: #fefefe;
        color: #333;
        padding: 12px 18px;
        border-radius: 16px;
        margin-right: auto;
        max-width: 80%;
        font-size: 1.05rem;
        line-height: 1.5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 6px;
        margin-bottom: 6px;
    }
    [data-testid="stSidebar"] {
        background-color: black !important;
        border:1.5px soild white;
    }
    .modern-search-title {
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .modern-history-title {
        color: #ffffff;
        font-size: 1.1rem;
        margin-top: 15px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .upload-icon {
        font-size: 1.3rem;
        color: #ffffff;
        margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===========================
    #       SIDEBAR
    # ===========================
    with st.sidebar:
        st.title("EasyStep-Math😊 🪄")
        st.markdown("""
**Features**:
- 🤖 **Chat-based** math solver with moderate explanations
- 🖼️ **Image OCR** to extract & solve questions from images
- 📚 **Formula references** (classes 11, 12, engineering)
- 📈 **Plotly** for dynamic graphing
---
""")

        # Searching for math concepts
        st.markdown("<div class='modern-search-title'>Explore Key Formulas & Concepts</div>", unsafe_allow_html=True)
        concept_query = st.text_input("", placeholder="E.g. 'derivative rules', 'class 12 formulas'...")
        if concept_query:
            lower_query = concept_query.strip().lower()
            if lower_query in MATH_CONCEPTS:
                st.markdown(MATH_CONCEPTS[lower_query])
            else:
                st.warning("Concept not found. Try 'derivative rules', 'class 11 formulas', etc.")

        st.markdown("---")

        # Conversation History
        st.markdown("<div class='modern-history-title'>Conversation History</div>", unsafe_allow_html=True)
        if "chat_history" in st.session_state and st.session_state["chat_history"]:
            user_queries = [item["content"] for item in st.session_state["chat_history"] if item["role"] == "user"]
            if user_queries:
                for i, q in enumerate(user_queries, 1):
                    st.markdown(f"**{i}.** {q}")
            else:
                st.info("No user queries yet.")
        else:
            st.info("No conversation history yet.")

        st.markdown("---")

        # Image Upload
        st.markdown("<div class='modern-history-title'><span class='upload-icon'>🖼️</span>Upload Image</div>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("", type=["png","jpg","jpeg"])
        if uploaded_image is not None:
            try:
                img = Image.open(uploaded_image)
                try:
                    extracted_text = pytesseract.image_to_string(img)
                    if extracted_text.strip():
                        st.success(f"**Extracted Question**:\n\n{extracted_text.strip()}")
                        if st.button("Use This as My Question"):
                            st.session_state["chat_history"].append({"role": "user", "content": extracted_text})
                            with st.chat_message("user", avatar="👤"):
                                st.markdown(f"<div class='user-bubble'>{extracted_text}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("No readable text detected. Try a clearer image.")
                except pytesseract.pytesseract.TesseractNotFoundError:
                    st.error("Tesseract is not installed or not in your PATH. Please install it to use OCR.")
            except UnidentifiedImageError:
                st.error("Could not identify the image. Please upload a valid PNG/JPG/JPEG.")
            except Exception as e:
                st.error(f"Error reading image: {e}")

        st.markdown("---")

        # New Chat
        if st.button("New Chat"):
            st.session_state.pop("chat_history", None)
            st.success("New conversation started! 🆕")

    # ===========================
    #    MAIN CHAT CONTAINER
    # ===========================
    st.markdown("""
    <div class='chat-container'>
      <h1 class='chat-title'>EasyStep-Math😊</h1>
      <p class='chat-subtitle'>Type your math questions below, or upload an image in the sidebar. I'll respond with moderate detail and a friendly tone! 🤓</p>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display conversation
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            # Show user message with user avatar
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            # Show assistant message with assistant avatar
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ===========================
    #       CHAT INPUT
    # ===========================
    user_input = st.chat_input("Type your math question or greeting here...")

    if user_input and user_input.strip():
        # Show user's message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

        # Check for special queries
        special_reply = handle_special_queries(user_input, st.session_state["chat_history"])
        if special_reply is not None:
            assistant_response = special_reply
        else:
            # Attempt to parse a plot command
            plot_generated = False
            plot_figure = None
            parsed_plot = parse_plot_command(user_input)
            if parsed_plot:
                plot_figure = generate_plot(*parsed_plot)
                plot_generated = True

            with st.spinner("Thinking..."):
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for entry in st.session_state["chat_history"]:
                    messages.append({"role": entry["role"], "content": entry["content"]})

                llm = ChatGroq(
                    temperature=0.7,
                    groq_api_key=GROQ_API_KEY,
                    model_name="mixtral-8x7b-32768"
                )
                try:
                    response = asyncio.run(llm.ainvoke(messages))
                    assistant_response = response.content
                except Exception as e:
                    assistant_response = f"Error: {str(e)}"

            if plot_generated and plot_figure is not None:
                st.plotly_chart(plot_figure, use_container_width=True)
                assistant_response += "\n\n(Generated a dynamic Plotly graph based on your request.)"

        # Show assistant's response
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='assistant-bubble'>{assistant_response}</div>", unsafe_allow_html=True)


# ===========================
#    PLOT HELPER FUNCTIONS
# ===========================
def parse_plot_command(text):
    text_lower = text.lower()
    if "plot" not in text_lower:
        return None
    match = re.search(r"plot\s+(.+)\s+from\s+(-?\d+)\s+to\s+(-?\d+)", text_lower)
    if match:
        expr_str = match.group(1).strip()
        try:
            x_min = float(match.group(2))
            x_max = float(match.group(3))
            return (expr_str, x_min, x_max)
        except ValueError:
            return None
    return None

def generate_plot(expr_str, x_min, x_max):
    x = sympy.Symbol('x', real=True)
    try:
        parsed_expr = parse_expr(expr_str, transformations=sympy.parsing.sympy_parser.standard_transformations)
    except Exception:
        return None
    xs = np.linspace(x_min, x_max, 200)
    f = sympy.lambdify(x, parsed_expr, 'numpy')
    try:
        ys = f(xs)
    except Exception:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f"{expr_str}"))
    fig.update_layout(
        title=f"Plot of {expr_str} from {x_min} to {x_max}",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white"
    )
    return fig

# ===========================
#       ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()
