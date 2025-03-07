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

# ==============================
#      CONFIGURATION
# ==============================
GROQ_API_KEY = "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"

# A simple knowledge base for math concepts (for sidebar search)
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
"""
}

# ==============================
#     REFINED SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = """
You are a seasoned mathematician with expertise in advanced fields such as differential geometry, topology, and abstract algebra, as well as their practical applications in physics and computer science. Your responses must be mathematically rigorous, clear, and accessible—much like a professor explaining complex concepts to students.

Your knowledge spans:
- Differential Geometry, Topology, and Abstract Algebra
- Real and Complex Analysis, Number Theory, and Probability Theory
- Programming with Python and Mathematica for computational illustrations

Instructions:
1. Provide precise, step-by-step solutions for mathematical problems.
2. Use LaTeX formatting for all mathematical expressions: wrap display equations in $$...$$ and inline expressions in \\(...\\).
3. Separate explanations from equations using proper indentation and formatting.
4. If a problem admits multiple solution methods, briefly compare them.
5. Provide dynamic visualizations (e.g., Plotly graphs) when appropriate.
6. If you are uncertain about an answer, state the uncertainty explicitly—do not fabricate details.
7. Emulate a distinguished math professor: be clear, methodical, and insightful.
8. Conclude your responses with a clearly highlighted final answer (e.g., in bold and slightly larger font).
9. When sharing code or LaTeX examples, enclose them within triple backticks with the appropriate language tag.

Let’s begin our conversation!
"""

# ==============================
#   SPECIAL QUERY HANDLER
# ==============================
def handle_special_queries(user_text: str) -> str or None:
    """
    If user specifically asks 'Who created this chatbot?', respond with 
    details about Nandesh Kalashetti. Provide a shorter or longer version 
    depending on whether the user includes the word 'short' or 'long'.
    Otherwise, return None.
    """
    text_lower = user_text.lower()

    if "who created this chatbot" in text_lower:
        # Check if user also says 'short'
        if "short" in text_lower:
            return (
                "**Short Answer**: This chatbot was created by Nandesh Kalashetti, "
                "a Full-Stack Web/Gen-AI Developer."
            )
        else:
            # Long version
            return (
                "Nandesh Kalashetti is a Full-Stack Web/Gen-AI Developer. You can reach out to him via email at "
                "nandeshkalshetti1@gmail.com or give him a call at 9420732657. He is located in Samarth Nagar, "
                "Akkalkot. For more information, you can visit his portfolio at "
                "nandesh-kalashettiportfilio2386.netlify.app or check out his GitHub profile at "
                "github.com/Universe7Nandu. He also has a LeetCode profile at leetcode.com/u/Nandesh2386 "
                "and you can connect with him on LinkedIn at linkedin.com/in/nandesh-kalashetti-333a78250."
            )

    return None

# ==============================
#  ASYNC PATCH & APP START
# ==============================
nest_asyncio.apply()

def main():
    st.set_page_config(
        page_title="Enterprise MathPal",
        layout="wide",
        page_icon="🧮"
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
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        margin: 0;
        padding: 0;
    }
    .chat-container {
        max-width: 950px;
        margin: 40px auto;
        background: rgba(255,255,255,0.97);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in-out;
    }
    .chat-title {
        text-align: center;
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .chat-subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.1rem;
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
    .user-bubble {
        background-color: #3498DB;
        color: #fff;
        padding: 12px 18px;
        border-radius: 12px;
        margin-left: auto;
        max-width: 80%;
        font-size: 1rem;
    }
    .assistant-bubble {
        background-color: #ECF0F1;
        color: #333;
        padding: 12px 18px;
        border-radius: 12px;
        margin-right: auto;
        max-width: 80%;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(44,62,80,0.95) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===========================
    #       SIDEBAR
    # ===========================
    with st.sidebar:
        st.title("Enterprise MathPal 🏢🔢")
        st.markdown("""
**An enterprise-grade mathematics platform** with:
- Rigorous, step-by-step solutions in LaTeX
- Dynamic Plotly visualizations
- Multi-turn context-aware conversation
- Quick reference to math concepts

---
""")
        # Math concept search
        st.subheader("Search Math Concepts")
        concept_query = st.text_input("Enter a concept (e.g., 'derivative rules')")
        if concept_query:
            lower_query = concept_query.strip().lower()
            if lower_query in MATH_CONCEPTS:
                st.markdown(MATH_CONCEPTS[lower_query])
            else:
                st.warning("Concept not found. Try 'derivative rules' or 'integration rules'.")

        st.markdown("---")
        if st.button("New Chat"):
            st.session_state.pop("chat_history", None)
            st.success("New conversation started!")

    # ===========================
    #    MAIN CHAT CONTAINER
    # ===========================
    st.markdown("""
    <div class='chat-container'>
      <h1 class='chat-title'>Enterprise MathPal</h1>
      <p class='chat-subtitle'>Ask advanced math questions, request visualizations, or search math concepts. Enjoy clear, rigorous solutions!</p>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display conversation history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ===========================
    #       CHAT INPUT
    # ===========================
    user_input = st.chat_input("Type your advanced math query here (e.g., 'Solve x^2=4' or 'Plot x^2 from -2 to 2')")

    if user_input and user_input.strip():
        # Display user's message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

        # Check if user asked a special query (e.g. who created the chatbot?)
        special_reply = handle_special_queries(user_input)
        if special_reply is not None:
            # We skip the LLM and respond with the special reply
            assistant_response = special_reply
        else:
            # Attempt to parse a plot command before calling the LLM
            plot_generated = False
            plot_figure = None
            parsed_plot = parse_plot_command(user_input)
            if parsed_plot:
                plot_figure = generate_plot(*parsed_plot)  # (expr, x_min, x_max)
                plot_generated = True

            with st.spinner("Processing your request..."):
                # Build the message list for the LLM including multi-turn context
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

        # Display final assistant response
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(f"<div class='assistant-bubble'>{assistant_response}</div>", unsafe_allow_html=True)

# ===========================
#    PLOT HELPER FUNCTIONS
# ===========================
def parse_plot_command(text):
    """
    Parses commands like 'plot x^2 from -2 to 2'
    Returns a tuple (expr_str, x_min, x_max) if matched, else None.
    """
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
    """
    Generates a Plotly plot for the given expression and x-range.
    """
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
