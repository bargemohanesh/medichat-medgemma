import gradio as gr
import requests

CUSTOM_CSS = """
body {
    background-color: #F5F7FA
}
h1 {
    color: #0B3C5D
}
.gr-button-primary {
    background-color: #0B3C5D !important;
    border: none;
}
.gr-button-primary:hover {
    background-color: #1CA7A6 !important;
}
"""

API_URL = "http://127.0.0.1:8000/ask"

with gr.Blocks(title = "medichat - Medgemma",
               css = CUSTOM_CSS
) as demo:
    gr.Markdown("# 🧠 **Medichat - Medgemma Powered Medical Assistant**")
    gr.Markdown("## 🛡️ **Privacy-preserving · Offline-capable · Safety-first**.")
    gr.Markdown("### Ask general medical questions. **educational information only**")

    chatbot = gr.Chatbot(label = "conversation")
    state = gr.State([])

    question = gr.Textbox(label = "Question", placeholder = "e. g., What is diabetes?", lines=2)
    btn = gr.Button("Ask")

    request_id = gr.Textbox(label="Request ID")
    #answer = gr.Textbox(label="Answer", lines=20)
    disclaimer = gr.Textbox(label="Disclaimer")
    
    def chat_with_api(user_message, history):
        if history is None:
            history = []

        if not user_message or user_message.strip() == "":
            return history, history,"", ""

        r = requests.post(API_URL, json = {"question":user_message}, timeout=300)
        r.raise_for_status()
        data = r.json()

        history = history + [
            {"role":"user", "content": user_message},
            {"role": "assistant", "content": data["answer"]}
        ]
        return history, history, data["request_id"], data["disclaimer"]

    btn.click(fn=chat_with_api, inputs = [question, state], outputs = [chatbot, state, request_id, disclaimer])

demo.launch(server_name = "0.0.0.0", server_port=7860)

