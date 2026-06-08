"""
Abhivyakti — Indian Sign Language to Speech Converter
Main Gradio app entry point
"""

import gradio as gr
import numpy as np

# ─────────────────────────────────────────────
# TODO (Phase 3): Import trained model & TTS
# from src.inference import ISLInference
# from src.tts import speak
# ─────────────────────────────────────────────

def placeholder_predict(image):
    """
    Placeholder function — will be replaced with real inference in Phase 3.
    For now, returns a dummy response so the UI can be tested.
    """
    return "🚧 Model not trained yet. Complete Phase 1 & 2 first.", 0.0


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="Abhivyakti — ISL Sign to Speech",
    theme=gr.themes.Soft(primary_hue="orange"),
    css="""
        .header { text-align: center; padding: 20px; }
        .output-box { font-size: 1.4em; font-weight: bold; }
    """
) as demo:

    gr.HTML("""
        <div class="header">
            <h1>🤟 Abhivyakti</h1>
            <p style="color: gray;">Indian Sign Language → Speech Converter</p>
            <p><em>अभिव्यक्ति — Expression</em></p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="📷 Your Webcam Feed"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Recognition Output")

            predicted_word = gr.Textbox(
                label="Recognized Sign",
                placeholder="Sign will appear here...",
                elem_classes=["output-box"]
            )
            confidence = gr.Number(
                label="Confidence Score",
                precision=2
            )

            gr.Markdown("### 💬 Sentence Buffer")
            sentence_buffer = gr.Textbox(
                label="Sentence",
                placeholder="Words will accumulate here...",
                lines=3
            )
            speak_btn = gr.Button("🔊 Speak Sentence", variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    gr.Markdown("""
    ---
    **Supported Signs (Phase 2):** Hello · Help · Water · Thank You · Yes · No · 
    Hospital · Food · Pain · Please · Stop · Come · Go · Name · I/Me · You · Good · Bad · More · Time
    
    > ⚠️ Currently in development. Model will be live after Phase 2 training.
    """)

    # Wire up the webcam stream to prediction (Phase 3)
    webcam.stream(
        fn=placeholder_predict,
        inputs=[webcam],
        outputs=[predicted_word, confidence]
    )

if __name__ == "__main__":
    demo.launch()
