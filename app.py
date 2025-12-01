import os
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import pipeline
from TTS.api import TTS

# --- Device Setup ---
device = "cpu"

# --- 1. STT Setup (Whisper) ---
print("Loading Whisper...")
STT_MODEL_NAME = "openai/whisper-tiny.en"
stt_pipe = pipeline("automatic-speech-recognition", model=STT_MODEL_NAME, device=device)

# --- 2. LLM Setup (YOUR GGUF MODEL) ---
print("Downloading and loading your GGUF model...")

# Download your specific model file
model_path = hf_hub_download(repo_id="gzsol/model_1b", filename="model-Q8_0.gguf")

# Initialize Llama-cpp (Optimized for CPU)
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Context window
    n_threads=2,  # CPU threads
    verbose=False,
)

# --- 3. TTS Setup (Coqui) ---
print("Loading TTS...")
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)


# --- Core Functions ---


def chat_with_bot(message, history):
    """
    Chat with your GGUF model.
    """
    # Ensure history is a list
    if history is None:
        history = []

    if not message or not message.strip():
        return history, ""

    try:
        # Format history for Llama (OpenAI format)
        messages = []
        # Add system prompt if desired
        messages.append(
            {"role": "system", "content": "You are a helpful AI assistant."}
        )

        # Add conversation history
        for msg in history:
            messages.append(msg)  # msg is already {"role": "...", "content": "..."}

        # Add new user message
        messages.append({"role": "user", "content": message})

        # Generate response using your GGUF model
        output = llm.create_chat_completion(
            messages=messages, max_tokens=256, temperature=0.7, stream=False
        )

        response = output["choices"][0]["message"]["content"]

        # Append to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return history, response

    except Exception as e:
        print(f"LLM Error: {e}")
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, f"Error: {e}"


def text_to_speech_from_chat(chat_response):
    """Takes the chat response and converts it to speech."""
    if not chat_response or chat_response.startswith("Error"):
        return None, "No valid response to synthesize."

    output_path = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        output_path = temp_file.name
        temp_file.close()

        tts_model.tts_to_file(
            text=chat_response,
            file_path=output_path,
        )
        return output_path, "Speech synthesis complete."

    except Exception as e:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        return None, f"Error during TTS: {e}"


def speech_to_text_and_chat(audio_file_path, history):
    """Performs STT, then Chatbot generation, returning the final response text and audio."""
    if audio_file_path is None:
        return "Please upload an audio file.", history, "", None, "Awaiting input."

    # 1. STT
    try:
        result = stt_pipe(audio_file_path)
        transcribed_text = result["text"]
    except Exception as e:
        return f"Error during STT: {e}", history, "", None, f"Error during STT: {e}"

    # 2. Chatbot (Your GGUF Model)
    updated_history, last_response_text = chat_with_bot(transcribed_text, history)

    # 3. TTS
    audio_path, status_text = text_to_speech_from_chat(last_response_text)

    return (
        transcribed_text,
        updated_history,
        last_response_text,
        audio_path,
        status_text,
    )


# --- Gradio Interface ---
custom_css = """
#status { font-weight: bold; color: #2563eb; }
.chatbot { height: 400px; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ GGUF Voice Assistant (Running your model_1b)")
    gr.Markdown("**Note:** This app uses `gzsol/model_1b` (GGUF) on CPU.")

    # Global State
    # We no longer need 'chat_history_ids' because llama_cpp handles context internally via the messages list

    with gr.Tabs():

        # --- TAB 1: FULL VOICE CHAT ---
        with gr.TabItem("🗣️ Voice Assistant"):
            # CRITICAL FIX: type="messages"
            voice_chat_history = gr.Chatbot(
                label="Conversation Log",
                elem_classes=["chatbot"],
                value=[],
                type="messages",
            )

            with gr.Row():
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Input Audio",
                )
                voice_audio_out = gr.Audio(label="AI Voice Response", autoplay=True)

            voice_transcription = gr.Textbox(label="User Transcription")
            voice_response_text = gr.Textbox(label="AI Response (Text)")
            voice_status = gr.Textbox(elem_id="status", label="Status")

            run_btn = gr.Button("Transcribe, Chat & Speak", variant="primary")
            clear_voice_btn = gr.Button("Clear")

            run_btn.click(
                fn=speech_to_text_and_chat,
                inputs=[audio_in, voice_chat_history],
                outputs=[
                    voice_transcription,
                    voice_chat_history,
                    voice_response_text,
                    voice_audio_out,
                    voice_status,
                ],
            )

            clear_voice_btn.click(
                lambda: (None, [], "", None, ""),
                None,
                [
                    audio_in,
                    voice_chat_history,
                    voice_response_text,
                    voice_audio_out,
                    voice_status,
                ],
            )

        # --- TAB 2: TEXT CHAT ---
        with gr.TabItem("💬 Text Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_classes=["chatbot"],
                value=[],
                type="messages",
            )
            msg = gr.Textbox(label="Message")
            submit_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear")

            def chat_text_wrapper(message, history):
                h, _ = chat_with_bot(message, history)
                return h

            msg.submit(chat_text_wrapper, [msg, chatbot], [chatbot]).then(
                lambda: "", None, msg
            )
            submit_btn.click(chat_text_wrapper, [msg, chatbot], [chatbot]).then(
                lambda: "", None, msg
            )
            clear_btn.click(lambda: [], None, chatbot)

demo.launch(css=custom_css)
