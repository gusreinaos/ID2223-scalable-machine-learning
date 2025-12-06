# Lab 2 - LLM Tuning and Serving

## Demo

**Hugging Face Space:** [gzsol/llm-finetune-lab2](https://huggingface.co/spaces/gzsol/llm-finetune-lab2)

This link gives access to our fine-tuned Llama-3.2-1B model with voice and text chat.

## Project Overview

This project implements Parameter Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation) on a Large Language Model. We fine-tuned Llama-3.2-1B-Instruct on the FineTome dataset and deployed it as a multimodal voice assistant using Gradio ina HuggingFace space.

## Deliverables

### Task 1: Fine-Tune and Deploy

✅ **Training Notebook:** [Lab2-adjusted.ipynb](Lab2-adjusted.ipynb)  
✅ **Deployed UI:** [Hugging Face Space](https://huggingface.co/spaces/gzsol/llm-finetune-lab2)  
✅ **Model Artifacts:**
- LoRA Adapters: [gzsol/lora_model_1b](https://huggingface.co/gzsol/lora_model_1b)
- Merged 16-bit: [gzsol/model_1b](https://huggingface.co/gzsol/model_1b)
- GGUF (CPU inference): Converted for deployment

### Task 2: Performance Improvements

## Implementation Details

### Model Selection
**Base Model:** `unsloth/Llama-3.2-1B-Instruct`


### Training Configuration

#### LoRA Hyperparameters
```python
r = 16                   # LoRA rank
lora_alpha = 16          # Scaling factor
lora_dropout = 0.05      # Dropout
target_modules = [       # Attention and MLP layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

**Justification:**
- **Rank 16:** Standard choice from [LoRA paper](https://arxiv.org/abs/2106.09685)
- **Alpha = Rank:** Recommended configuration to keep learning stable
- **Target all layers:** Maximum coverage for comprehensive adaptation
- **5% Dropout:** Helps to prevent overfitting

#### Training Hyperparameters
```python
learning_rate = 3e-4           # Effective for instruction tuning
batch_size = 4                 # Based on T4 GPU memory (16GB)
gradient_accumulation = 2      # Effective batch size: 8
epochs = 1                     # Dataset quality for 1-epoch training
optimizer = "adamw_8bit"       # Memory-efficient
lr_scheduler = "cosine"        # Smooth convergence
warmup_ratio = 0.01            # Minimal warmup for small model
```

**Justification:**
- **Learning Rate (3e-4):** Higher than typical (2e-4) for faster convergence on small model
- **Batch Size (4):** Maximum that fits in T4 memory with gradient checkpointing
- **8-bit AdamW:** Reduces optimizer memory by ~50% with negligible performance loss
- **Cosine Schedule:** Proven effective in transformer training (Vaswani et al.)

### Training Results
**Performance Metrics:**
- Training Loss: 0.878
- Validation Loss: 0.932
- Training Time: 22 minutes (T4 GPU)
- GPU Memory: 5.2GB peak (35.6% utilization)
- Parameters Trained: 11.2M / 1,247M (0.90%)

**WandB Dashboard:**

TODO: We need to put the picture here for the WanDB dashboard of the metrics

And a bit of text on what we see

## Gradio UI

Our Gradio deployment boasts the following features:

### 1. Voice Assistant
- **Speech-to-Text:** Whisper-tiny for audio transcription
- **LLM Inference:** GGUF quantized model for CPU efficiency
- **Text-to-Speech:** Coqui TTS for voice output

### 2. Web context fetching from DuckDuckGo

We have added in-context learning to improve the output of the model by querying DuckDuckGo's web search API. This is used, whenever the user uses any of the defined `search_keywords` to look for up-to-date information.

```python
def get_web_context(message):
    search_keywords = ['current', 'latest', 'recent', 'today', 'weather', 'price', '2024', '2025']
    
    if any(keyword in message.lower() for keyword in search_keywords):
        return web_results
```

Some results of this functionality can be appreciated in the following images:
![Prompt example](docs/prompt.png)
The following promot which made use of the word `current` triggered the following web search through DuckDuckGo's API at inference time:
```txt
Result 1: President of the United States - Wikipedia
  Body: The president of the United States ( POTUS ) b is the head of state and head of government of the Un...
Result 2: President of the United States - Wikipedia, the free
  Body: The President of the United States (often abbreviated "POTUS") is the head of ... The 43rd and curre...
Result 3: President of the United States - Simple English Wikipedia, the
  Body: Donald Trump is the 47th and current president of the United States , in office since January 20, 20...
```
This information is further added to the context and used to answer the user appropriately.

## Task 2: Model & Data Improvements

### Model-Centric Approaches

#### LoRA optimization
We improve the model by using LoRA. This only updates a small part of the model instead of its entirity. We apply it to all the most important parts of the transformer, such as the attention and feed-forward layers.

#### Efficient training
We make use of mixed precision training (FP16/BF16) so the model can train faster and use less memory. Also, we use an 8-bit AdamW optimizer, which reduces memory use even more. Combining these can help us train with larger batches of data and get better results even with limited hardware which is the case of the Google Collab we were using.


### Data-Centric Approaches

#### Web search in-context learning
To make the model smarter, we have added web search integration using DuckDuckGo's search API. When the model detects a question that needs current information by filtering for certain keywords, it searches the web and adds the results to the prompt before answering. This has helped improve the model output with more up-to-date answers.

## Usage

### Training
1. Open [Lab2-adjusted.ipynb](Lab2-adjusted.ipynb) in Colab
2. Mount Google Drive for checkpoints
3. Set WandB API key (optional)
4. Run all cells

### Inference
```bash
pip install -r requirements.txt
python app.py
```

## Team
Georg Zsolnai and Oscar Reina Gustafsson

