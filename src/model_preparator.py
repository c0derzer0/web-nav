import torch
from unsloth import FastLanguageModel
from config import model_config, lora_config

def prepare_model():
    """Load and prepare the model for training."""
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
    )
    
    # Prepare model for training with LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=lora_config.target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
        loftq_config=lora_config.loftq_config,
    )
    
    return model, tokenizer 