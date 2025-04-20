from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from config import training_config, model_config, hf_config
from unsloth.chat_templates import train_on_responses_only


def setup_training_args():
    """Setup training arguments using parameters from config."""
    return TrainingArguments(
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_steps=training_config.warmup_steps,
        num_train_epochs=training_config.num_train_epochs,
        max_steps=training_config.max_steps,
        learning_rate=training_config.learning_rate,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        logging_steps=training_config.logging_steps,
        optim=training_config.optim,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        seed=training_config.seed,
        output_dir=training_config.output_dir,
        run_name=training_config.run_name,
        report_to=training_config.report_to,
        save_strategy=training_config.save_strategy,
        eval_strategy=training_config.eval_strategy,
        save_steps=training_config.save_steps,
        eval_steps=training_config.eval_steps,
    )

def train_model(model, tokenizer, dataset):
    """Train the model."""
    # Setup training arguments
    training_args = setup_training_args()
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=training_config.dataset_num_proc,
        packing=training_config.packing,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    # Train the model
    trainer.train()
    
    return trainer

def save_and_convert_model(model, tokenizer):
    """Save the model and push to Hugging Face Hub in GGUF format."""
    model.push_to_hub_gguf(
        repo_id=hf_config.repo_id,
        tokenizer=tokenizer,
        quantization_method=hf_config.quantization_method,
        token=hf_config.hf_token
    ) 