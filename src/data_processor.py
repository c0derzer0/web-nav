from datasets import load_dataset
import json
from unsloth.chat_templates import get_chat_template
from config import dataset_config

def load_and_process_data(tokenizer):
    """Load and process the dataset with chat template formatting."""
    # Get chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_config.train_file)
    
    # Split into train and test
    dataset = dataset['train'].train_test_split(
        test_size=dataset_config.test_size,
        seed=dataset_config.seed
    )
    
    # Apply chat template to dataset
    dataset = dataset.map(
        apply_template,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return dataset

def apply_template(examples, tokenizer):
    """Apply chat template to examples."""
    # Handle both single example and batched examples
    if isinstance(examples["conversations"], list) and isinstance(
        examples["conversations"][0], list
    ):
        # Batched processing
        formatted_chats = []
        for conv in examples["conversations"]:
            messages = []
            for msg in conv:
                if isinstance(msg, str):
                    try:
                        msg_dict = json.loads(msg)
                        messages.append(msg_dict)
                    except json.JSONDecodeError:
                        messages.append({"role": "user", "content": msg})
                else:
                    messages.append(msg)

            formatted_chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_chats.append(formatted_chat)
        return {"text": formatted_chats}
    else:
        # Single example processing
        messages = []
        for msg in examples["conversations"]:
            if isinstance(msg, str):
                try:
                    msg_dict = json.loads(msg)
                    messages.append(msg_dict)
                except json.JSONDecodeError:
                    messages.append({"role": "user", "content": msg})
            else:
                messages.append(msg)

        formatted_chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": formatted_chat} 