import json
from langfuse import Langfuse
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import load_dataset

load_dotenv()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

### Dedicated fetch_* methods for core entities

# Fetch list of traces, supports filters and pagination
observations = langfuse.fetch_observations(name="ChatOpenAI", page=1)
print(len(observations.data))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


def process_output(output):
    # Extract the function arguments and name from the GPT output
    if "tool_calls" in output["additional_kwargs"]:
        tool_call = output["additional_kwargs"]["tool_calls"][0]
        function_args = json.loads(tool_call["function"]["arguments"])
        function_name = tool_call["function"]["name"]
    else:
        return {"role": "assistant", "content": output["content"]}

    # Construct the LLaMA format
    llama_format = {
        "name": function_name,
        "parameters": {
            "current_state": function_args["current_state"],
            "action": function_args["action"],
        },
    }
    formatted_output = {
        "role": "assistant",
        "content": json.dumps(llama_format),
    }
    return formatted_output


def process_observation():
    chat = []
    for i, observation in enumerate(observations.data):
        print(f"Processing observation {i}")
        if observation.output:
            # Get the entire conversation
            conversation = observation.input
            if not isinstance(conversation, list):
                conversation = [conversation]

            # Process the output and ensure it's a string
            processed_output = process_output(observation.output)
            if isinstance(processed_output, dict):
                processed_output = json.dumps(processed_output)

            # Format each message in the conversation
            formatted_conversation = []
            for msg in conversation:
                if isinstance(msg, dict):
                    formatted_conversation.append(json.dumps(msg))
                else:
                    formatted_conversation.append(str(msg))

            # Add the processed output
            formatted_conversation.append(processed_output)

            # Add the complete conversation to chat
            chat.append({"conversations": formatted_conversation})

    # Write to JSONL file
    with open("chat_1.jsonl", "w") as jsonl_file:
        for message in chat:
            jsonl_file.write(json.dumps(message) + "\n")

    print(f"Processed {len(chat)} conversations")


def apply_chat_template_to_jsonl(jsonl_file_path, output_path="formatted_chat_dataset"):
    """
    Apply chat template to a JSONL file and save the formatted dataset.

    Args:
        jsonl_file_path (str): Path to the input JSONL file
        output_path (str): Path to save the formatted dataset
    """
    # Load the dataset
    dataset = load_dataset("json", data_files=jsonl_file_path)

    # Apply chat template
    def apply_template(examples):
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
            return {"formatted_chat": formatted_chats}
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
            return {"formatted_chat": formatted_chat}

    # Apply the template to the dataset
    formatted_dataset = dataset.map(apply_template, batched=True)

    # Save the formatted dataset
    formatted_dataset.save_to_disk(output_path)

    return formatted_dataset


# Example usage:
if __name__ == "__main__":
    # First process observations and save to JSONL
    # process_observation()

    # Then apply chat template to the JSONL file
    formatted_dataset = apply_chat_template_to_jsonl("chat_1.jsonl")
    print("\nFormatted dataset structure:")
    print(formatted_dataset)

    # Print the first row
    print("\nFirst row of the dataset:")
    first_row = formatted_dataset["train"][0]
    print(json.dumps(first_row, indent=2))
