import json
from langfuse import Langfuse
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

### Dedicated fetch_* methods for core entities

# Fetch list of traces, supports filters and pagination
observations = langfuse.fetch_observations(name="ChatOpenAI", page=2)
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
            # Process the input messages
            input_messages = observation.input
            if not isinstance(input_messages, list):
                input_messages = [input_messages]

            # Process the output
            processed_output = process_output(observation.output)

            # Create a consistent format for each message
            for msg in input_messages:
                chat.append({"conversations": [msg, processed_output]})

    # Write to JSONL file
    with open("chat.jsonl", "w") as jsonl_file:
        for message in chat:
            jsonl_file.write(json.dumps(message) + "\n")

    print(f"Processed {len(chat)} messages")


process_observation()
