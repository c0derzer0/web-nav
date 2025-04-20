import json
from langfuse import Langfuse
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from datasets import load_dataset

# Load environment variables from the root directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

def initialize_langfuse():
    """Initialize Langfuse client with environment variables."""
    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )

def process_output(output):
    """Process the output from Langfuse observation."""
    # Extract the function arguments and name from the GPT output
    try:
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
    except Exception as e:
        print(f"Error processing output: {e}")
        return {"role": "assistant", "content": output["content"]}

def process_conversation(conversation, processed_output):
    """Process a single conversation and its output."""
    if not isinstance(conversation, list):
        conversation = [conversation]

    # Format each message in the conversation
    formatted_conversation = []
    for msg in conversation:
        if isinstance(msg, dict):
            formatted_conversation.append(json.dumps(msg))
        else:
            formatted_conversation.append(str(msg))

    # Add the processed output
    if isinstance(processed_output, dict):
        processed_output = json.dumps(processed_output)
    formatted_conversation.append(processed_output)

    return {"conversations": formatted_conversation}

def process_observations(langfuse_client, output_file):
    """Process all pages of observations and save to JSONL file."""
    chat = []
    page = 1
    
    while True:
        print(f"Processing page {page}")
        observations = langfuse_client.fetch_observations(name="ChatOpenAI", page=page)
        
        if not observations.data:
            break
            
        for i, observation in enumerate(observations.data):
            print(f"Processing observation {i} on page {page}")
            if observation.output:
                # Process the output
                processed_output = process_output(observation.output)
                
                # Process the conversation
                formatted_conversation = process_conversation(
                    observation.input, 
                    processed_output
                )
                
                # Add to chat list
                chat.append(formatted_conversation)
        
        page += 1

    # Write to JSONL file in the datasets directory
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', output_file)
    with open(output_path, "w") as jsonl_file:
        for message in chat:
            jsonl_file.write(json.dumps(message) + "\n")

    print(f"Processed {len(chat)} conversations across {page-1} pages")

def main():
    """Main function to process observations."""
    langfuse_client = initialize_langfuse()
    output_file = "web_navigation_data.jsonl"
    process_observations(langfuse_client, output_file)

if __name__ == "__main__":
    main()