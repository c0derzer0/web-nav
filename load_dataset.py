from datasets import load_dataset

dataset = load_dataset("json", data_files="chat_1.jsonl")

print(dataset)
