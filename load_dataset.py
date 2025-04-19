from datasets import load_dataset

dataset = load_dataset("json", data_files="chat.jsonl")

print(dataset)
