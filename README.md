# Web Navigation Model Training

This project trains a language model for web navigation tasks using data from Langfuse.

## Project Structure

```
.
├── datasets/              # Directory for dataset files
│   └── web_navigation_data.jsonl
├── notebooks/            # Jupyter notebooks for experimentation
│   └── training.ipynb
├── src/                  # Source code
│   ├── config.py        # Configuration settings
│   ├── data_processor.py # Data processing utilities
│   ├── make_dataset.py  # Script to create dataset from Langfuse
│   ├── model_preparator.py # Model preparation utilities
│   └── trainer.py       # Training utilities
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```
HF_TOKEN=your_huggingface_token
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=your_langfuse_host
```

## Usage

1. Generate dataset from Langfuse:
```bash
python src/make_dataset.py
```

2. Train the model:
```bash
python src/main.py
```

## Development

- Use the notebooks in the `notebooks/` directory for experimentation
- The `src/` directory contains the production code
- Dataset files are stored in the `datasets/` directory