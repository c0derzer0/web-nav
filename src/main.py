import wandb
from model_preparator import prepare_model
from data_processor import load_and_process_data
from trainer import train_model, save_and_convert_model

def main():
    # Initialize wandb
    wandb.login()
    
    # Prepare model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = prepare_model()
    
    # Load and process dataset
    print("Loading and processing dataset...")
    dataset = load_and_process_data(tokenizer)
    
    # Train model
    print("Starting training...")
    trainer = train_model(model, tokenizer, dataset)
    
    # Save and convert model
    print("Saving and converting model...")
    save_and_convert_model(model, tokenizer)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 