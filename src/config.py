from dataclasses import dataclass, field
from unsloth import is_bfloat16_supported
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length: int = None  # None for automatic RoPE scaling
    dtype: str = None  # None for auto detection
    load_in_4bit: bool = False  # Use 4bit quantization

@dataclass
class LoRAConfig:
    r: int = 16  # LoRA attention dimension
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lora_alpha: int = 16
    lora_dropout: float = 0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: dict = None

@dataclass
class TrainingConfig:
    # Training parameters from notebook
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 2
    max_steps: int = 15
    learning_rate: float = 2e-4
    fp16: bool = not is_bfloat16_supported()  # Will be set based on hardware
    bf16: bool = is_bfloat16_supported()  # Will be set based on hardware
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    run_name: str = "web_navigation_run_1"
    report_to: str = "wandb"
    save_strategy: str = "steps"
    eval_strategy: str = "steps"
    save_steps: int = 10
    eval_steps: int = 10
    
    # Additional required parameters
    dataset_num_proc: int = 2
    packing: bool = False

@dataclass
class DatasetConfig:
    train_file: str = os.path.join(PROJECT_ROOT, "datasets", "test_json_files", "chat_1.jsonl")
    test_size: float = 0.1
    seed: int = 42

@dataclass
class HuggingFaceConfig:
    repo_id: str = "hiddenVariable/llama_3.2_3b_instruct_web_navigation"
    hf_token: str = os.getenv("HF_TOKEN")
    quantization_method: str = "f16"

# Initialize configs with default values
model_config = ModelConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
dataset_config = DatasetConfig()
hf_config = HuggingFaceConfig() 