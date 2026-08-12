import os
os.environ["HF_DATASETS_CACHE"] = "/nvme-ssd1/zwy/navida/.cache/huggingface_cache"
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"
import sys
# rank = int(os.environ.get("RANK", -1))
# os.makedirs("rank_logs", exist_ok=True)
# log_file = open(f"rank_logs/rank{rank}.log", "w", buffering=1)
# sys.stdout = log_file
# sys.stderr = log_file
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import logging
import transformers
from transformers import (
    set_seed, 
    AutoProcessor, 
    TrainingArguments,
    TrainerCallback,
    Trainer
)
from peft import LoraConfig
from transformers.trainer_utils import get_last_checkpoint
from src.dataset.dataset import TTTRLDataset
from src.trainer.test_time_trainer import TTTRLTrainer
from habitat_extensions import measures, task

@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "Dataset path."})


@dataclass
class ModelArguments:
    model_name_or_path: str = None
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # torch_dtype: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Override the default `torch.dtype` and load the model under this dtype.",
    #         "choices": ["auto", "bfloat16", "float16", "float32"],
    #     },
    # )
    # attn_implementation: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
    #         "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
    #     },
    # )

@dataclass
class TTTRLArguments(TrainingArguments):
    simulator_config_path: str = None
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    log_save_path: str = None


logger = logging.getLogger(__name__)


def collate_fn(examples):
    return examples


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    handlers = [logging.StreamHandler(sys.stdout)]
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        os.makedirs(training_args.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(training_args.output_dir, f"train.log"))
        file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S", )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)

    log_level = training_args.get_process_log_level()
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    ################
    # Load datasets
    ################
    dataset = TTTRLDataset(data_args.dataset_path)


    ############################
    # Initialize the NaVIDA TTTRLTrainer
    ############################
    # training_args.dataset_kwargs = {
    #     "skip_prepare_dataset": True,
    # }

    training_args.remove_unused_columns = False

    if model_args.use_peft:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            exclude_modules=".*visual.*"
        )
        logger.info(f"Using peft model")
    else:
        peft_config = None

    trainer = TTTRLTrainer(
        model=model_args.model_name_or_path,
        simulator_config_path=training_args.simulator_config_path,
        num_generations = training_args.num_generations,
        temperature = training_args.temperature,
        log_save_path = training_args.log_save_path,
        peft_config = peft_config,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    trainer.processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset": data_args.dataset_path,
        "tags": ["NaVIDA Training"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TTTRLArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)