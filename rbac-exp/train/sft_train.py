#!/usr/bin/env python3
"""
SFT Training for Column-Level RBAC Text-to-SQL.

This script trains models to generate SQL with column-level access control awareness.
The model learns to:
1. Generate SQL when the user has permission to access required columns
2. Refuse with "Sorry, I cannot answer." when permission is denied

Usage:
    # Single GPU
    python rbac-exp/train/sft_train.py \
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
        --dataset column_level_rbac_spider_train \
        --template chatml \
        --output_dir rbac-exp/output/adapter/qwen2.5-7b-column-rbac

    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus 4 rbac-exp/train/sft_train.py \
        --deepspeed rbac-exp/configs/ds_config.json \
        --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
        --dataset column_level_rbac_spider_train \
        --template chatml \
        --output_dir rbac-exp/output/adapter/qwen2.5-14b-column-rbac
"""
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Setup paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_PATH)
RBAC_EXP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RBAC_EXP_PATH)

# Import from local llm_base (copied from experiments)
from configs.config import IGNORE_INDEX
from llm_base.config_parser import get_train_args
from llm_base.load_tokenizer import load_model_and_tokenizer
from llm_base.loggings import LogCallback, get_logger
from llm_base.model_trainer import (
    ComputeMetrics,
    Seq2SeqPeftTrainer,
    get_logits_processor,
    plot_loss,
)
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

# Import Column-Level RBAC data utilities
from data_process.sft_data_utils import (
    load_sft_dataset,
    get_template_and_fix_tokenizer,
    preprocess_supervised_dataset,
    split_dataset,
)

if TYPE_CHECKING:
    from configs import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
    )
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    Run supervised fine-tuning for Column-Level RBAC.
    
    This function:
    1. Loads the Column-Level RBAC training dataset
    2. Loads model and tokenizer with LoRA configuration
    3. Preprocesses data with appropriate chat template
    4. Trains using Seq2SeqPeftTrainer with DeepSpeed support
    """
    logger.info("=" * 80)
    logger.info("Column-Level RBAC SFT Training")
    logger.info("=" * 80)
    
    # ========== Load Model and Tokenizer ==========
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(
        model_args, finetuning_args, training_args.do_train
    )
    
    # ========== Load Dataset ==========
    logger.info(f"Loading dataset: {data_args.dataset}")
    
    # Load raw dataset
    max_samples = getattr(data_args, 'max_samples', None)
    dataset = load_sft_dataset(data_args.dataset, max_samples=max_samples)
    
    logger.info(f"Loaded {len(dataset)} samples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    # ========== Get Template and Fix Tokenizer ==========
    logger.info(f"Using template: {data_args.template}")
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)
    
    # ========== Preprocess Dataset ==========
    logger.info("Preprocessing dataset...")
    
    max_source_length = getattr(data_args, 'max_source_length', 2048)
    max_target_length = getattr(data_args, 'max_target_length', 512)
    
    tokenized_dataset = preprocess_supervised_dataset(
        dataset,
        tokenizer,
        template,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    
    logger.info(f"Tokenized {len(tokenized_dataset)} samples")
    
    # ========== Debug: Print Sample ==========
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        logger.info("-" * 80)
        logger.info("Sample tokenized data:")
        logger.info(f"Input length: {len(sample['input_ids'])} tokens")
        decoded_input = tokenizer.decode(sample['input_ids'][:300])
        logger.info(f"Decoded input (first 300 tokens): {decoded_input}...")
        
        # Check for RBAC elements
        has_rbac = "Role Access Policy" in decoded_input or "Accessible Columns" in decoded_input
        logger.info(f"Contains RBAC policy: {has_rbac}")
        logger.info("-" * 80)
    
    # ========== Data Collator ==========
    ignore_pad = getattr(data_args, 'ignore_pad_token_for_loss', True)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if ignore_pad else tokenizer.pad_token_id,
    )
    
    # ========== Training Arguments ==========
    training_args_dict = training_args.to_dict()
    training_args_dict.update(
        dict(
            generation_max_length=training_args.generation_max_length
            or max_target_length,
            generation_num_beams=getattr(data_args, 'eval_num_beams', 1)
            or training_args.generation_num_beams,
        )
    )
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    
    # ========== Split Dataset ==========
    eval_ratio = getattr(data_args, 'eval_dataset_size', 0.02)
    dataset_splits = split_dataset(tokenized_dataset, eval_ratio=eval_ratio)
    
    logger.info(f"Train samples: {len(dataset_splits['train_dataset'])}")
    if dataset_splits.get('eval_dataset'):
        logger.info(f"Eval samples: {len(dataset_splits['eval_dataset'])}")
    
    # ========== Initialize Trainer ==========
    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer)
        if training_args.predict_with_generate
        else None,
        **dataset_splits
    )
    
    # ========== Generation kwargs ==========
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = list(
        set([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids)
    )
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    
    # ========== Training ==========
    if training_args.do_train:
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
        
        logger.info("Training completed!")
    
    # ========== Evaluation ==========
    if training_args.do_eval:
        logger.info("=" * 80)
        logger.info("Starting evaluation...")
        logger.info("=" * 80)
        
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        
        if training_args.predict_with_generate:
            metrics.pop("eval_loss", None)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info("Evaluation completed!")
    
    # ========== Prediction ==========
    if training_args.do_predict:
        logger.info("=" * 80)
        logger.info("Starting prediction...")
        logger.info("=" * 80)
        
        predict_results = trainer.predict(
            tokenized_dataset, metric_key_prefix="predict", **gen_kwargs
        )
        
        if training_args.predict_with_generate:
            predict_results.metrics.pop("predict_loss", None)
        
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
        
        logger.info("Prediction completed!")


def train(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """Main entry point for training."""
    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
    ) = get_train_args(args)
    
    callbacks = [LogCallback()] if callbacks is None else callbacks
    
    run_sft(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        callbacks,
    )


def export_model(
    args: Optional[Dict[str, Any]] = None, 
    max_shard_size: Optional[str] = "10GB"
):
    """Export trained model to a directory."""
    model_args, _, training_args, finetuning_args, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except Exception as e:
        logger.warning(f"Cannot save tokenizer: {e}. Please copy the files manually.")


if __name__ == "__main__":
    train()
