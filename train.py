import os
import argparse

import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser, Trainer,
                          set_seed, TrainingArguments, TrainerCallback)
from datasets import load_dataset, concatenate_datasets

def tokenize(entry, tokenizer):
   outputs = tokenizer(
      entry["Input.text"],
      truncation=True,
      max_length = 50,
   )
   return outputs

def main(args):
  set_seed(2)

  tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", cache_dir="/gscratch/xlab/olo126/.cache")
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

  if args.warmup:
    toxigen_train = load_dataset("toxigen/toxigen-data", name="annotations", split="train", cache_dir="/gscratch/xlab/olo126/.cache")
    print(len(toxigen_train.select(range(0, len(toxigen_train), 3))))
    toxigen_train = toxigen_train.select(range(0, len(toxigen_train), 3)).shuffle(seed=2)
    train_dataset = toxigen_train.select(range(1000)).map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer})
    print(train_dataset[0])

  else:
    train_file = args.dataset_file
    if isinstance(train_file, str):
      train_file = [train_file]
    processed_datasets = load_dataset("json", data_files=train_file)["train"].map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer}).shuffle(seed=2)
    train_dataset = processed_datasets

  model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir="/gscratch/xlab/olo126/.cache")

  embedding_size = model.get_input_embeddings().weight.shape[0]
  if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
    if isinstance(model, PeftModel):
      model.get_input_embeddings().weight.requires_grad = False
      model.get_output_embeddings().weight.requires_grad = False

  if not isinstance(model, PeftModel) and args.lora:
    print("USING LORA")
    lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=args.lora_r,
      lora_alpha=args.lora_a,
      lora_dropout=args.lora_dropout,
      target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # for checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

  training_args = TrainingArguments(
     output_dir=args.output_dir + f"_{args.epochs}e",
     lr_scheduler_type='linear',
     warmup_ratio=0.03,
     save_strategy='epoch',
     num_train_epochs=args.epochs,
     bf16=True,
     tf32=False,
     overwrite_output_dir=True,
     report_to='wandb',
     seed=2,
     optim="adamw_torch",
     learning_rate=2e-05,
     per_device_train_batch_size=1,
     gradient_accumulation_steps=32,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
      tokenizer=tokenizer, mlm=False),
  )

  train_result = trainer.train()
  trainer.save_model()
  metrics = train_result.metrics
  metrics['train_samples'] = len(train_dataset)
  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  trainer.save_state()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--warmup', action="store_true")
  parser.add_argument('--warmup_set')
  parser.add_argument('--dataset_file')
  parser.add_argument('--pt_percentage')
  parser.add_argument('--output_dir')
  parser.add_argument('--model_path')
  parser.add_argument('--epochs', type = int, default = 4)
  parser.add_argument('--rand_set', action="store_true")
  parser.add_argument('--lora',  action="store_false")
  parser.add_argument('--lora_r', default = 128)
  parser.add_argument('--lora_a', default = 512)
  parser.add_argument('--lora_dropout', default = 0.1)
  parser.add_argument('--lora_target_modules', default = ["q_proj", "k_proj", "v_proj", "o_proj"])
  parser.add_argument('--changing_data', default=False)
  args = parser.parse_args()
  main(args)