import argparse, inspect, platform
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import torch

@dataclass
class FinetuneCfg:
    model_id: str
    dataset_path: str
    out_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 1024
    use_4bit: bool = True
    use_8bit: bool = False
    warmup_ratio: float = 0.03
    logging_steps: int = 50
    save_steps: int = 1000
    optim: str = "adamw_torch"  # default seguro para Windows

def has_bitsandbytes():
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except Exception:
        return False

def choose_default_optim():
    # Em Windows ou sem bitsandbytes, use AdamW padrão
    if platform.system() == "Windows" or not has_bitsandbytes():
        return "adamw_torch"
    return "paged_adamw_32bit"

def load_base_model(model_id: str, use_4bit: bool, use_8bit: bool):
    quant_args = {}
    if torch.cuda.is_available():
        if use_4bit and has_bitsandbytes():
            try:
                from transformers import BitsAndBytesConfig
                quant_args["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                pass
        elif use_8bit and has_bitsandbytes():
            quant_args["load_in_8bit"] = True
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **quant_args,
    )

def to_text(ex, tok):
    return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset-path", default="data/amazon_sft.jsonl")
    ap.add_argument("--out", default="out/tinyllama-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--use-8bit", action="store_true")
    ap.add_argument("--optim", default=None, help="Força o otimizador (ex.: adamw_torch, paged_adamw_32bit)")
    args = ap.parse_args()

    optim = args.optim or choose_default_optim()

    cfg = FinetuneCfg(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        out_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_seq_len=args.seq_len,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        use_4bit=not args.no_4bit,
        use_8bit=args.use_8bit,
        optim=optim,
    )

    print(f"[train_sft] Sistema: {platform.system()} | bitsandbytes: {has_bitsandbytes()} | optim: {cfg.optim}")

    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=cfg.dataset_path, split="train")
    ds = ds.map(lambda ex: to_text(ex, tok))

    model = load_base_model(cfg.model_id, cfg.use_4bit, cfg.use_8bit)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    args_tr = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim=cfg.optim,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        report_to="none",
    )

    # Compatibilidade com múltiplas versões do TRL
    import inspect as _inspect
    sig = _inspect.signature(SFTTrainer.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = {
        "model": model,
        "args": args_tr,
        "train_dataset": ds,
        "tokenizer": tok,
        "dataset_text_field": "text",
        "formatting_func": (lambda ex: ex["text"]),
        "max_seq_length": cfg.max_seq_len,
        "packing": True,
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    trainer = SFTTrainer(**filtered)

    trainer.train()
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)

    print(f"Adapter salvo em: {cfg.out_dir}")

if __name__ == "__main__":
    main()
