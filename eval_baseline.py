import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SYSTEM = (
    "Você é um assistente de catálogo. Responda em português, com clareza."
)

PROMPT_TMPL = (
    "<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\nTítulo do produto: {title}\n"
    "Pergunta: {question}\n[/USER]\n[ASSISTANT]"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--title", required=True)
    ap.add_argument("--question", default="Quais são as principais características e benefícios?")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    prompt = PROMPT_TMPL.format(system=SYSTEM, title=args.title, question=args.question)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.3)
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
