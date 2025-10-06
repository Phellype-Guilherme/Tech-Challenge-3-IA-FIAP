import argparse
import sys
import types

# --- Safe imports with fallbacks ---
try:
    from transformers import AutoTokenizer
except Exception as e:
    print("[inference] ERRO: transformers não está instalado corretamente. Tente: pip install -U transformers", file=sys.stderr)
    raise

# Flags para saber o que existe na sua instalação
HAS_CAUSAL = False
HAS_S2S = False
HAS_PIPELINE = False

try:
    from transformers import AutoModelForCausalLM  # type: ignore
    HAS_CAUSAL = True
except Exception:
    pass

try:
    from transformers import AutoModelForSeq2SeqLM  # type: ignore
    HAS_S2S = True
except Exception:
    pass

try:
    from transformers import pipeline  # type: ignore
    HAS_PIPELINE = True
except Exception:
    pass

# PEFT é opcional; se existir, aplicamos o adapter
HAS_PEFT = False
try:
    from peft import PeftModel  # type: ignore
    HAS_PEFT = True
except Exception:
    pass

import torch

SYSTEM = (
    "Você é um assistente de catálogo. Responda EXCLUSIVAMENTE em português do Brasil. "
    "Se o conteúdo base estiver em inglês, traduza fielmente para PT-BR e apresente a resposta. "
    "Use tópicos curtos quando útil."
)

PROMPT_TMPL = (
    "<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\nTítulo do produto: {title}\n"
    "Pergunta: {question}\n[/USER]\n[ASSISTANT]"
)

def make_inputs(tok, text, device):
    return tok(text, return_tensors="pt").to(device)

def do_generate(model, tok, text, max_new_tokens, temperature, top_p, device):
    x = make_inputs(tok, text, device)
    with torch.no_grad():
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else None,
        )
    return tok.decode(y[0], skip_special_tokens=True)

def load_base_and_adapter(model_id, adapter_dir, device_map, offload_dir):
    """
    Tenta carregar o modelo base + adapter, retornando (tokenizer, model, device).
    Nunca depende exclusivamente de AutoModelForCausalLM; cai para Seq2Seq ou pipeline.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model_kwargs = {}

    # device_map handling (nem todas versões suportam; por isso try/except)
    if device_map and device_map.lower() not in ("none", "cpu"):
        model_kwargs["device_map"] = device_map  # "auto"
    else:
        # CPU only
        pass

    if offload_dir:
        # transformers usa "offload_folder", mas versões antigas podem não aceitar; ignorar silenciosamente
        model_kwargs["offload_folder"] = offload_dir

    # 1) Tentar CAUSAL
    if HAS_CAUSAL:
        try:
            from transformers import AutoModelForCausalLM  # reimport local
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **model_kwargs,
            )
            if HAS_PEFT and adapter_dir:
                try:
                    # Algumas versões aceitam device_map/offload via kwargs; fallback se não
                    model = PeftModel.from_pretrained(model, adapter_dir, **{k: v for k, v in [("device_map", model_kwargs.get("device_map")), ("offload_dir", offload_dir)] if v is not None})
                except TypeError:
                    model = PeftModel.from_pretrained(model, adapter_dir)
            device = next(model.parameters()).device
            return tok, model, device
        except Exception as e:
            print(f"[inference] Aviso: falha ao carregar AutoModelForCausalLM: {e}", file=sys.stderr)

    # 2) Tentar SEQ2SEQ (T5/FLAN etc.)
    if HAS_S2S:
        try:
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                **{k: v for k, v in model_kwargs.items() if k != "offload_folder"}  # algumas versões não aceitam
            )
            if HAS_PEFT and adapter_dir:
                try:
                    model = PeftModel.from_pretrained(model, adapter_dir)
                except TypeError:
                    model = PeftModel.from_pretrained(model, adapter_dir)
            device = next(model.parameters()).device
            return tok, model, device
        except Exception as e:
            print(f"[inference] Aviso: falha ao carregar AutoModelForSeq2SeqLM: {e}", file=sys.stderr)

    # 3) Fallback final: pipeline
    if not HAS_PIPELINE:
        raise RuntimeError("Sua instalação do transformers não suporta nem AutoModel* nem pipeline. Reinstale transformers.")

    from transformers import pipeline
    # Se chegamos aqui, vamos construir pipeline de text-generation
    # Tentamos usar o tokenizer carregado para evitar novo download
    try:
        pipe = pipeline("text-generation", model=model_id, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    except Exception:
        pipe = pipeline("text-generation", model=model_id, device=0 if torch.cuda.is_available() else -1)

    # Empacotamos o pipeline em um objeto simples com .generate e .device
    class PipeWrapper:
        def __init__(self, pipe, tok):
            self.pipe = pipe
            self.tok = tok
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        def generate(self, **kwargs):
            # convertemos de volta em texto simples para pipeline
            input_ids = kwargs.get("input_ids")
            if input_ids is not None:
                text = self.tok.batch_decode(input_ids, skip_special_tokens=True)[0]
            else:
                text = kwargs.get("text", "")
            out = self.pipe(text, max_new_tokens=kwargs.get("max_new_tokens", 200), do_sample=True)
            full = out[0]["generated_text"]
            # retornamos algo parecido com generate: recodificamos
            return self.tok(full, return_tensors="pt")["input_ids"]

        def parameters(self):
            # Para compatibilidade com quem chama .device
            return [torch.zeros(1, device=self.device, requires_grad=False)]

    model = PipeWrapper(pipe, tok)
    device = model.device
    return tok, model, device

def translate_pt_if_needed(answer_text, tok, model, device, max_new_tokens, temperature, top_p, force_pt):
    if not force_pt:
        return answer_text
    # Heurística simples para detectar inglês
    lower = answer_text.lower()
    english_markers = [" the ", " and ", " with ", "features", "benefits", "wireless", "battery life", "comfortable"]
    if sum(m in lower for m in english_markers) < 2:
        return answer_text  # provavelmente já está em PT

    prompt = (
        "<s>[SYSTEM]\nVocê é um tradutor técnico. "
        "Traduza fielmente para português do Brasil, mantendo termos técnicos e sentido.\n[/SYSTEM]\n"
        f"[USER]\nTraduza o texto a seguir para PT-BR:\n\n{answer_text}\n[/USER]\n[ASSISTANT]"
    )
    # Se for pipeline wrapper, a geração já é tratada por do_generate()
    return do_generate(model, tok, prompt, max_new_tokens, temperature, top_p, device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter", default=None, help="Diretório do adapter treinado (opcional)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--question", default="Quais são as principais características e benefícios?")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device-map", default="cpu", help="cpu|auto (padrão: cpu para evitar offload)")
    ap.add_argument("--offload-dir", default=None, help="Pasta para offload quando usar device_map=auto (ex.: offload_infer)")
    ap.add_argument("--force-pt", action="store_true", help="Tradução automática para PT-BR se detectar inglês")
    args = ap.parse_args()

    tok, model, device = load_base_and_adapter(args.model_id, args.adapter, args.device_map, args.offload_dir)

    prompt = PROMPT_TMPL.format(system=SYSTEM, title=args.title, question=args.question)
    answer = do_generate(model, tok, prompt, args.max_new_tokens, args.temperature, args.top_p, device)
    answer = translate_pt_if_needed(answer, tok, model, device, args.max_new_tokens, args.temperature, args.top_p, args.force_pt)

    print(answer)

if __name__ == "__main__":
    main()
