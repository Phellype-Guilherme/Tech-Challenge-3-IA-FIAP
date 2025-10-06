import argparse, json, random, os
from pathlib import Path
from html import unescape

TEMPLATES = [
    "Com base no título, descreva o produto em detalhes.",
    "Quais são as principais características e benefícios?",
    "Faça um resumo técnico do produto, incluindo especificações importantes.",
    "Explique o que é este produto e para quem ele é indicado.",
    "Liste os destaques e diferenciais do produto."
]

SYSTEM_PROMPT = (
    "Você é um assistente de catálogo. Responda de forma objetiva, em português, "
    "com bullets quando útil. Se a pergunta for ambígua, responda com o melhor "
    "resumo possível do produto."
)

def build_messages(title: str, content: str, question: str):
    user = (
        f"Título do produto: {title}\n"
        f"Pergunta: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": content.strip()},
    ]

def iter_records(path: str):
    """Suporta:
    1) JSONL: uma linha por objeto
    2) JSON array: arquivo começa com '[' e contém objetos
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            for rec in data:
                yield rec
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def clean_text(x: str) -> str:
    x = (x or "").strip()
    x = unescape(x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Caminho para trn.json (JSONL ou JSON array)")
    ap.add_argument("--output", default="data/amazon_sft.jsonl")
    ap.add_argument("--max-samples", type=int, default=120000, help="Limite total de exemplos (após expansão).")
    ap.add_argument("--variants-per-title", type=int, default=2, help="Quantas perguntas distintas por título.")
    ap.add_argument("--min-len", type=int, default=10, help="mínimo de caracteres do conteúdo")
    ap.add_argument("--max-content-len", type=int, default=1200, help="trunca descrições muito longas")
    args = ap.parse_args()

    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    written = 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        for rec in iter_records(args.input):
            title = clean_text(rec.get("title", ""))
            content = clean_text(rec.get("content", ""))

            if not title or not content or len(content) < args.min_len:
                continue

            if len(content) > args.max_content_len:
                content = content[:args.max_content_len].rstrip() + "…"

            for _ in range(max(1, args.variants_per_title)):
                q = rng.choice(TEMPLATES)
                ex = {"messages": build_messages(title, content, q)}
                f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_samples:
                    break
            if written >= args.max_samples:
                break

    print(f"Gerado: {written} exemplos em {args.output}")

if __name__ == "__main__":
    main()
