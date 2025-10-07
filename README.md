# 🧠 Tech Challenge - Fine-tuning de Modelos de Linguagem com AmazonTitles-1.3MM

Este projeto foi desenvolvido como parte do **Tech Challenge da FIAP (Fase 3 – Inteligência Artificial)** e tem como objetivo aplicar técnicas de **fine-tuning em modelos fundacionais** (como LLaMA, TinyLLaMA, Mistral, etc.) utilizando o dataset **AmazonTitles-1.3MM**.

O propósito é treinar um modelo capaz de **gerar descrições de produtos a partir de seus títulos**, simulando perguntas reais de usuários sobre itens disponíveis no catálogo da Amazon.

---

## 📁 Estrutura do Projeto

```
Tech-Challenge-3-IA-FIAP/
├── data/
│   ├── trn.json                 # Dataset original (fonte AmazonTitles-1.3MM) baixa e adicionar nesta pasta
│   └── amazon_sft.jsonl         # Dataset preparado para fine-tuning
├── out/
│   └── tinyllama-lora/          # Diretório do adapter salvo após o fine-tuning
├── prep_data.py                 # Script de pré-processamento e preparação do dataset
├── eval_baseline.py             # Avaliação do modelo base antes do treinamento
├── train_sft.py                 # Execução do fine-tuning (LoRA/QLoRA)
├── inference.py                 # Geração de respostas com o modelo treinado
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação do projeto
```

---

## 📊 Sobre o Dataset

O **AmazonTitles-1.3MM** contém consultas e títulos de produtos da Amazon associados às suas descrições, coletados a partir de interações reais de usuários.

Cada entrada do arquivo `trn.json` possui o formato:
```json
{
  "uid": "0000031909",
  "title": "Girls Ballet Tutu Neon Pink",
  "content": "High quality 3 layer ballet tutu. 12 inches in length",
  "target_ind": [...],
  "target_rel": [...]
}
```

Para o fine-tuning, são utilizadas apenas as colunas:
- **title** → título do produto  
- **content** → descrição (texto alvo)

Essas informações são transformadas em prompts de entrada para treinar o modelo a responder perguntas como:
> "Quais são as principais características e benefícios deste produto?"

---

## ⚙️ Como Executar o Projeto

### 1️⃣ Preparar os Dados
```powershell
python .\prep_data.py `
  --input "C:\Users\vkrlo\OneDrive\Área de Trabalho\Tech-Challenge-3-IA-FIAP\data\trn.json" `
  --output .\data\amazon_sft.jsonl `
  --variants-per-title 2 `
  --max-samples 200000 `
  --min-len 10 `
  --max-content-len 1200
```

### 2️⃣ Avaliar o Modelo Base (pré-treino)
```powershell
python .\eval_baseline.py `
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --title "Fone de Ouvido Bluetooth JBL Tune 510BT" `
  --question "Quais são as principais características e benefícios?"
```

### 3️⃣ Executar o Fine-Tuning
```powershell
python .\train_sft.py `
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --dataset-path .\data\amazon_sft.jsonl `
  --out .\out\tinyllama-lora `
  --epochs 1 --seq-len 1024 --batch 2 --grad-accum 8
```

### 4️⃣ Fazer Inferência
```powershell
python .\inference.py `
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --adapter .\out\tinyllama-lora `
  --title "Fone de Ouvido Bluetooth JBL Tune 510BT" `
  --question "Quais são as principais características e benefícios?" `
  --max-new-tokens 420 `
  --device-map cpu `
  --force-pt
```

---

## 🧠 Técnicas Utilizadas

- **Fine-Tuning Supervisionado (SFT)** com LoRA / QLoRA  
- **Modelos base compatíveis com Hugging Face Transformers**
- **Tokenização e truncamento dinâmico**
- **Avaliação baseline antes do treino**
- **Inferência com tradução automática para PT-BR**
- **Offload automático para CPU (compatível com Windows)**

---

## ⚡ Dicas de Execução

- No Windows, se o `bitsandbytes` não estiver disponível, use:
  ```bash
  --optim adamw_torch
  ```
- Ajuste `--seq-len`, `--batch` e `--grad-accum` conforme o limite de memória.
- Para rodar sem GPU, adicione `--device-map cpu` e `--offload-dir offload_infer`.

---

## 📦 Saídas Geradas

- **`out/tinyllama-lora/`** → Adapter do modelo fine-tunado.  
- **`data/amazon_sft.jsonl`** → Dataset formatado para treinamento.  
- **Respostas inferidas** → Saída textual em português (via `--force-pt`).

---

## 📚 Bibliotecas Principais

- `transformers` – Modelos fundacionais e tokenização  
- `datasets` – Manipulação e split de dados  
- `trl` – Fine-tuning supervisionado (SFTTrainer)  
- `peft` – Adaptação leve com LoRA / QLoRA  
- `accelerate` – Treinamento otimizado (CPU/GPU/offload)  
- `torch` – Backend de deep learning  

---

## 👨‍💻 Autor

**Phellype Guilherme Pereira da Silva**  
**RM:** 361625  
**Projeto:** Fase 3 - Pós Tech FIAP - Inteligência Artificial  
**Instituição:** [FIAP - Faculdade de Informática e Administração Paulista](https://www.fiap.com.br)
