"""
Baixa o modelo base (processor + pesos) do Hugging Face Hub e salva em uma
pasta local, para que o fine-tuning (finetuning.ipynb) carregue o modelo a
partir do disco em vez de baixar do Hub a cada execucao.

Uso:
    python download_base_model.py
    python download_base_model.py --model microsoft/swin-base-patch4-window7-224-in22k --out swin-base-local

Se a rede exigir proxy para acessar o Hugging Face Hub, defina as variaveis
de ambiente HTTP_PROXY/HTTPS_PROXY antes de rodar este script (evite colocar
credenciais diretamente no codigo).
"""

import argparse

from transformers import AutoImageProcessor, AutoModelForImageClassification


def download_model(model_name, out_dir):
    print(f"Baixando processador de '{model_name}'...")
    processor = AutoImageProcessor.from_pretrained(model_name)

    print(f"Baixando modelo de '{model_name}'...")
    model = AutoModelForImageClassification.from_pretrained(model_name)

    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Modelo e processor salvos em: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Baixa e salva localmente o modelo base do Hugging Face.")
    parser.add_argument(
        "--model",
        default="microsoft/swin-base-patch4-window7-224-in22k",
        help="Nome do modelo no Hugging Face Hub (default: microsoft/swin-base-patch4-window7-224-in22k)",
    )
    parser.add_argument(
        "--out",
        default="swin-base-local",
        help="Pasta local de destino (default: swin-base-local)",
    )
    args = parser.parse_args()

    download_model(args.model, args.out)


if __name__ == "__main__":
    main()
