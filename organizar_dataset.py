import os, shutil

base = "C:/TCC/archive/Student-engagement-dataset"
destino = "C:/TCC/ER_Dataset"

mapeamento = {
    "Engaged/Focused":           3,
    "Engaged/Confused":          2,
    "Engaged/Frustrated":        2,
    "Not Engaged/Bored":         1,
    "Not Engaged/Drowsy":        0,
    "Not Engaged/Looking Away":  0,
}

for pasta, nivel in mapeamento.items():
    origem = os.path.join(base, pasta.replace("/", os.sep))
    dest_nivel = os.path.join(destino, str(nivel))
    os.makedirs(dest_nivel, exist_ok=True)
    for img in os.listdir(origem):
        src = os.path.join(origem, img)
        # renomeia para evitar conflito entre pastas
        novo_nome = f"{pasta.replace('/', '_')}_{img}"
        dst = os.path.join(dest_nivel, novo_nome)
        shutil.copy2(src, dst)
    print(f"Copiado: {pasta} → nível {nivel}")

# Contagem final
print("\nTotal por nível:")
for nivel in range(4):
    pasta = os.path.join(destino, str(nivel))
    total = len(os.listdir(pasta))
    print(f"  Nível {nivel}: {total} imagens")