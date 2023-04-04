import csv


def adicionar_coluna_csv(arquivo_csv, valores):
    # Abrir o arquivo CSV e ler os valores em uma lista de listas
    with open(arquivo_csv, 'r') as f:
        reader = csv.reader(f)
        linhas = [linha for linha in reader]

    # Adicionar cada valor em uma nova coluna
    for i, linha in enumerate(linhas):
        valor = valores[i] if i < len(valores) else ''
        linha.append(valor)

    # Escrever a lista atualizada no arquivo CSV
    with open(arquivo_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(linhas)
# abrir o arquivo CSV


def extrair_ultima_coluna(arquivo_csv):
    # Abrir o arquivo CSV e ler os valores em uma lista de listas
    with open(arquivo_csv, 'r') as arquivo:
        conteudo = csv.reader(arquivo)
        linhas = list(conteudo)

    # Criar uma lista com a última coluna
    ultima_coluna = []
    for linha in linhas:
        ultima_coluna.append(linha[-1])

    return ultima_coluna

adicionar_coluna_csv('./fase/csv/data_tiff.csv', extrair_ultima_coluna('./fase/csv/data_shpTest.csv'))