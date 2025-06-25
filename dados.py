import pandas as pd
import os

data_set = pd.DataFrame()
# Caminho para o arquivo CSV
df = os.path.join("archive", "merged_data.csv")

import pandas as pd
import numpy as np
import re
import ast
from collections import Counter
from datetime import datetime

# Configurações iniciais
pd.set_option('display.max_columns', None)
data_set = pd.DataFrame()

# Funções auxiliares melhoradas
def parse_list_from_str(texto):
    """Converte strings para listas de forma robusta"""
    if pd.isna(texto) or texto == '':
        return []
    
    if isinstance(texto, list):
        return [str(item).strip() for item in texto if str(item).strip()]
    
    if isinstance(texto, str):
        # Tenta interpretar como lista literal primeiro
        try:
            lista = ast.literal_eval(texto)
            if isinstance(lista, list):
                return [str(item).strip() for item in lista if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
        
        # Se não for uma lista válida, trata como string separada por vírgulas
        itens = [item.strip() for item in texto.split(',') if item.strip()]
        return itens if itens else []
    
    return [str(texto).strip()]

def extrair_requisitos(requisitos_str, ano_lancamento=None):
    """Extrai informações de requisitos do sistema"""
    if pd.isna(requisitos_str) or not isinstance(requisitos_str, str):
        return None, None, 'HD'  # Valor padrão

    sistema_operacional = None
    memoria_ram = None
    armazenamento_tipo = 'HD'

    # Processa as partes dos requisitos
    partes = [part.strip() for part in requisitos_str.split('|') if part.strip()]
    campos = {}
    current_key = None
    
    for parte in partes:
        if parte.endswith(':'):
            current_key = parte[:-1].strip().lower()
        elif current_key:
            campos[current_key] = parte.strip()
            current_key = None

    # Determina sistema operacional
    if 'os' in campos:
        so_text = campos['os'].lower()
        windows_versions = {
            'windows 11': 'Windows 11',
            'windows 10': 'Windows 10',
            'windows 8': 'Windows 8',
            'windows 7': 'Windows 7',
            'windows vista': 'Windows Vista',
            'windows xp': 'Windows XP',
            'windows': 'Windows'
        }
        
        for key, value in windows_versions.items():
            if key in so_text:
                sistema_operacional = value
                break
        else:
            sistema_operacional = campos['os']

    # Define SO padrão baseado no ano se não encontrado
    if sistema_operacional is None and ano_lancamento is not None:
        if ano_lancamento >= 2021:
            sistema_operacional = 'Windows 10/11'
        elif ano_lancamento >= 2015:
            sistema_operacional = 'Windows 10'
        elif ano_lancamento >= 2012:
            sistema_operacional = 'Windows 8'
        elif ano_lancamento >= 2009:
            sistema_operacional = 'Windows 7'
        elif ano_lancamento >= 2006:
            sistema_operacional = 'Windows Vista'
        else:
            sistema_operacional = 'Windows XP ou anterior'

    # Extrai memória RAM
    if 'memory' in campos:
        memoria_match = re.search(r'(\d+)\s*GB', campos['memory'], re.IGNORECASE)
        if memoria_match:
            memoria_ram = int(memoria_match.group(1))

    # Determina tipo de armazenamento
    if 'storage' in campos or 'additional notes' in campos:
        storage_text = ''
        if 'storage' in campos:
            storage_text += campos['storage'] + ' '
        if 'additional notes' in campos:
            storage_text += campos['additional notes']

        armazenamento_tipo = 'SSD' if 'ssd' in storage_text.lower() else 'HD'

    return sistema_operacional, memoria_ram, armazenamento_tipo

def extrair_feedback(texto):
    """Extrai porcentagem e total de reviews"""
    if pd.isna(texto) or texto == '':
        return None, None

    padrao = r'(\d+)% of the ([\d,]+) user reviews'
    match = re.search(padrao, texto)

    if match:
        return int(match.group(1)), int(match.group(2).replace(',', ''))
    return None, None

def converter_preco(valor_str):
    """Converte string de preço para float"""
    if pd.isna(valor_str):
        return None
        
    valor_str = str(valor_str).strip().lower()
    
    if 'free' in valor_str or valor_str in ('0', '0.0'):
        return 0.0
    
    try:
        return float(valor_str.replace('$', '').replace(',', '').strip())
    except ValueError:
        return None

# Dicionário de câmbio dólar -> real por ano
cambio_por_ano = {
    2013: 2.2, 2014: 2.3, 2015: 3.3, 2016: 3.5, 2017: 3.2,
    2018: 3.8, 2019: 3.9, 2020: 5.2, 2021: 5.3, 2022: 5.1, 2023: 5.0
}

# Processamento dos dados
print(f"{datetime.now().strftime('%H:%M:%S')} - Iniciando leitura do arquivo...")
df = pd.read_csv(os.path.join("..", "archive", "merged_data.csv"))
print(f"{datetime.now().strftime('%H:%M:%S')} - Arquivo lido com {len(df)} registros")

# Processar preços
print(f"{datetime.now().strftime('%H:%M:%S')} - Processando preços...")
preco_dolar_lancamento = [converter_preco(v) for v in df['Original Price']]
preco_dolar_desconto = [converter_preco(v) for v in df['Discounted Price']]

# Processar anos de lançamento
print(f"{datetime.now().strftime('%H:%M:%S')} - Processando datas...")
anos_lancamento = []
for data in df['Release Date']:
    data_str = str(data).lower().strip()
    if 'coming soon' in data_str or 'to be announced' in data_str:
        anos_lancamento.append(2024)
    else:
        data_convertida = pd.to_datetime(data, errors='coerce')
        anos_lancamento.append(data_convertida.year if not pd.isna(data_convertida) else None)

# Calcular preços em reais
print(f"{datetime.now().strftime('%H:%M:%S')} - Convertendo para reais...")
preco_real_lancamento = []
preco_real_desconto = []
for dolar, ano in zip(preco_dolar_lancamento, anos_lancamento):
    if dolar is not None and ano in cambio_por_ano:
        preco_real_lancamento.append(dolar * cambio_por_ano[ano])
    else:
        preco_real_lancamento.append(None)

for dolar, ano in zip(preco_dolar_desconto, anos_lancamento):
    if dolar is not None and ano in cambio_por_ano:
        preco_real_desconto.append(dolar * cambio_por_ano[ano])
    else:
        preco_real_desconto.append(None)

# Processar requisitos do sistema
print(f"{datetime.now().strftime('%H:%M:%S')} - Processando requisitos...")
sistemas_operacionais = []
memorias_ram = []
tipos_armazenamento = []

for idx, requisitos in enumerate(df['Minimum Requirements']):
    ano = anos_lancamento[idx] if idx < len(anos_lancamento) else None
    so, ram, storage = extrair_requisitos(requisitos, ano)
    sistemas_operacionais.append(so)
    memorias_ram.append(ram)
    tipos_armazenamento.append(storage)

# Processar feedbacks
print(f"{datetime.now().strftime('%H:%M:%S')} - Processando feedbacks...")
porcentagem_recente = []
total_recente = []
porcentagem_total = []
total_final = []

for feedback in df['Recent Reviews Number']:
    porc, total = extrair_feedback(feedback)
    porcentagem_recente.append(porc)
    total_recente.append(total)

for feedback in df['All Reviews Number']:
    porc, total = extrair_feedback(feedback)
    porcentagem_total.append(porc)
    total_final.append(total)

# Construir o DataFrame final
print(f"{datetime.now().strftime('%H:%M:%S')} - Construindo DataFrame final...")
data_set['titulo'] = df['Title']
data_set['preco_dolar_lancamento'] = preco_dolar_lancamento
data_set['preco_real_lancamento'] = preco_real_lancamento
data_set['preco_dolar_desconto'] = preco_dolar_desconto
data_set['preco_real_desconto'] = preco_real_desconto
data_set['ano_lancamento'] = pd.Series(anos_lancamento).fillna(0).astype(int)
data_set['descricao'] = df['Game Description']
data_set['nota_recente'] = df['Recent Reviews Summary']
data_set['nota_final'] = df['All Reviews Summary']
data_set['porcentagem_recente'] = pd.Series(porcentagem_recente).fillna(0).astype(int)
data_set['link_do_jogo'] = df['Link']
data_set['avaliacao_total_recente'] = pd.Series(total_recente).fillna(0).astype(int)
data_set['porcentagem_final'] = pd.Series(porcentagem_total).fillna(0).astype(int)
data_set['avaliacao_total_final'] = pd.Series(total_final).fillna(0).astype(int)
data_set['desenvolvedora'] = [parse_list_from_str(d) for d in df['Developer']]
data_set['editora'] = df['Publisher'].apply(lambda x: [x.strip()] if pd.notna(x) else [])
data_set['linguagens_suportadas'] = [parse_list_from_str(l) for l in df['Supported Languages']]
data_set['sistema_operacional'] = sistemas_operacionais
data_set['memoria_ram'] = memorias_ram
data_set['tipo_armazenamento'] = tipos_armazenamento
data_set['tags'] = [parse_list_from_str(l) for l in df['Popular Tags']]
data_set['funcionalidades'] = [parse_list_from_str(d) for d in df['Game Features']]
