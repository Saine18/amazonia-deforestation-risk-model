#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================================
# SCRIPT 2: INTERPOLAÇÃO ESPACIAL DE DADOS CLIMÁTICOS (CORRIGIDO)
# Objetivo: Interpolar dados pontuais (estações) para uma mancha contínua (grid)
# Método: IDW (Inverse Distance Weighting)
# Entrada: Prioriza dados LIMPOS (sem outliers) gerados no Script 1
# ============================================================================

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
import warnings
import sys

warnings.filterwarnings('ignore')

print("=" * 100)
print("🗺️  INTERPOLAÇÃO ESPACIAL - ESTAÇÃO SECA PARA GRID 5KM")
print("   Método: IDW (Inverse Distance Weighting)")
print("=" * 100)

# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================

# Usar caminho relativo à localização do script
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_PROCESSED = BASE_DIR / 'data' / 'processed'
DIR_CLIMA = DIR_PROCESSED / 'clima'
DIR_GRID = DIR_PROCESSED / 'grid'
DIR_OUTPUT = DIR_GRID
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)

# Parâmetros IDW
POWER = 2             # Expoente da distância (padrão: 2)
MAX_DISTANCE_KM = 300 # Raio de busca (300km garante cobertura na Amazônia)
MIN_NEIGHBORS = 3     # Mínimo de estações para interpolar um ponto
MAX_NEIGHBORS = 15    # Máximo de estações a considerar

print(f"\n📂 Configuração:")
print(f"   Base: {BASE_DIR}")
print(f"   Pasta de clima: {DIR_CLIMA}")
print(f"   Saída: {DIR_OUTPUT}")

# ============================================================================
# 2. CARREGAR DADOS (COM PROTEÇÃO CONTRA OUTLIERS)
# ============================================================================

print(f"\n{'='*100}")
print("📥 CARREGANDO DADOS")
print(f"{'='*100}")

# 2.1 Carregar estações (Prioridade para arquivo LIMPO)
PATH_LIMPO = DIR_CLIMA / 'estacoes_inmet_norte_climatologia_limpo.gpkg'
PATH_BRUTO = DIR_CLIMA / 'estacoes_inmet_norte_climatologia_2008_2024.gpkg'

if PATH_LIMPO.exists():
    print("✅ Usando arquivo de dados LIMPOS (Outliers removidos).")
    PATH_ESTACOES = PATH_LIMPO
elif PATH_BRUTO.exists():
    print("⚠️  AVISO: Arquivo limpo não encontrado. Usando dados BRUTOS.")
    PATH_ESTACOES = PATH_BRUTO
else:
    print(f"\n❌ ERRO: Nenhum arquivo de estações encontrado em {DIR_CLIMA}")
    print(f"   Procurado: {PATH_LIMPO.name} ou {PATH_BRUTO.name}")
    sys.exit(1)

gdf_estacoes = gpd.read_file(PATH_ESTACOES)
print(f"   ✅ {len(gdf_estacoes)} estações carregadas")

# 2.2 Carregar ou Criar Grid Base
print(f"\n🗺️  Preparando Grid de 5km...")

possiveis_grids = list(DIR_GRID.glob('*.gpkg'))
PATH_GRID = None

# Tenta achar um grid existente que não seja o de clima
for p in possiveis_grids:
    if '5km' in p.name and 'clima' not in p.name:
        PATH_GRID = p
        break

if PATH_GRID and PATH_GRID.exists():
    print(f"   ✅ Grid base encontrado: {PATH_GRID.name}")
    gdf_grid = gpd.read_file(PATH_GRID)
else:
    print(f"⚠️  Grid base não encontrado. Gerando grid automático para a Região Norte...")
    # Cria grid cobrindo a extensão das estações
    bbox = box(-74.0, -14.0, -46.0, 6.0) # Extensão aprox. do Norte
    gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox]}, crs="EPSG:4326")
    
    # Projeção Cônica de Albers para América do Sul (métrica precisa)
    ALBERS_PROJ = "ESRI:102033"
    try:
        gdf_bbox = gdf_bbox.to_crs(ALBERS_PROJ)
    except:
        # Fallback se a projeção ESRI não estiver disponível na lib PROJ local
        ALBERS_PROJ = "EPSG:31981" # UTM 21S (aproximação aceitável)
        gdf_bbox = gdf_bbox.to_crs(ALBERS_PROJ)

    xmin, ymin, xmax, ymax = gdf_bbox.total_bounds
    cell_size = 5000 # 5km
    
    x = np.arange(xmin, xmax, cell_size)
    y = np.arange(ymin, ymax, cell_size)
    xx, yy = np.meshgrid(x, y)
    
    points = [Point(px, py) for px, py in zip(xx.flatten(), yy.flatten())]
    gdf_grid = gpd.GeoDataFrame({'geometry': points}, crs=ALBERS_PROJ)
    print(f"   ✅ Grid gerado com {len(gdf_grid):,} pontos.")

# ============================================================================
# 3. SINCRONIZAR PROJEÇÕES (CRÍTICO PARA IDW)
# ============================================================================

# Albers Equal Area (Ideal para cálculo de distância) ou a projeção que o grid usou
TARGET_CRS = gdf_grid.crs 

if gdf_estacoes.crs != TARGET_CRS:
    gdf_estacoes = gdf_estacoes.to_crs(TARGET_CRS)

# ============================================================================
# 4. EXECUTAR INTERPOLAÇÃO (IDW)
# ============================================================================

print(f"\n{'='*100}")
print("⚙️  EXECUTANDO INTERPOLAÇÃO (IDW)")
print(f"{'='*100}")

# Coordenadas
coords_grid = np.array([[g.x, g.y] for g in gdf_grid.geometry])
coords_estacoes = np.array([[g.x, g.y] for g in gdf_estacoes.geometry])
valores_estacoes = gdf_estacoes['dry_season_length'].values

# Árvore KDTree para busca rápida
tree = cKDTree(coords_estacoes)

max_dist_m = MAX_DISTANCE_KM * 1000
resultados = np.zeros(len(coords_grid))
vizinhos_count = np.zeros(len(coords_grid), dtype=int)

# Processamento em lote para não estourar memória
batch_size = 50000
num_batches = (len(coords_grid) // batch_size) + 1

print(f"   Interpolando {len(coords_grid):,} pontos em {num_batches} lotes...")

for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(coords_grid))
    if start >= end: break
    
    batch_coords = coords_grid[start:end]
    
    # Busca vizinhos
    dists, idxs = tree.query(batch_coords, k=MAX_NEIGHBORS, distance_upper_bound=max_dist_m)
    
    batch_vals = []
    batch_neighs = []
    
    for j in range(len(batch_coords)):
        # Filtra vizinhos válidos (não infinitos)
        valid = dists[j] != float('inf')
        d = dists[j][valid]
        ix = idxs[j][valid]
        
        if len(d) < MIN_NEIGHBORS:
            # Se não tem vizinhos, assume média regional (fallback)
            val = np.mean(valores_estacoes) 
            n = 0
        else:
            # IDW: Peso = 1 / distância^2
            # Adiciona valor minúsculo (1e-12) para evitar divisão por zero
            w = 1.0 / (d + 1e-12)**POWER
            w /= w.sum()
            val = np.dot(w, valores_estacoes[ix])
            n = len(d)
            
        batch_vals.append(val)
        batch_neighs.append(n)
        
    resultados[start:end] = batch_vals
    vizinhos_count[start:end] = batch_neighs
    
    sys.stdout.write(f"\r   Progresso: {end}/{len(coords_grid)} ({(end/len(coords_grid)*100):.1f}%)")

print(f"\n   ✅ Interpolação concluída.")

# ============================================================================
# 5. SALVAR RESULTADOS
# ============================================================================

print(f"\n{'='*100}")
print("💾 SALVANDO RESULTADOS")
print(f"{'='*100}")

gdf_grid['dry_season_length'] = np.round(resultados, 2)
gdf_grid['vizinhos_clima'] = vizinhos_count

# Converter para Lat/Lon para facilitar uso no QGIS/Web
gdf_final = gdf_grid.to_crs("EPSG:4326")

# Salvar GPKG
PATH_OUT_GPKG = DIR_OUTPUT / 'grid_norte_5km_clima_interpolado.gpkg'
gdf_final.to_file(PATH_OUT_GPKG, driver='GPKG')
print(f"✅ Grid GeoPackage salvo: {PATH_OUT_GPKG}")

# Salvar CSV Leve
PATH_OUT_CSV = DIR_OUTPUT / 'grid_norte_5km_clima_interpolado.csv'
df_csv = pd.DataFrame({
    'lon': gdf_final.geometry.x,
    'lat': gdf_final.geometry.y,
    'dry_season_length': gdf_final['dry_season_length']
})
df_csv.to_csv(PATH_OUT_CSV, index=False)
print(f"✅ CSV de coordenadas salvo: {PATH_OUT_CSV}")

print("\n📊 Estatísticas Finais:")
print(f"   Média Meses Secos: {df_csv['dry_season_length'].mean():.2f}")
print(f"   Máximo Meses Secos: {df_csv['dry_season_length'].max():.2f}")
print("=" * 100)
