#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de instalação - Modelo de Desmatamento
Verifica se todas as bibliotecas essenciais estão instaladas
"""

import sys
print("=" * 60)
print("TESTE DE INSTALAÇÃO - AMBIENTE PYTHON")
print("=" * 60)
print(f"\n🐍 Python: {sys.version}\n")

# Lista de bibliotecas para testar
libs_to_test = [
    ("geopandas", "GeoPandas"),
    ("rasterio", "Rasterio"),
    ("xarray", "xarray"),
    ("rioxarray", "rioxarray"),
    ("sklearn", "Scikit-learn"),
    ("xgboost", "XGBoost"),
    ("matplotlib", "Matplotlib"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("folium", "Folium"),
    ("plotly", "Plotly"),
    ("yaml", "PyYAML"),
    ("reportlab", "ReportLab"),
]

print("📦 TESTANDO BIBLIOTECAS:\n")

errors = []
for module_name, display_name in libs_to_test:
    try:
        __import__(module_name)
        print(f"  ✅ {display_name:20s} OK")
    except ImportError as e:
        print(f"  ❌ {display_name:20s} ERRO: {e}")
        errors.append(display_name)

print("\n" + "=" * 60)

if not errors:
    print("🎉 SUCESSO!   Todas as bibliotecas estão instaladas!")
    print("\n✅ Seu ambiente está PRONTO para o projeto!")
    print("\n📋 Próximos passos:")
    print("   1.  Iniciar JupyterLab: jupyter lab")
    print("   2. Começar a baixar dados (Fase 2)")
    print("   3.  Seguir o roadmap PDF")
else:
    print(f"⚠️  {len(errors)} biblioteca(s) com problema:")
    for lib in errors:
        print(f"   - {lib}")

print("=" * 60)