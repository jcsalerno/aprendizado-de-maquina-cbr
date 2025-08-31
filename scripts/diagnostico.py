import cbrkit.sim
import inspect

print("="*50)
print("Conteúdo de 'cbrkit.sim':")
print("="*50)

# Listar todos os nomes em cbrkit.sim
for name in dir(cbrkit.sim):
    # Ignorar membros privados/especiais
    if not name.startswith('_'):
        print(f"- {name}")

print("\n" + "="*50)
print("Tentando inspecionar submódulos comuns:")
print("="*50)

# Tentar inspecionar submódulos que já tentamos
submodules_to_check = ['collections', 'sets', 'helpers']

for mod_name in submodules_to_check:
    try:
        # Tenta importar o submódulo dinamicamente
        module = __import__(f"cbrkit.sim.{mod_name}", fromlist=[mod_name])
        print(f"\nConteúdo de 'cbrkit.sim.{mod_name}':")
        
        # Listar funções e classes dentro do submódulo
        found = False
        for name, obj in inspect.getmembers(module):
            if not name.startswith('_'):
                print(f"  - {name}")
                found = True
        if not found:
            print("  (Nenhum membro público encontrado)")

    except ImportError:
        print(f"\nFalha ao importar 'cbrkit.sim.{mod_name}'. O módulo não existe.")
    except Exception as e:
        print(f"\nOcorreu um erro ao inspecionar 'cbrkit.sim.{mod_name}': {e}")

