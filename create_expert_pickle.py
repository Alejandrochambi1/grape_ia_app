# --- create_expert_pickle.py ---
import pickle
import json
import os
from expert_system_class import SistemaExpertoAvanzado # IMPORTA LA CLASE

MODELS_BASE_PATH = 'models' # Asegúrate que esta carpeta exista
os.makedirs(MODELS_BASE_PATH, exist_ok=True)

# Carga tus feature_names_list.json (necesario para instanciar la clase)
feature_names_path = os.path.join(MODELS_BASE_PATH, 'feature_names_list.json')
feature_names = None
try:
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    print(f"Nombres de características cargados desde {feature_names_path}")
except FileNotFoundError:
    print(f"ERROR: {feature_names_path} no encontrado. No se puede crear el pickle sin él.")
    # O usa tu lista de fallback si es solo para una prueba rápida,
    # pero asegúrate que el archivo real esté en tu repo para Render.
    # feature_names = ['lista', 'de', 'fallback']
    exit() # Salir si no se puede cargar el archivo esencial

if not feature_names:
    print("Error: feature_names está vacío después de intentar cargar.")
    exit()

# Crea una instancia del sistema experto
sistema_experto_instancia = SistemaExpertoAvanzado(feature_names)
print("Instancia de SistemaExpertoAvanzado creada.")

# Guarda la instancia en un archivo pickle
pickle_path = os.path.join(MODELS_BASE_PATH, 'sistema_experto_obj.pkl')
try:
    with open(pickle_path, 'wb') as f_pkl:
        pickle.dump(sistema_experto_instancia, f_pkl)
    print(f"Pickle '{pickle_path}' guardado exitosamente.")
    print("Este es el archivo que debes subir a GitHub.")
except Exception as e:
    print(f"Error al guardar el pickle: {e}")