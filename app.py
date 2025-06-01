# --- START OF FILE app.py ---
import os
import json
import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
import cv2
import io # Necesario para np.frombuffer si lees de un stream en algún momento
import base64 # No se usa directamente aquí, pero podría ser útil para enviar imágenes

# Para características avanzadas (deben coincidir con las usadas en Colab)
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from skimage.segmentation import slic
from skimage.measure import regionprops # 'label' no se usa directamente en analisis_avanzado_pixeles_local
from skimage.morphology import disk, opening # 'closing' no se usa
from scipy import ndimage
from expert_system_class import SistemaExpertoAvanzado # ASUMIENDO que el archivo se llama expert_system_class.py

# --- Variables Globales y Configuración ---
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads' # Aunque no guardemos permanentemente, Flask podría usarlo para el objeto 'archivo'
MODELS_BASE_PATH = 'models'     # Ruta a la carpeta de modelos relativa a app.py

# Crear carpetas si no existen (útil para desarrollo local, Render las tendrá del repo)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# La carpeta 'models' debería existir en tu repositorio y ser copiada por Render

modelo_cnn_global = None
modelo_arbol_decision_global = None
sistema_experto_global = None
feature_scaler_global = None
mapeo_clases_global = None
feature_names_list_global = None


def cargar_todos_los_modelos():
    global modelo_cnn_global, modelo_arbol_decision_global, sistema_experto_global, \
           feature_scaler_global, mapeo_clases_global, feature_names_list_global
    
    print("Iniciando carga de modelos y artefactos...")
    models_loaded_status = {'cnn': False, 'tree': False, 'expert': False, 'scaler': False, 'mapeo': False, 'features_names': False}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_path_abs = os.path.join(current_dir, MODELS_BASE_PATH)
    print(f"Buscando modelos en la ruta base absoluta: {models_path_abs}")

    try:
        # Nombres de Características
        features_path = os.path.join(models_path_abs, "feature_names_list.json")
        print(f"Intentando cargar feature_names_list desde: {features_path}")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                feature_names_list_global = json.load(f)
            print("✓ Nombres de características cargados.")
            models_loaded_status['features_names'] = True
        else:
            print(f"⚠️ FALLO: Archivo feature_names_list.json NO encontrado en {features_path}.")
            # ... (tu fallback)

        # Mapeo de Clases
        mapeo_path = os.path.join(models_path_abs, "mapeo_clases.json")
        print(f"Intentando cargar mapeo_clases desde: {mapeo_path}")
        if os.path.exists(mapeo_path):
            with open(mapeo_path, 'r') as f:
                mapeo_clases_global = json.load(f)
            mapeo_clases_global = {int(k) if k.isdigit() else k: v for k, v in mapeo_clases_global.items()}
            print("✓ Mapeo de Clases cargado.")
            models_loaded_status['mapeo'] = True
        else:
            print(f"⚠️ FALLO: Archivo mapeo_clases.json NO encontrado en {mapeo_path}.")
            # ... (tu fallback)


        # Sistema Experto (DESPUÉS de feature_names y mapeo)
        sistema_experto_path = os.path.join(models_path_abs, "sistema_experto_obj.pkl")
        print(f"Intentando cargar sistema_experto_obj desde: {sistema_experto_path}")
        if os.path.exists(sistema_experto_path):
             with open(sistema_experto_path, 'rb') as f:
                sistema_experto_global = pickle.load(f)
             if hasattr(sistema_experto_global, 'feature_names') and feature_names_list_global:
                sistema_experto_global.feature_names = feature_names_list_global # Asegurar que tenga los feature_names correctos
             print("✓ Sistema Experto cargado desde archivo.")
             models_loaded_status['expert'] = True
        elif feature_names_list_global: 
            sistema_experto_global = SistemaExpertoAvanzado(feature_names_list_global)
            print("✓ Sistema Experto inicializado (sin .pkl, usando clase y feature_names).")
            models_loaded_status['expert'] = True
        else:
            print(f"⚠️ FALLO: Sistema Experto no pudo ser cargado (no existe {sistema_experto_path}) ni inicializado (faltan feature_names).")


        # Modelo CNN
        cnn_path = os.path.join(models_path_abs, "modelo_cnn_mejorado.keras")
        print(f"Intentando cargar modelo_cnn_mejorado desde: {cnn_path}")
        if os.path.exists(cnn_path):
            modelo_cnn_global = load_model(cnn_path)
            print("✓ Modelo CNN cargado.")
            models_loaded_status['cnn'] = True
        else:
            print(f"⚠️ FALLO: Archivo modelo_cnn_mejorado.keras NO encontrado en {cnn_path}.")

        # Árbol de Decisión
        tree_path = os.path.join(models_path_abs, "modelo_arbol_decision.pkl")
        print(f"Intentando cargar modelo_arbol_decision desde: {tree_path}")
        if os.path.exists(tree_path):
            with open(tree_path, 'rb') as f:
                modelo_arbol_decision_global = pickle.load(f)
            print("✓ Árbol de Decisión cargado.")
            models_loaded_status['tree'] = True
        else:
            print(f"⚠️ FALLO: Archivo modelo_arbol_decision.pkl NO encontrado en {tree_path}.")

        # Scaler
        scaler_path = os.path.join(models_path_abs, "feature_scaler.pkl")
        print(f"Intentando cargar feature_scaler desde: {scaler_path}")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                feature_scaler_global = pickle.load(f)
            print("✓ Scaler de características cargado.")
            models_loaded_status['scaler'] = True
        else:
            print(f"⚠️ FALLO: Archivo feature_scaler.pkl NO encontrado en {scaler_path}.")
        
        # ... (resto de la función) ...
    except Exception as e:
        print(f"Error crítico durante la carga de modelos: {e}")
        import traceback
        traceback.print_exc() # Imprime el traceback completo para más detalles
    
    # ... (impresión del estado final) ...
    return models_loaded_status

def analisis_avanzado_pixeles_local(img_array_rgb, target_size=(256, 256)):
    try:
        img_rgb = cv2.resize(img_array_rgb, target_size)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Para LAB, OpenCV espera BGR. Si ya tienes RGB, puedes convertir RGB->BGR->LAB o RGB->XYZ->LAB
        # Opción 1: Convertir a BGR primero
        img_bgr_temp = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr_temp, cv2.COLOR_BGR2LAB)

        color_stats = {
            'rgb_mean': np.mean(img_rgb, axis=(0, 1)), 'rgb_std': np.std(img_rgb, axis=(0, 1)),
            'hsv_mean': np.mean(img_hsv, axis=(0, 1)), 'hsv_std': np.std(img_hsv, axis=(0, 1)),
            'lab_mean': np.mean(img_lab, axis=(0, 1)), 'lab_std': np.std(img_lab, axis=(0, 1)),
        }
        
        black_pixels = np.sum((img_rgb[:,:,0] < 50) & (img_rgb[:,:,1] < 50) & (img_rgb[:,:,2] < 50))
        black_ratio = black_pixels / (img_rgb.shape[0] * img_rgb.shape[1])
        esca_mask = ((img_hsv[:,:,0] >= 10) & (img_hsv[:,:,0] <= 30) & (img_hsv[:,:,1] >= 50) & (img_hsv[:,:,2] >= 100) & (img_hsv[:,:,2] <= 200))
        esca_ratio = np.sum(esca_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
        blight_mask = ((img_hsv[:,:,0] <= 20) & (img_hsv[:,:,1] >= 100) & (img_hsv[:,:,2] >= 50) & (img_hsv[:,:,2] <= 150))
        blight_ratio = np.sum(blight_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
        healthy_mask = ((img_hsv[:,:,0] >= 40) & (img_hsv[:,:,0] <= 80) & (img_hsv[:,:,1] >= 50) & (img_hsv[:,:,2] >= 50))
        healthy_ratio = np.sum(healthy_mask) / (img_rgb.shape[0] * img_rgb.shape[1])
        
        glcm_features = {}
        glcm_props_list = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        try:
            glcm = graycomatrix(img_gray, distances=[1, 3], angles=[0, np.pi/2], levels=256, symmetric=True, normed=True)
            for prop in glcm_props_list:
                glcm_features[f'{prop}_mean'] = np.mean(graycoprops(glcm, prop))
        except ValueError: # A veces la imagen es muy uniforme y GLCM falla
             for prop in glcm_props_list: glcm_features[f'{prop}_mean'] = 0.0

        radius = 3; n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
        
        gabor_responses = []
        gabor_feature_names_temp = []
        idx_gabor = 0
        for theta_deg in [0, 45, 90, 135]:
            for frequency in [0.1, 0.5]:
                try:
                    real, _ = gabor(img_gray, frequency=frequency, theta=np.radians(theta_deg))
                    gabor_responses.extend([np.mean(np.abs(real)), np.std(real)])
                except ValueError:
                     gabor_responses.extend([0.0, 0.0])
                gabor_feature_names_temp.append(f'gabor_{idx_gabor}')
                idx_gabor+=1
                gabor_feature_names_temp.append(f'gabor_{idx_gabor}')
                idx_gabor+=1

        edges_canny = cv2.Canny(img_gray, 50, 150)
        edge_density_canny = np.sum(edges_canny > 0) / (edges_canny.shape[0] * edges_canny.shape[1])
        kernel = disk(3)
        img_opened = opening(img_gray, kernel)
        opening_diff = np.mean(np.abs(img_gray.astype(float) - img_opened.astype(float)))
        
        try:
            segments = slic(img_rgb, n_segments=30, compactness=10, sigma=1, start_label=1, channel_axis=-1, enforce_connectivity=True)
            regions = regionprops(segments, intensity_image=img_gray)
            region_areas = [r.area for r in regions]
            region_stats = {'mean_area': np.mean(region_areas) if region_areas else 0, 'num_regions': len(regions)}
        except Exception as e_slic: # SLIC puede fallar con ciertas imágenes
            # print(f"Advertencia: SLIC falló: {e_slic}. Usando valores por defecto para region_stats.")
            region_stats = {'mean_area': 0, 'num_regions': 1}


        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        spectral_energy = np.sum(magnitude_spectrum**2)
        
        features_dict = {
            **{f'rgb_mean_{i}': color_stats['rgb_mean'][i] for i in range(3)},
            **{f'rgb_std_{i}': color_stats['rgb_std'][i] for i in range(3)},
            **{f'hsv_mean_{i}': color_stats['hsv_mean'][i] for i in range(3)},
            **{f'hsv_std_{i}': color_stats['hsv_std'][i] for i in range(3)},
            **{f'lab_mean_{i}': color_stats['lab_mean'][i] for i in range(3)},
            **{f'lab_std_{i}': color_stats['lab_std'][i] for i in range(3)},
            'black_pixel_ratio': black_ratio, 'esca_pixel_ratio': esca_ratio,
            'blight_pixel_ratio': blight_ratio, 'healthy_pixel_ratio': healthy_ratio,
            **glcm_features,
            **{f'lbp_{i}': lbp_hist[i] for i in range(min(10, len(lbp_hist)))}, # Asegurar que no exceda 10
            **{f'gabor_{i}': gabor_responses[i] for i in range(min(len(gabor_feature_names_temp), len(gabor_responses), 10))}, # Asegurar que no exceda 10
            'edge_density_canny': edge_density_canny, 'opening_diff': opening_diff,
            **region_stats, 'spectral_energy': spectral_energy
        }
        
        if feature_names_list_global:
            ordered_feature_values = [features_dict.get(name, 0.0) for name in feature_names_list_global]
        else:
            print("Error: feature_names_list_global no está disponible para ordenar características.")
            # Intentar un orden por defecto, pero esto es propenso a errores
            ordered_feature_values = list(features_dict.values())[:len(feature_names_list_global or [])]


        return features_dict, np.array([ordered_feature_values], dtype=np.float32)

    except Exception as e:
        print(f"Error severo en analisis_avanzado_pixeles_local: {e}")
        num_expected_features = len(feature_names_list_global) if feature_names_list_global else 50 # Estimado
        return {name: 0.0 for name in (feature_names_list_global or [])}, np.zeros((1, num_expected_features), dtype=np.float32)


def generar_recomendaciones_api(enfermedad_nombre_str):
    recomendaciones = {
        "Black Rot": ["Eliminar y destruir frutos y cancros infectados.", "Aplicar fungicidas protectores (ej. mancozeb, captan, o estrobilurinas) desde el brote hasta la cosecha según indicación.", "Mejorar la circulación de aire en el dosel mediante poda adecuada."],
        "Esca (Black Measles)": ["Realizar poda sanitaria para remover madera muerta o infectada, desinfectando herramientas.", "Proteger heridas grandes de poda con pasta cicatrizante.", "Considerar tratamientos preventivos con productos a base de Trichoderma o arsenito de sodio (donde sea legal y con precaución). Evitar estrés hídrico y nutricional."],
        "Leaf Blight (Isariopsis Leaf Spot)": ["Aplicar fungicidas protectores como mancozeb o productos a base de cobre al inicio de los síntomas.", "Mejorar el drenaje del suelo y la circulación de aire.", "Eliminar y destruir hojas caídas y restos de poda infectados al final de la temporada."],
        "Healthy": ["Continuar con buenas prácticas de manejo del viñedo.", "Realizar monitoreo regular para detectar cualquier signo temprano de enfermedad o plaga.", "Mantener un buen equilibrio nutricional y riego adecuado."]
    }
    # Buscar la enfermedad por coincidencia parcial para más flexibilidad
    for key_disease, recs in recomendaciones.items():
        if enfermedad_nombre_str.lower().startswith(key_disease.lower()):
            return recs
    return ["No se encontraron recomendaciones específicas. Consulte a un agrónomo para un plan de manejo detallado."]


def sistema_integrado_api_local(img_array_rgb):
    global mapeo_clases_global, feature_names_list_global, modelo_cnn_global, \
           modelo_arbol_decision_global, sistema_experto_global, feature_scaler_global

    if not mapeo_clases_global or not feature_names_list_global:
        print("Error crítico: Mapeo de clases o nombres de características no cargados.")
        return {'error': 'Configuración interna del sistema incompleta. Contacte al administrador.'}

    num_clases = len(mapeo_clases_global)
    features_dict, features_array_ordered = analisis_avanzado_pixeles_local(img_array_rgb)

    if features_array_ordered.size == 0 or features_array_ordered.shape[1] != len(feature_names_list_global):
         print(f"Error en la forma de las características extraídas. Esperado: (1, {len(feature_names_list_global)}), Obtenido: {features_array_ordered.shape}")
         return {'error': 'Fallo en la extracción o formato de características de la imagen.'}

    # CNN
    clase_idx_cnn, confianza_cnn, pred_probs_cnn_dict = -1, 0.0, {mapeo_clases_global.get(i, f"Clase {i}"): 0.0 for i in range(num_clases)}
    pred_probs_cnn_vector = np.zeros(num_clases)

    if modelo_cnn_global:
        img_cnn_resized = cv2.resize(img_array_rgb, (224, 224))
        img_cnn_array = img_to_array(img_cnn_resized)
        img_cnn_array_expanded = np.expand_dims(img_cnn_array, axis=0)
        img_cnn_processed = efficientnet_preprocess_input(img_cnn_array_expanded)
        pred_probs_cnn_vector = modelo_cnn_global.predict(img_cnn_processed, verbose=0)[0]
        clase_idx_cnn = np.argmax(pred_probs_cnn_vector)
        confianza_cnn = float(pred_probs_cnn_vector[clase_idx_cnn])
        pred_probs_cnn_dict = {mapeo_clases_global.get(i, f"Clase {i}"): float(pred_probs_cnn_vector[i]) for i in range(num_clases)}
    else: # Simulación
        print("ADVERTENCIA: Modelo CNN no cargado, simulando predicción.")
        clase_idx_cnn = np.random.randint(0, num_clases)
        confianza_cnn = np.random.uniform(0.6, 0.9)
        pred_probs_cnn_vector = np.random.dirichlet(np.ones(num_clases), size=1)[0] # Suma 1
        pred_probs_cnn_vector[clase_idx_cnn] = confianza_cnn # Ajustar para que la confianza coincida
        pred_probs_cnn_vector /= np.sum(pred_probs_cnn_vector) # Renormalizar si es necesario
        pred_probs_cnn_dict = {mapeo_clases_global.get(i, f"Clase {i}"): float(pred_probs_cnn_vector[i]) for i in range(num_clases)}


    # Árbol de Decisión
    clase_idx_dt, confianza_dt, pred_probs_dt_dict = -1, 0.0, {mapeo_clases_global.get(i, f"Clase {i}"): 0.0 for i in range(num_clases)}
    pred_probs_dt_vector = np.zeros(num_clases)

    if modelo_arbol_decision_global and feature_scaler_global:
        if features_array_ordered.shape[1] == feature_scaler_global.n_features_in_:
            features_scaled = feature_scaler_global.transform(features_array_ordered)
            pred_probs_dt_vector = modelo_arbol_decision_global.predict_proba(features_scaled)[0]
            clase_idx_dt = np.argmax(pred_probs_dt_vector)
            confianza_dt = float(pred_probs_dt_vector[clase_idx_dt])
            pred_probs_dt_dict = {mapeo_clases_global.get(i, f"Clase {i}"): float(pred_probs_dt_vector[i]) for i in range(num_clases)}
        else:
            print(f"ADVERTENCIA: Discrepancia en features para DT. Esperado: {feature_scaler_global.n_features_in_}, Obtenido: {features_array_ordered.shape[1]}. Simulando DT.")
            # ... (simulación similar a CNN)
            clase_idx_dt = np.random.randint(0, num_clases)
            confianza_dt = np.random.uniform(0.5, 0.8)
            pred_probs_dt_vector = np.random.dirichlet(np.ones(num_clases), size=1)[0]
            pred_probs_dt_vector[clase_idx_dt] = confianza_dt
            pred_probs_dt_vector /= np.sum(pred_probs_dt_vector)
            pred_probs_dt_dict = {mapeo_clases_global.get(i, f"Clase {i}"): float(pred_probs_dt_vector[i]) for i in range(num_clases)}
    else:
        print("ADVERTENCIA: Modelo DT o Scaler no cargado, simulando predicción DT.")
        # ... (simulación similar a CNN)
        clase_idx_dt = np.random.randint(0, num_clases)
        confianza_dt = np.random.uniform(0.5, 0.8)
        pred_probs_dt_vector = np.random.dirichlet(np.ones(num_clases), size=1)[0]
        pred_probs_dt_vector[clase_idx_dt] = confianza_dt
        pred_probs_dt_vector /= np.sum(pred_probs_dt_vector)
        pred_probs_dt_dict = {mapeo_clases_global.get(i, f"Clase {i}"): float(pred_probs_dt_vector[i]) for i in range(num_clases)}


    # Sistema Experto
    clase_nombre_es, confianza_es, clase_idx_es = "Indeterminado (SE)", 0.0, -1
    regla_activada_es = "N/A"
    if sistema_experto_global:
        diagnostico_es_list = sistema_experto_global.diagnosticar(features_array_ordered.flatten().tolist())
        if diagnostico_es_list:
            diagnostico_es_principal = diagnostico_es_list[0]
            clase_nombre_es = diagnostico_es_principal['enfermedad']
            confianza_es = float(diagnostico_es_principal['confianza'])
            regla_activada_es = diagnostico_es_principal.get('regla_activada', 'N/A')
            for idx_map, nombre_map_val in mapeo_clases_global.items():
                if nombre_map_val.lower() in clase_nombre_es.lower(): # Coincidencia más flexible
                    clase_idx_es = idx_map
                    break
    else:
        print("ADVERTENCIA: Sistema Experto no cargado, simulando predicción SE.")
        clase_idx_es = np.random.randint(0, num_clases)
        confianza_es = np.random.uniform(0.4, 0.7)
        clase_nombre_es = mapeo_clases_global.get(clase_idx_es, "Simulado_SE")
        regla_activada_es = "simulada_default"


    # Votación ponderada
    votos = np.zeros(num_clases)
    # Pesos base
    w_cnn, w_dt, w_es = 0.60, 0.25, 0.15
    
    # Ajustar pesos si algún modelo no está disponible
    active_models_weight_sum = 0
    if modelo_cnn_global: active_models_weight_sum += w_cnn
    if modelo_arbol_decision_global and feature_scaler_global and features_array_ordered.shape[1] == feature_scaler_global.n_features_in_:
        active_models_weight_sum += w_dt
    if sistema_experto_global: active_models_weight_sum += w_es

    if active_models_weight_sum == 0: # Ningún modelo real cargado, todos simulados o fallaron
        print("Error: Ningún modelo de IA está operativo. Diagnóstico imposible.")
        # Devolver un error o un resultado indeterminado muy bajo
        clase_final_idx = -1 # Indicar fallo
        confianza_final_score = 0.0
        clase_final_nombre_str = "Error en Modelos"
    else:
        # CNN
        if modelo_cnn_global:
            votos += pred_probs_cnn_vector * (w_cnn / active_models_weight_sum)
        # DT
        if modelo_arbol_decision_global and feature_scaler_global and clase_idx_dt != -1 and features_array_ordered.shape[1] == feature_scaler_global.n_features_in_:
             # Aquí usamos el vector de probabilidades del DT
            votos += pred_probs_dt_vector * (w_dt / active_models_weight_sum)
        # ES
        if sistema_experto_global and clase_idx_es != -1:
            # El SE da una clase y confianza, no un vector de probs. Lo añadimos a su clase.
            prob_es_vector = np.zeros(num_clases)
            prob_es_vector[clase_idx_es] = confianza_es
            votos += prob_es_vector * (w_es / active_models_weight_sum)
            # Alternativamente, si el SE es muy fuerte en una regla, podría tener más peso.
            # Por ahora, usamos la confianza directa.

        clase_final_idx = np.argmax(votos)
        confianza_final_score = float(votos[clase_final_idx])
        clase_final_nombre_str = mapeo_clases_global.get(clase_final_idx, "Indeterminado Final")
    
    recomendaciones_finales = generar_recomendaciones_api(clase_final_nombre_str)
    
    # Para el frontend, asegurarse que las probabilidades sean listas de Python, no arrays NumPy
    cnn_probs_for_json = {k: float(v) for k,v in pred_probs_cnn_dict.items()}
    dt_probs_for_json = {k: float(v) for k,v in pred_probs_dt_dict.items()}


    return {
        'clase_final': clase_final_nombre_str,
        'confianza_final': confianza_final_score,
        'resultados_individuales': {
            'cnn': {'clase': mapeo_clases_global.get(clase_idx_cnn, "N/A"), 'confianza': confianza_cnn, 'probabilidades': cnn_probs_for_json},
            'arbol': {'clase': mapeo_clases_global.get(clase_idx_dt, "N/A"), 'confianza': confianza_dt, 'probabilidades': dt_probs_for_json},
            'experto': {'clase': clase_nombre_es, 'confianza': confianza_es, 'regla_activada': regla_activada_es}
        },
        'caracteristicas_clave': {
            k: float(features_dict.get(k, 0.0)) if isinstance(features_dict.get(k, 0.0), (np.float32, np.float64)) else features_dict.get(k, 0.0)
            for k in ['healthy_pixel_ratio', 'black_pixel_ratio', 'edge_density_canny', 'num_regions'][:4] # Solo algunas para mostrar
        },
        'recomendaciones': recomendaciones_finales
    }

# --- Rutas Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analizar', methods=['POST'])
def analizar_imagen_endpoint():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
    
    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    try:
        imagen_bytes = archivo.read()
        nparr = np.frombuffer(imagen_bytes, np.uint8)
        img_cv2_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2_bgr is None:
            return jsonify({'error': 'No se pudo decodificar la imagen. Formato no válido o archivo corrupto.'}), 400
        
        img_cv2_rgb = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2RGB)
        
        print("Procesando imagen con sistema integrado...")
        resultado_completo = sistema_integrado_api_local(img_cv2_rgb)
        
        if 'error' in resultado_completo:
            print(f"Error devuelto por sistema_integrado: {resultado_completo['error']}")
            return jsonify(resultado_completo), 500 # Devolver error del sistema si ocurrió
            
        print(f"Diagnóstico final: {resultado_completo.get('clase_final', 'N/A')}")
        return jsonify(resultado_completo)

    except Exception as e:
        print(f"Excepción general en /analizar: {e}")
        # Loggear el traceback completo para depuración
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ocurrió un error inesperado en el servidor: {str(e)}'}), 500

# --- Inicialización de la Aplicación ---
# Cargar modelos una vez al iniciar la aplicación
# Esto es crucial para el rendimiento en producción (ej. con Gunicorn)
status_carga_app_global = {}
try:
    print("==================================================")
    print("INICIANDO APLICACIÓN FLASK GRAPEAI")
    print("Cargando modelos y artefactos al inicio del servidor...")
    print("==================================================")
    status_carga_app_global = cargar_todos_los_modelos()
    print("--------------------------------------------------")
    print(f"ESTADO FINAL DE CARGA DE MODELOS: {status_carga_app_global}")
    print("--------------------------------------------------")
    
    if not mapeo_clases_global:
        print("¡ADVERTENCIA CRÍTICA AL INICIO! El mapeo de clases no se cargó.")
    if not feature_names_list_global:
        print("¡ADVERTENCIA CRÍTICA AL INICIO! La lista de nombres de características no se cargó.")
    if not modelo_cnn_global:
        print("ADVERTENCIA AL INICIO: Modelo CNN no cargado.")
    if not modelo_arbol_decision_global:
        print("ADVERTENCIA AL INICIO: Modelo Árbol de Decisión no cargado.")
    if not sistema_experto_global:
        print("ADVERTENCIA AL INICIO: Sistema Experto no cargado/inicializado.")
    if not feature_scaler_global:
        print("ADVERTENCIA AL INICIO: Scaler no cargado.")

except Exception as e_init:
    print(f"Error FATAL durante la inicialización de la aplicación (carga de modelos): {e_init}")
    import traceback
    traceback.print_exc()
    # En un entorno de producción real, podrías querer que la app no inicie si los modelos no cargan.
    # Por ahora, permitimos que inicie para ver los logs en Render.

if __name__ == '__main__':
    # Esta parte solo se ejecuta cuando corres `python app.py` directamente
    # Gunicorn/Render no ejecutarán esto, importarán el objeto `app`
    print("Ejecutando Flask app en modo desarrollo local...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

# --- END OF FILE app.py ---