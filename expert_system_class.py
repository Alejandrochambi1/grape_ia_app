# --- START OF FILE expert_system_class.py ---
# import numpy as np # Solo si la clase lo usa directamente

class SistemaExpertoAvanzado:
    def __init__(self, feature_names):
        self.reglas = []
        self.feature_names = feature_names
        self.inicializar_reglas_avanzadas()

    def inicializar_reglas_avanzadas(self):
        # ... (tus reglas exactamente como las tienes) ...
        self.reglas.append({
            'nombre': 'Black Rot',
            'condiciones': {
                'black_pixel_ratio_min': 0.05, 'rgb_std_0_min': 40,
                'contrast_mean_min': 0.2, 'edge_density_canny_min': 0.08,
            }, 'confianza': 0.80
        })
        # ... (Añade TODAS tus reglas aquí) ...
        self.reglas.append({
            'nombre': 'Healthy',
            'condiciones': {
                'healthy_pixel_ratio_min': 0.35, 'rgb_mean_1_min': 90,
                'hsv_mean_0_min': 38, 'hsv_mean_0_max': 85,
                'homogeneity_mean_min': 0.6, 'num_regions_max': 40
            }, 'confianza': 0.85
        })


    def evaluar_condiciones(self, features_dict, condiciones_regla):
        # ... (tu lógica de evaluar_condiciones exactamente como la tienes) ...
        for param_key, valor_limite in condiciones_regla.items():
            feature_name_base = param_key.rsplit('_', 1)[0] if param_key.endswith(('_min', '_max')) else param_key
            if feature_name_base not in features_dict:
                return False
            valor_actual = features_dict[feature_name_base]
            if param_key.endswith('_min') and valor_actual < valor_limite: return False
            if param_key.endswith('_max') and valor_actual > valor_limite: return False
        return True

    def diagnosticar(self, features_list_ordered):
        # ... (tu lógica de diagnosticar exactamente como la tienes) ...
        if not self.feature_names:
            # print("Error SE: Nombres de características no inicializados en SistemaExperto.")
            return [{'enfermedad': 'Error Interno SE', 'confianza': 0.0, 'regla_activada': 'Config Error'}]
        if len(features_list_ordered) != len(self.feature_names):
            return [{'enfermedad': 'Error en Features SE', 'confianza': 0.1, 'regla_activada': 'Input Mismatch'}]

        features_dict = dict(zip(self.feature_names, features_list_ordered))
        resultados = []
        for regla in self.reglas:
            if self.evaluar_condiciones(features_dict, regla['condiciones']):
                resultados.append({'enfermedad': regla['nombre'], 'confianza': regla['confianza'], 'regla_activada': regla['nombre']})
        if not resultados:
            default_disease = 'Indeterminado'
            default_confidence = 0.20
            if features_dict.get('healthy_pixel_ratio', 0) > 0.35 and features_dict.get('black_pixel_ratio', 0) < 0.02 :
                 default_disease, default_confidence = 'Healthy', 0.45
            elif features_dict.get('black_pixel_ratio', 0) > 0.05:
                 default_disease, default_confidence = 'Black Rot', 0.40
            # ... (tu lógica de fallback completa) ...
            resultados.append({'enfermedad': default_disease, 'confianza': default_confidence, 'regla_activada': 'default_fallback_rule'})
        resultados.sort(key=lambda x: x['confianza'], reverse=True)
        return resultados
# --- END OF FILE expert_system_class.py ---