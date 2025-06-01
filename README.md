# GrapeAI - Sistema de Detección de Enfermedades en Uvas

## Descripción
Sistema inteligente que integra tres técnicas de IA para detectar enfermedades en plantas de uva:
- Redes Neuronales Convolucionales (CNN)
- Árboles de Decisión
- Sistemas Expertos

## Estructura del Proyecto
\`\`\`
grape-disease-flask-app/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias
├── README.md             # Este archivo
├── templates/
│   └── index.html        # Plantilla HTML principal
├── static/
│   ├── css/
│   │   └── style.css     # Estilos CSS
│   ├── js/
│   │   └── script.js     # JavaScript del frontend
│   └── uploads/          # Carpeta para imágenes subidas
└── models/               # Carpeta para modelos entrenados (opcional)
    ├── modelo_cnn_avanzado.keras
    ├── arbol_decision_avanzado.pkl
    ├── sistema_experto_avanzado.pkl
    ├── scaler_avanzado.pkl
    ├── mapeo_clases_avanzado.json
    └── feature_names.json
\`\`\`

## Instalación

1. Crear entorno virtual:
\`\`\`bash
py -3.11 -m venv venv
.\venv\Scripts\activate
\`\`\`

2. Instalar dependencias:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Ejecutar la aplicación:
\`\`\`bash
python app.py
\`\`\`

4. Abrir navegador en: http://localhost:5000

## Características

### Sistema Funcional Sin Modelos
- El sistema funciona incluso sin los modelos entrenados
- Utiliza simulaciones inteligentes basadas en características de imagen
- Sistema experto siempre funcional con reglas codificadas

### Análisis de Imágenes
- Análisis de píxeles por color (RGB, HSV, LAB)
- Detección de patrones específicos por enfermedad
- Análisis de textura y bordes
- Segmentación de regiones

### Integración de Técnicas
- Votación ponderada adaptativa
- Consenso entre sistemas
- Explicabilidad de resultados

## Uso

1. Subir imagen de hoja de uva
2. El sistema analiza automáticamente con las tres técnicas
3. Muestra diagnóstico integrado con visualizaciones
4. Proporciona recomendaciones específicas

## Enfermedades Detectadas
- **Black Rot**: Manchas negras circulares
- **Esca**: Manchas amarillentas/marrones irregulares  
- **Leaf Blight**: Manchas rojizas con halos
- **Healthy**: Hoja sana

## Tecnologías
- **Backend**: Flask, TensorFlow, OpenCV, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Particles.js
- **IA**: CNN (EfficientNet), Árboles de Decisión, Sistemas Expertos

## Notas
- El sistema está diseñado para funcionar con o sin modelos pre-entrenados
- Incluye sistema de demostración para pruebas
- Interfaz responsive y moderna
- API REST para integración
\`\`\`
