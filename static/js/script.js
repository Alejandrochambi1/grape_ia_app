document.addEventListener("DOMContentLoaded", () => {
    if (window.particlesJS) {
        particlesJS("particles-js", {
            "particles": {
                "number": {
                    "value": 70, 
                    "density": {
                        "enable": true,
                        "value_area": 800 
                    }
                },
                "color": {
                    "value": "#FFECB3" // Amarillo claro (var(--secondary-light))
                },
                "shape": {
                    "type": "circle",
                    "stroke": {
                        "width": 0,
                        "color": "#000000"
                    },
                },
                "opacity": {
                    "value": 0.7, 
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 0.7,
                        "opacity_min": 0.1,
                        "sync": false
                    }
                },
                "size": {
                    "value": 3, 
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 2.5,
                        "size_min": 0.3,
                        "sync": false
                    }
                },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#FFECB3", // Amarillo claro
                    "opacity": 0.25, // Líneas más sutiles
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 4, 
                    "direction": "none",
                    "random": true,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {
                        "enable": true,
                        "rotateX": 500,
                        "rotateY": 1000
                    }
                }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": {
                    "onhover": {
                        "enable": true,
                        "mode": "repulse" 
                    },
                    "onclick": {
                        "enable": true,
                        "mode": "push" 
                    },
                    "resize": true
                },
                "modes": {
                    "grab": {
                        "distance": 180,
                        "line_linked": {
                            "opacity": 1
                        }
                    },
                    "bubble": {
                        "distance": 220,
                        "size": 7,
                        "duration": 2,
                        "opacity": 0.8,
                    },
                    "repulse": {
                        "distance": 120,
                        "duration": 0.4
                    },
                    "push": {
                        "particles_nb": 3
                    },
                    "remove": {
                        "particles_nb": 2
                    }
                }
            },
            "retina_detect": true
        });
    } else {
        console.warn("Particles.js no está cargado. El fondo animado no funcionará.");
    }

    initializeApp();
    setupEventListeners();
    setupScrollEffects();
});

let currentImageFile = null; 
let currentAnalysisResults = null; 

function initializeApp() {
  setupSmoothScrolling();
  setupDragAndDrop();
  setupIntersectionObserver();
  setActiveNavLink(); 
}

function setupEventListeners() {
  const imageInput = document.getElementById("imageInput");
  if (imageInput) imageInput.addEventListener("change", handleImageSelect);

  const analyzeBtn = document.getElementById("analyzeBtn");
  if (analyzeBtn) analyzeBtn.addEventListener("click", analyzeImage);

  const techItems = document.querySelectorAll(".tech-item");
  techItems.forEach((item) => {
    item.addEventListener("click", () => showTechInfo(item.dataset.tech));
  });
}

function setupScrollEffects() {
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;
  window.addEventListener("scroll", () => {
    if (window.scrollY > 80) { 
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
    setActiveNavLink(); 
  });
}

function setActiveNavLink() {
    let currentSection = "";
    const sections = document.querySelectorAll("section[id]");
    const navLinks = document.querySelectorAll(".nav-link");

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (pageYOffset >= sectionTop - 100) { 
            currentSection = section.getAttribute("id");
        }
    });

    navLinks.forEach(link => {
        link.classList.remove("active");
        if (link.getAttribute("href") === `#${currentSection}`) {
            link.classList.add("active");
        }
    });
}


function setupSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
          block: "start", 
        });
        document.querySelectorAll(".nav-link").forEach(link => link.classList.remove("active"));
        this.classList.add("active");
      }
    });
  });
}

function setupDragAndDrop() {
  const uploadArea = document.getElementById("uploadArea");
  if (!uploadArea) return;

  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    uploadArea.addEventListener(eventName, () => uploadArea.classList.add("dragover"));
  });

  ["dragleave", "drop"].forEach((eventName) => {
    uploadArea.addEventListener(eventName, () => uploadArea.classList.remove("dragover"));
  });

  uploadArea.addEventListener("drop", handleDrop);
}

function setupIntersectionObserver() {
  const observerOptions = { threshold: 0.15, rootMargin: "0px 0px -50px 0px" }; 
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("fade-in-up"); 
        observer.unobserve(entry.target); 
      }
    });
  }, observerOptions);

  document.querySelectorAll(".about-card, .disease-card, .system-result, .feature-item, .recommendation-item").forEach((el) => {
    observer.observe(el);
  });
}


function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    handleImageFile(files[0]);
  }
}

function handleImageSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleImageFile(file);
  }
}

function handleImageFile(file) {
  if (!file.type.startsWith("image/")) {
    showNotification("Por favor, selecciona un archivo de imagen válido (JPG, PNG, etc.).", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) { 
    showNotification("La imagen es demasiado grande (máx. 10MB).", "error");
    return;
  }

  currentImageFile = file; 

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById("previewImg");
    previewImg.src = e.target.result;

    document.getElementById("uploadArea").style.display = "none";
    document.getElementById("imagePreview").style.display = "flex"; 

    updateStatus("Imagen cargada. Lista para analizar.", "active");
    document.getElementById("results").style.display = "none";
  };
  reader.readAsDataURL(file);
}

function changeImage() {
  document.getElementById("uploadArea").style.display = "block";
  document.getElementById("imagePreview").style.display = "none";
  const imageInput = document.getElementById("imageInput");
  if (imageInput) imageInput.value = ""; 
  currentImageFile = null;
  document.getElementById("previewImg").src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZWVlZmYxIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJQb3BwaW5zLCBBcmlhbCIgZm9udC1zaXplPSIxNnB4IiBmaWxsPSIjYWRhZGIzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Vista Previa</text></svg>";
  updateStatus("Esperando imagen...", "waiting");

  document.getElementById("analysisProgress").style.display = "none";
  document.getElementById("featuresViz").style.display = "none";
  document.getElementById("results").style.display = "none";
}

async function analyzeImage() {
  if (!currentImageFile) {
    showNotification("Por favor, selecciona una imagen primero.", "error");
    return;
  }

  showAnalysisProgress(); 

  const formData = new FormData();
  formData.append("imagen", currentImageFile);

  try {
    updateStatus("Enviando imagen...", "processing", 1); 
    const response = await fetch("/analizar", { method: "POST", body: formData });
    
    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch (e) {
        errorData = { error: `Error del servidor: ${response.status}` };
      }
      throw new Error(errorData.error || `Error HTTP: ${response.status}`);
    }

    const data = await response.json();
    if (data.error) { 
      throw new Error(data.error);
    }
    currentAnalysisResults = data; 
    processAnalysisStepsWithRealData(data); 
  } catch (error) {
    console.error("Error en el análisis:", error);
    showNotification(`Error al analizar la imagen: ${error.message}`, "error");
    updateStatus("Error en análisis", "error", 5); 
  }
}

function processAnalysisStepsWithRealData(realData) {
  const stepsInfo = [
    { step: 1, message: "Imagen cargada", delay: 500 },
    { step: 2, message: "Ejecutando red neuronal...", delay: 1200, action: () => simulateFeatureExtractionWithRealData(realData) },
    { step: 3, message: "Analizando con árbol de decisión...", delay: 1000 },
    { step: 4, message: "Consultando sistema experto...", delay: 1000 },
    { step: 5, message: "Generando resultado final...", delay: 800 }
  ];

  let currentStepIndex = 0;

  function executeNextStep() {
    if (currentStepIndex < stepsInfo.length) {
      const { step, message, delay, action } = stepsInfo[currentStepIndex];
      
      updateStatus(message, "processing", step);
      if (action) action();

      setTimeout(() => {
        if (currentStepIndex < stepsInfo.length -1) { 
           const currentStepEl = document.querySelector(`.progress-step[data-step="${step}"]`);
           if (currentStepEl) {
               currentStepEl.classList.remove("active");
               currentStepEl.classList.add("completed");
           }
        }
        currentStepIndex++;
        executeNextStep();
      }, delay);
    } else {
      completeAnalysisWithRealData(realData);
    }
  }
  executeNextStep();
}


function simulateFeatureExtractionWithRealData(realData) {
  const features = realData.caracteristicas_clave || {
    black_pixel_ratio: 0,
    healthy_pixel_ratio: 0, 
    edge_density_canny: 0, 
    num_regions: 0,    
  };

  animateFeatureBar("blackPixels", (features.black_pixel_ratio || 0) * 100);
  animateFeatureBar("greenPixels", (features.healthy_pixel_ratio || 0) * 100);
  animateFeatureBar("edgeDensity", (features.edge_density_canny || 0) * 100); 
  const textureValue = Math.min(100, (features.num_regions || 0) * 2); 
  animateFeatureBar("texture", textureValue);
}

function completeAnalysisWithRealData(realData) {
  const lastStepEl = document.querySelector(`.progress-step[data-step="5"]`);
  if (lastStepEl) {
    lastStepEl.classList.remove("active");
    lastStepEl.classList.add("completed");
  }
  updateStatus("Análisis completado", "active"); 
  displayResults(realData);
}

function showAnalysisProgress() {
  const progressContainer = document.getElementById("analysisProgress");
  const featuresViz = document.getElementById("featuresViz");

  if (progressContainer) progressContainer.style.display = "flex"; 
  if (featuresViz) featuresViz.style.display = "block";

  animateFeatureBar("blackPixels", 0);
  animateFeatureBar("greenPixels", 0);
  animateFeatureBar("edgeDensity", 0);
  animateFeatureBar("texture", 0);
  
  document.querySelectorAll(".progress-step").forEach((stepEl) => {
    stepEl.classList.remove("active", "completed");
  });
  const firstStep = document.querySelector('.progress-step[data-step="1"]');
  if (firstStep) firstStep.classList.add("active");
}

function animateFeatureBar(featureId, value) {
  const fill = document.getElementById(featureId); 
  const valueEl = document.getElementById(featureId + "Value");

  if (fill && valueEl) {
    const validValue = Math.max(0, Math.min(100, parseFloat(value) || 0)); 
    fill.style.width = validValue + "%";
    valueEl.textContent = Math.round(validValue) + "%";
  }
}


function getDiseaseConfig(diseaseName) {
    const diseaseStyles = {
        "Black Rot": { icon: "fas fa-bacteria", color: "#795548", borderColor: "#5D4037", description: "Detectadas manchas necróticas oscuras, compatibles con Pudrición Negra (Black Rot)." },
        "Esca (Black Measles)": { icon: "fas fa-band-aid", color: "#FF9800", borderColor: "#F57C00", description: "Signos de marchitez y manchas foliares atigradas, sugieren la enfermedad de la Yesca (Esca)." },
        "Leaf Blight (Isariopsis Leaf Spot)": { icon: "fas fa-wind", color: "#F44336", borderColor: "#D32F2F", description: "Presencia de manchas marrones o rojizas, indicativo de Tizón Foliar (Leaf Blight)." },
        "Healthy": { icon: "fas fa-leaf", color: "var(--success-color)", borderColor: "var(--natural-green-dark)", description: "La hoja aparenta estar sana, sin signos visuales de las enfermedades comunes analizadas." },
        "Indeterminado": { icon: "fas fa-question-circle", color: "#9E9E9E", borderColor: "#757575", description: "El diagnóstico no es concluyente. Se recomienda una inspección más detallada o consultar a un experto."}
    };
    return diseaseStyles[diseaseName] || diseaseStyles["Indeterminado"];
}


function displayResults(results) {
  const resultsSection = document.getElementById("results");
  if (!resultsSection) return;
  resultsSection.style.display = "block";

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

  fillMainDiagnosis(results);
  fillSystemsResults(results);
  fillRecommendations(results.recomendaciones || getRecommendations(results.clase_final));

  currentAnalysisResults = results; 
}

function fillMainDiagnosis(results) {
  const iconEl = document.getElementById("diagnosisIcon");
  const titleEl = document.getElementById("diagnosisTitle");
  const descriptionEl = document.getElementById("diagnosisDescription");
  const confidenceFillEl = document.getElementById("confidenceFill");
  const confidenceValueEl = document.getElementById("confidenceValue");
  const diagnosisCardEl = document.querySelector(".diagnosis-card");


  const config = getDiseaseConfig(results.clase_final);

  if (iconEl) iconEl.innerHTML = `<i class="${config.icon}"></i>`;
  if (titleEl) titleEl.textContent = results.clase_final;
  if (descriptionEl) descriptionEl.textContent = config.description;
  
  if (diagnosisCardEl) diagnosisCardEl.style.borderColor = config.borderColor;
  if (iconEl) iconEl.style.background = config.color; 

  const confidencePercentage = (parseFloat(results.confianza_final) || 0) * 100;
  if (confidenceFillEl) {
    confidenceFillEl.style.width = `${confidencePercentage}%`;
    if (confidencePercentage > 80) confidenceFillEl.style.background = 'var(--primary-color)';
    else if (confidencePercentage > 60) confidenceFillEl.style.background = 'var(--secondary-color)';
    else confidenceFillEl.style.background = 'var(--warning-color)';
  }
  if (confidenceValueEl) confidenceValueEl.textContent = `${Math.round(confidencePercentage)}%`;
}


function fillSystemsResults(results) {
  const sistemas = results.resultados_individuales;
  if (!sistemas) return;

  const cnnPredEl = document.getElementById("cnnPrediction");
  const cnnConfEl = document.getElementById("cnnConfidence");
  const cnnProbsEl = document.getElementById("cnnProbabilities");
  if (sistemas.cnn) {
    if (cnnPredEl) cnnPredEl.textContent = sistemas.cnn.clase || "-";
    if (cnnConfEl) cnnConfEl.textContent = `${Math.round((sistemas.cnn.confianza || 0) * 100)}%`;
    if (cnnProbsEl && sistemas.cnn.probabilidades) {
      cnnProbsEl.innerHTML = Object.entries(sistemas.cnn.probabilidades)
        .sort(([,a],[,b]) => b-a) 
        .slice(0, 3) 
        .map(([disease, prob]) => `<div>${disease}: <strong>${Math.round(prob * 100)}%</strong></div>`)
        .join("");
    }
  }

  const treePredEl = document.getElementById("treePrediction");
  const treeConfEl = document.getElementById("treeConfidence");
  const treeFeatEl = document.getElementById("treeFeatures"); 
   if (sistemas.arbol) {
    if (treePredEl) treePredEl.textContent = sistemas.arbol.clase || "-";
    if (treeConfEl) treeConfEl.textContent = `${Math.round((sistemas.arbol.confianza || 0) * 100)}%`;
    if (treeFeatEl) { 
        let featuresText = "<ul>";
        if (results.caracteristicas_clave) {
            for (const [key, value] of Object.entries(results.caracteristicas_clave)) {
                featuresText += `<li><strong>${key.replace(/_/g, ' ')}:</strong> ${(typeof value === 'number' ? value.toFixed(2) : value)}</li>`;
            }
        } else {
            featuresText += "<li>Color</li><li>Textura</li><li>Bordes</li>"; 
        }
        featuresText += "</ul>";
        treeFeatEl.innerHTML = featuresText;
    }
  }


  const expPredEl = document.getElementById("expertPrediction");
  const expConfEl = document.getElementById("expertConfidence");
  const expRulesEl = document.getElementById("expertRules");
  if (sistemas.experto) {
    if (expPredEl) expPredEl.textContent = sistemas.experto.clase || "-";
    if (expConfEl) expConfEl.textContent = `${Math.round((sistemas.experto.confianza || 0) * 100)}%`;
    if (expRulesEl) expRulesEl.innerHTML = `Regla activada: <strong>${sistemas.experto.regla_activada || "Análisis de píxeles"}</strong>`;
  }
}


function fillRecommendations(recommendationsArray) {
  const container = document.getElementById("recommendationsContent");
  if (!container || !recommendationsArray || recommendationsArray.length === 0) {
    if (container) container.innerHTML = "<p>No hay recomendaciones específicas en este momento.</p>";
    return;
  }

  container.innerHTML = recommendationsArray.map(rec => `
    <div class="recommendation-item">
        <i class="fas fa-check-circle"></i> 
        <span>${rec}</span>
    </div>
  `).join("");
}


function updateStatus(message, type, stepNumber = null) {
  const statusText = document.getElementById("statusText");
  const statusDot = document.querySelector(".status-indicator .status-dot"); 

  if (statusText) statusText.textContent = message;

  if (statusDot) {
    statusDot.className = "status-dot"; 
    if (type !== "waiting") {
      statusDot.classList.add(type);
    }
  }

  if (stepNumber) {
    document.querySelectorAll(".progress-step").forEach(stepEl => {
      const currentStepData = parseInt(stepEl.dataset.step);
      if (currentStepData < stepNumber) {
        stepEl.classList.remove("active");
        stepEl.classList.add("completed");
      } else if (currentStepData === stepNumber) {
        stepEl.classList.remove("completed");
        stepEl.classList.add("active");
      } else {
        stepEl.classList.remove("active", "completed");
      }
    });
  }
}


function showNotification(message, type = "info") {
  const existingNotification = document.querySelector(".notification");
  if (existingNotification) existingNotification.remove(); 

  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`; 
  
  let iconClass = "fas fa-info-circle"; 
  if (type === "error") iconClass = "fas fa-exclamation-circle";
  else if (type === "success") iconClass = "fas fa-check-circle";
  else if (type === "warning") iconClass = "fas fa-exclamation-triangle";

  notification.innerHTML = `
    <i class="${iconClass}"></i>
    <span>${message}</span>
    <button class="close-notification" onclick="this.parentElement.remove()">
        <i class="fas fa-times"></i>
    </button>
  `;
  
  document.body.appendChild(notification);

  setTimeout(() => {
    if (notification.parentElement) { 
      notification.style.opacity = '0'; 
      setTimeout(() => notification.remove(), 300); 
    }
  }, 5000); 
}


function scrollToDetector() {
  const detectorSection = document.getElementById("detector");
  if (detectorSection) {
      detectorSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function showDemo() {
  showNotification("La demostración interactiva se activa al subir una imagen. ¡Prueba el detector!", "info");
}

function showTechInfo(tech) {
  const techInfo = {
    cnn: "Red Neuronal Convolucional (CNN): Usa aprendizaje profundo (EfficientNetB0) para identificar patrones visuales complejos en las imágenes de las hojas, logrando alta precisión en la clasificación.",
    tree: "Árbol de Decisión: Emplea un modelo basado en reglas extraídas de más de 70 características de imagen (color, textura GLCM/LBP, morfología) para un diagnóstico interpretable.",
    expert: "Sistema Experto: Aplica un conjunto de reglas heurísticas, codificando conocimiento de agrónomos, para identificar enfermedades basándose en umbrales de características específicas.",
  };
  showNotification(techInfo[tech] || "Información detallada no disponible para esta tecnología.", "info");
}

function downloadReport() {
  if (!currentAnalysisResults) {
    showNotification("No hay resultados para descargar. Por favor, analiza una imagen primero.", "error");
    return;
  }

  let reportText = `Reporte de Diagnóstico - GrapeAI\n`;
  reportText += `Fecha: ${new Date().toLocaleString('es-ES')}\n\n`;
  reportText += `Diagnóstico Principal: ${currentAnalysisResults.clase_final}\n`;
  reportText += `Confianza General: ${Math.round(currentAnalysisResults.confianza_final * 100)}%\n\n`;
  
  reportText += `Resultados por Sistema IA:\n`;
  reportText += `  - CNN: ${currentAnalysisResults.resultados_individuales.cnn.clase} (Confianza: ${Math.round(currentAnalysisResults.resultados_individuales.cnn.confianza * 100)}%)\n`;
  reportText += `  - Árbol Decisión: ${currentAnalysisResults.resultados_individuales.arbol.clase} (Confianza: ${Math.round(currentAnalysisResults.resultados_individuales.arbol.confianza * 100)}%)\n`;
  reportText += `  - Sistema Experto: ${currentAnalysisResults.resultados_individuales.experto.clase} (Confianza: ${Math.round(currentAnalysisResults.resultados_individuales.experto.confianza * 100)}%)\n\n`;

  reportText += `Características Clave Consideradas:\n`;
  for(const [key, value] of Object.entries(currentAnalysisResults.caracteristicas_clave || {})) {
      reportText += `  - ${key.replace(/_/g, ' ')}: ${(typeof value === 'number' ? value.toFixed(3) : value)}\n`;
  }
  reportText += `\nRecomendaciones:\n`;
  (currentAnalysisResults.recomendaciones || []).forEach(rec => {
    reportText += `  - ${rec}\n`;
  });

  const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `Reporte_GrapeAI_${currentAnalysisResults.clase_final.replace(/\s/g, '_')}_${new Date().toISOString().slice(0,10)}.txt`;
  document.body.appendChild(a); 
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showNotification("Reporte de texto descargado exitosamente.", "success");
}

function analyzeAnother() {
  const detectorSection = document.getElementById("detector");
  if (detectorSection) {
    detectorSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  setTimeout(() => {
    changeImage(); 
  }, 500);
}

async function shareResults() {
  if (!currentAnalysisResults) {
    showNotification("No hay resultados para compartir. Analiza una imagen primero.", "error");
    return;
  }

  const shareTitle = "Diagnóstico de Hoja de Uva - GrapeAI";
  const shareText = `GrapeAI diagnosticó: ${currentAnalysisResults.clase_final} con una confianza del ${Math.round(currentAnalysisResults.confianza_final * 100)}%.\n\nRecomendaciones principales: ${(currentAnalysisResults.recomendaciones || ["Revisar reporte completo."]).slice(0,2).join(", ")}.`;
  const shareUrl = window.location.href; 

  if (navigator.share) {
    try {
      await navigator.share({
        title: shareTitle,
        text: shareText,
        url: shareUrl,
      });
      showNotification("Resultados compartidos exitosamente.", "success");
    } catch (error) {
      console.error("Error al compartir:", error);
      if (error.name !== 'AbortError') { 
        showNotification("No se pudieron compartir los resultados.", "error");
      }
    }
  } else {
    try {
      await navigator.clipboard.writeText(`${shareTitle}\n${shareText}\n${shareUrl}`);
      showNotification("Información del resultado copiada al portapapeles.", "info");
    } catch (err) {
      console.error('Error al copiar al portapapeles:', err);
      showNotification("No se pudo copiar al portapapeles. Intenta manualmente.", "warning");
    }
  }
}