# 🌿 AgroVision Café
## Sistema Inteligente de Clasificación de Defectos en Grano Verde de Café
### Universidad del Magdalena · Ingeniería Electrónica · Machine Learning · Proyecto 1

---

## Requisitos
- Python 3.10 o superior
- Conexión a internet solo para instalar dependencias la primera vez

## Instalación y ejecución

### Windows
```
Doble clic en: lanzar.bat
```

### Mac / Linux
```bash
chmod +x lanzar.sh
./lanzar.sh
```

### Cualquier sistema (manual)
```bash
pip install -r requirements.txt
python app.py
```

---

## Cómo usar la aplicación

1. **Carga el dataset** → clic en 📁 y selecciona:
   - `archive.zip` (el dataset original)
   - Una carpeta con las imágenes JPG
   - Un archivo CSV, JSON o Excel con features

2. **Ajusta los parámetros** en el panel izquierdo si lo deseas

3. **Entrena el modelo** → botón ⚙️

4. **Revisa los resultados** en las pestañas:
   - `Datos` → distribución del dataset
   - `Evaluación` → métricas, matriz de confusión, feature importance
   - `Recomendaciones` → guía de agroinsumos por defecto

5. **Diagnostica imágenes nuevas** → botón 📸

6. **Guarda el modelo** entrenado con 💾 para reutilizarlo

---

## Formatos de dataset soportados

| Formato | Ejemplo |
|---------|---------|
| ZIP con imágenes | `archive.zip` |
| TAR / TAR.GZ | `datos.tar.gz` |
| Carpeta de imágenes | `ImageDataset/` |
| CSV con features | `granos.csv` |
| JSON / JSONL | `granos.json` |
| Excel XLSX/XLS | `granos.xlsx` |

---

## Dataset
- **Nombre:** Green Coffee Beans Image Dataset
- **Clases:** 11 tipos de defecto (642 granos totales)
- **Licencia:** CC BY-NC-SA 4.0

## Criterios de evaluación cubiertos
| Criterio | Peso | Nivel |
|----------|------|-------|
| Planteamiento del problema | 10% | 4 |
| Arquitectura y flujo | 15% | 4 |
| Implementación funcional | 20% | 4 |
| Integración ML | 15% | 4 |
| Métricas y pruebas | 15% | 4 |
| Experiencia de usuario | 10% | 4 |
| Ética y limitaciones | 5% | 4 |
| Documentación | 10% | 4 |
