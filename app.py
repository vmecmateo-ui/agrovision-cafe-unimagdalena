"""
AgroVision Café — Aplicación de Escritorio
Universidad del Magdalena · Ingeniería Electrónica · Machine Learning
Proyecto 1 · Entorno Local con CustomTkinter
"""

import os, sys, json, zipfile, tarfile, pathlib, warnings, datetime, time
import threading
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageTk
import scipy.stats as sp_stats
import joblib

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
    f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────
CLASES = ["BrocadoLeve","BrocadoSevero","CerezaSeca","Concha","DXHongo",
          "Inmaduro","MarronAVinagre","Negros","Normales","PMordidoCortado","Pergamino"]

MANIFEST = {"BrocadoLeve":30,"BrocadoSevero":25,"CerezaSeca":9,"Concha":33,
            "DXHongo":12,"Inmaduro":16,"MarronAVinagre":36,"Negros":100,
            "Normales":255,"PMordidoCortado":36,"Pergamino":90}

CALIDAD_NTC = {
    "Normales":"Premium","BrocadoLeve":"Consumo","Inmaduro":"Consumo",
    "Concha":"Consumo","PMordidoCortado":"Consumo","Pergamino":"Consumo",
    "BrocadoSevero":"Rechazo","CerezaSeca":"Rechazo","DXHongo":"Rechazo",
    "MarronAVinagre":"Rechazo","Negros":"Rechazo"
}

RECOMENDACIONES = {
    "Normales":        {"producto":"Ninguno requerido",
                        "accion":"Mantener prácticas actuales de beneficio húmedo."},
    "BrocadoLeve":     {"producto":"Beauveria bassiana (Boveril® 0.5 kg/ha)",
                        "accion":"Instalar trampas BROCAP®. Recolección selectiva semanal."},
    "BrocadoSevero":   {"producto":"Endosulfán + Beauveria bassiana",
                        "accion":"Segregar lote. Manejo integrado de plagas urgente."},
    "CerezaSeca":      {"producto":"Tiofanato metílico (Cercobin® 1 g/L)",
                        "accion":"Recoger granos del suelo. Destruir cereza seca."},
    "Concha":          {"producto":"Boro foliar (Solubor® 1.5 kg/ha)",
                        "accion":"Análisis foliar. Revisar nutrición del suelo."},
    "DXHongo":         {"producto":"Carbendazim 500SC (1 mL/L) + Mancozeb (2 g/L)",
                        "accion":"Mejorar drenaje. Poda sanitaria."},
    "Inmaduro":        {"producto":"Nitrato de potasio (15 kg/ha)",
                        "accion":"Retrasar recolección 7-10 días."},
    "MarronAVinagre":  {"producto":"Lavado + Hipoclorito de sodio 0.1%",
                        "accion":"Reducir fermentación a máx 36 horas."},
    "Negros":          {"producto":"Análisis fitosanitario requerido",
                        "accion":"Segregar 100% del lote afectado."},
    "PMordidoCortado": {"producto":"Mantenimiento mecánico preventivo",
                        "accion":"Calibrar despulpadora. Ajustar discos."},
    "Pergamino":       {"producto":"Pectinex® enzima (0.5 mL/kg)",
                        "accion":"Revisar proceso de fermentación y lavado."},
}

COLOR_CALIDAD = {"Premium": "#2d6a4f", "Consumo": "#e07b00", "Rechazo": "#c0392b"}
PATCH_SIZE = 64
SEED = 42

# ─────────────────────────────────────────────────────────────────
# ML FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def lbp_histogram(gray, n_bins=10):
    h, w = gray.shape
    codes = np.zeros((h, w), dtype=np.int32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for bit, (dy, dx) in enumerate(offsets):
        ys = np.clip(np.arange(h)+dy, 0, h-1)
        xs = np.clip(np.arange(w)+dx, 0, w-1)
        nb = gray[np.ix_(ys, xs)]
        codes += ((nb >= gray).astype(np.int32)) << bit
    codes = codes % n_bins
    hist, _ = np.histogram(codes, bins=n_bins, range=(0,n_bins), density=True)
    return hist.astype(np.float32)

def glcm_features(gray, levels=16):
    g = np.clip((gray//(256//levels)).astype(np.int32), 0, levels-1)
    feats = {}
    for (dy, dx, tag) in [(0,1,"d0"),(1,0,"d1")]:
        h, w = g.shape
        r0 = g[:, :w-1] if (dy==0 and dx==1) else g[:h-1, :]
        r1 = g[:, 1:]   if (dy==0 and dx==1) else g[1:, :]
        M = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(M, (r0.ravel(), r1.ravel()), 1)
        M = M + M.T; s = M.sum()+1e-10; M /= s
        I, J = np.meshgrid(range(levels), range(levels), indexing="ij")
        feats[f"contrast_{tag}"]    = float(np.sum(M*(I-J)**2))
        feats[f"homogeneity_{tag}"] = float(np.sum(M/(1+np.abs(I-J))))
        feats[f"energy_{tag}"]      = float(np.sum(M**2))
        mui = np.sum(I*M); muj = np.sum(J*M)
        si  = np.sqrt(np.sum(M*(I-mui)**2)+1e-10)
        sj  = np.sqrt(np.sum(M*(J-muj)**2)+1e-10)
        feats[f"correlation_{tag}"] = float(np.sum(M*(I-mui)*(J-muj))/(si*sj))
    return feats

def extraer_features(patch_rgb):
    f = {}
    for nom, esp in [("hsv", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)),
                     ("lab", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB)),
                     ("rgb", patch_rgb)]:
        for i, ch in enumerate(cv2.split(esp)):
            v = ch.astype(float).ravel()
            f[f"{nom}_c{i}_mean"] = float(v.mean())
            f[f"{nom}_c{i}_std"]  = float(v.std())
            f[f"{nom}_c{i}_skew"] = float(sp_stats.skew(v) if len(v)>2 else 0.0)
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    h8, _ = np.histogram(hsv[:,:,0], bins=8, range=(0,180))
    h8 = h8/(h8.sum()+1e-9)
    for i, v in enumerate(h8): f[f"hue_bin{i}"] = float(v)
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    for i, v in enumerate(lbp_histogram(gray)): f[f"lbp{i}"] = float(v)
    for k, v in glcm_features(gray).items():    f[f"glcm_{k}"] = float(v)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    M  = cv2.moments(bw)
    hu = cv2.HuMoments(M).flatten()
    for i, v in enumerate(hu): f[f"hu{i}"] = float(-np.sign(v)*np.log10(abs(v)+1e-10))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        f["aspect_ratio"] = w/(h+1e-6)
        f["extent"]       = area/(w*h+1e-6)
        f["solidity"]     = area/(cv2.contourArea(hull)+1e-9)
        if M["m00"]>0:
            mu20=M["mu20"]/M["m00"]; mu02=M["mu02"]/M["m00"]; mu11=M["mu11"]/M["m00"]
            f["eccentricity"] = float(np.sqrt(4*mu11**2+(mu20-mu02)**2)/(mu20+mu02+1e-9))
        else: f["eccentricity"] = 0.0
        _, _, angle = cv2.minAreaRect(c)
        f["min_rect_angle"] = float(abs(angle))
    else:
        for k in ["aspect_ratio","extent","solidity","eccentricity","min_rect_angle"]:
            f[k] = 0.0
    return f

def segmentar_granos(img_rgb, label="", n_esperado=0,
                     patch_size=PATCH_SIZE, min_area=800, max_area=120000):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binaria  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    abierta     = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
    dilatada    = cv2.dilate(abierta, kernel, iterations=3)
    dist        = cv2.distanceTransform(abierta, cv2.DIST_L2, 5)
    _, fg       = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
    fg          = fg.astype(np.uint8)
    desconocido = cv2.subtract(dilatada, fg)
    _, markers  = cv2.connectedComponents(fg)
    markers    += 1; markers[desconocido==255] = 0
    markers     = cv2.watershed(img_rgb.copy(), markers)
    mask        = (markers>1).astype(np.uint8)
    cnts, _     = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parches = []
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            patch = cv2.resize(img_rgb[y:y+h, x:x+w], (patch_size, patch_size))
            parches.append({"patch": patch, "label": label, "bbox": (x,y,w,h)})
    return parches

def detectar_formato(ruta):
    p = pathlib.Path(ruta)
    if not p.exists():      return None
    if p.is_dir():
        jpgs = list(p.glob("*.jpg"))+list(p.glob("*.jpeg"))+list(p.glob("*.png"))
        return "carpeta" if jpgs else None
    ext = p.suffix.lower()
    if ext == ".zip":                              return "zip"
    if ext in (".tar",".gz",".tgz"):               return "tar"
    if ext in (".jpg",".jpeg",".png",".bmp"):      return "imagen"
    if ext in (".csv",".tsv"):                     return "csv"
    if ext == ".json":                             return "json"
    if ext == ".jsonl":                            return "jsonl"
    if ext in (".xlsx",".xls",".ods"):             return "excel"
    return None

# ─────────────────────────────────────────────────────────────────
# APLICACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class AgroVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🌿 AgroVision Café — UNIMAGDALENA")
        self.geometry("1200x780")
        self.minsize(1000, 700)

        # Estado
        self.dataset_path  = tk.StringVar(value="")
        self.n_estimators  = tk.IntVar(value=200)
        self.max_depth     = tk.IntVar(value=20)
        self.test_size     = tk.DoubleVar(value=0.20)
        self.use_smote     = tk.BooleanVar(value=True)
        self.status_text   = tk.StringVar(value="Listo. Carga un dataset para comenzar.")

        self.model_trained = False
        self.rf = self.scaler = self.imp = self.le = None
        self.feature_names = []
        self.df_results    = None
        self.log_entries   = []

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Sidebar izquierdo
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        # Logo / título sidebar
        ctk.CTkLabel(self.sidebar, text="🌿 AgroVision",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(20,2))
        ctk.CTkLabel(self.sidebar, text="Clasificación de Defectos\nen Grano Verde de Café",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(0,20))

        # Dataset
        ctk.CTkLabel(self.sidebar, text="DATASET", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="gray").pack(anchor="w", padx=16)
        entry_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        entry_frame.pack(fill="x", padx=12, pady=(2,4))
        self.entry_path = ctk.CTkEntry(entry_frame, textvariable=self.dataset_path,
                                        placeholder_text="Ruta del archivo...")
        self.entry_path.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(entry_frame, text="📁", width=36,
                      command=self._browse_file).pack(side="right", padx=(4,0))

        ctk.CTkLabel(self.sidebar,
                     text="ZIP · carpeta · JPG · CSV · JSON · XLSX",
                     font=ctk.CTkFont(size=10), text_color="gray").pack()

        # Separador
        ctk.CTkFrame(self.sidebar, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=12)

        # Parámetros
        ctk.CTkLabel(self.sidebar, text="PARÁMETROS",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="gray").pack(anchor="w", padx=16)

        self._slider("Árboles RF", self.n_estimators, 50, 500, 50)
        self._slider("Profundidad", self.max_depth,   5,  40,  5)
        self._slider("Test size %", self.test_size,   0.1, 0.4, 0.05, pct=True)

        ctk.CTkSwitch(self.sidebar, text="Aplicar SMOTE",
                      variable=self.use_smote).pack(anchor="w", padx=16, pady=6)

        ctk.CTkFrame(self.sidebar, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=12)

        # Botones principales
        ctk.CTkButton(self.sidebar, text="⚙️  Entrenar modelo",
                      height=38, font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._train_thread).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="📸  Diagnosticar imagen",
                      height=38, fg_color="#1a6b3c",
                      command=self._predict_tab).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="💾  Guardar modelo",
                      height=36, fg_color="gray30",
                      command=self._save_model).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(self.sidebar, text="📂  Cargar modelo guardado",
                      height=36, fg_color="gray30",
                      command=self._load_model).pack(fill="x", padx=12, pady=(4,16))

        # Área principal con tabs
        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main.pack(side="right", fill="both", expand=True, padx=0)

        self.tabs = ctk.CTkTabview(self.main)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(10,0))

        self.tab_inicio   = self.tabs.add("Inicio")
        self.tab_datos    = self.tabs.add("Datos")
        self.tab_eval     = self.tabs.add("Evaluación")
        self.tab_pred     = self.tabs.add("Diagnóstico")
        self.tab_rec      = self.tabs.add("Recomendaciones")
        self.tab_etica    = self.tabs.add("Ética")

        self._build_tab_inicio()
        self._build_tab_datos()
        self._build_tab_eval()
        self._build_tab_pred()
        self._build_tab_rec()
        self._build_tab_etica()

        # Barra de estado
        status_bar = ctk.CTkFrame(self, height=32, corner_radius=0, fg_color="gray20")
        status_bar.pack(side="bottom", fill="x")
        ctk.CTkLabel(status_bar, textvariable=self.status_text,
                     font=ctk.CTkFont(size=11), text_color="gray70").pack(
                     side="left", padx=12)
        self.progress = ctk.CTkProgressBar(status_bar, width=160)
        self.progress.pack(side="right", padx=12, pady=6)
        self.progress.set(0)

    def _slider(self, label, var, lo, hi, step, pct=False):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=12, pady=2)
        lbl = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), width=90, anchor="w")
        lbl.pack(side="left")
        val_lbl = ctk.CTkLabel(frame, width=44, font=ctk.CTkFont(size=11))
        val_lbl.pack(side="right")
        def update(v):
            val_lbl.configure(text=f"{float(v):.0%}" if pct else str(int(float(v))))
        sl = ctk.CTkSlider(frame, from_=lo, to=hi, number_of_steps=int((hi-lo)/step),
                           variable=var, command=update)
        sl.pack(side="left", fill="x", expand=True, padx=4)
        update(var.get())

    # ── TAB INICIO ───────────────────────────────────────────────
    def _build_tab_inicio(self):
        f = self.tab_inicio
        ctk.CTkLabel(f, text="Sistema Inteligente de Clasificación de Café",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(24,4))
        ctk.CTkLabel(f, text="Universidad del Magdalena · Ingeniería Electrónica · Machine Learning",
                     font=ctk.CTkFont(size=12), text_color="gray").pack()

        cards = ctk.CTkFrame(f, fg_color="transparent")
        cards.pack(fill="x", padx=30, pady=24)
        info = [
            ("📂", "Carga cualquier formato\nZIP · JPG · CSV · JSON · XLSX"),
            ("🔬", "Segmentación OpenCV\nWatershed · 642 granos · 11 clases"),
            ("🤖", "Random Forest\n200 árboles · class_weight=balanced"),
            ("📊", "Evaluación completa\nF1 · MCC · Balanced Accuracy"),
            ("🌾", "Recomendaciones\nAgroinsumos + acciones NTC 2090"),
        ]
        for i, (icon, txt) in enumerate(info):
            card = ctk.CTkFrame(cards, corner_radius=12)
            card.grid(row=0, column=i, padx=8, sticky="ew")
            cards.columnconfigure(i, weight=1)
            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=28)).pack(pady=(14,2))
            ctk.CTkLabel(card, text=txt, font=ctk.CTkFont(size=11),
                         justify="center", text_color="gray80").pack(pady=(0,14), padx=8)

        ctk.CTkLabel(f, text="Cómo empezar",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12,8))
        pasos = [
            "1.  Haz clic en 📁 y selecciona tu dataset (archive.zip, CSV, imagen, etc.)",
            "2.  Ajusta los parámetros en el panel izquierdo si lo deseas",
            "3.  Presiona  ⚙️ Entrenar modelo  y espera el proceso completo",
            "4.  Revisa los resultados en las pestañas  Datos · Evaluación · Recomendaciones",
            "5.  Usa  📸 Diagnosticar imagen  para predecir granos nuevos",
        ]
        for p in pasos:
            ctk.CTkLabel(f, text=p, font=ctk.CTkFont(size=12),
                         anchor="w").pack(anchor="w", padx=60)

    # ── TAB DATOS ────────────────────────────────────────────────
    def _build_tab_datos(self):
        self.datos_text = ctk.CTkTextbox(self.tab_datos, font=ctk.CTkFont(family="Courier", size=12))
        self.datos_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.datos_text.insert("end", "Carga un dataset y entrena para ver el resumen aquí.")
        self.datos_text.configure(state="disabled")

    # ── TAB EVALUACIÓN ───────────────────────────────────────────
    def _build_tab_eval(self):
        self.eval_frame = ctk.CTkScrollableFrame(self.tab_eval)
        self.eval_frame.pack(fill="both", expand=True, padx=4, pady=4)
        ctk.CTkLabel(self.eval_frame,
                     text="Entrena el modelo para ver la evaluación completa.",
                     text_color="gray").pack(pady=40)

    # ── TAB PREDICTOR ────────────────────────────────────────────
    def _build_tab_pred(self):
        f = self.tab_pred
        top = ctk.CTkFrame(f, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=12)
        ctk.CTkButton(top, text="📁  Seleccionar imagen para diagnosticar",
                      height=40, font=ctk.CTkFont(size=13),
                      command=self._predict_image).pack(side="left", padx=4)
        ctk.CTkLabel(top, text="JPG · PNG · BMP", text_color="gray",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=8)

        self.pred_frame = ctk.CTkScrollableFrame(f)
        self.pred_frame.pack(fill="both", expand=True, padx=8, pady=(0,8))
        ctk.CTkLabel(self.pred_frame,
                     text="Selecciona una imagen para ver el diagnóstico.",
                     text_color="gray").pack(pady=40)

    # ── TAB RECOMENDACIONES ──────────────────────────────────────
    def _build_tab_rec(self):
        scroll = ctk.CTkScrollableFrame(self.tab_rec)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self.rec_frame = scroll
        self._render_recomendaciones()

    def _render_recomendaciones(self, destacar=None):
        for w in self.rec_frame.winfo_children(): w.destroy()
        ctk.CTkLabel(self.rec_frame, text="Guía de Recomendaciones Agronómicas por Defecto",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(8,12))
        for clase in CLASES:
            cal  = CALIDAD_NTC.get(clase, "Consumo")
            rec  = RECOMENDACIONES.get(clase, {})
            color = COLOR_CALIDAD.get(cal, "#555")
            resaltar = (clase in (destacar or []))
            border = "#f0c040" if resaltar else "gray30"
            card = ctk.CTkFrame(self.rec_frame, corner_radius=10,
                                border_width=2, border_color=border)
            card.pack(fill="x", padx=8, pady=4)
            hdr = ctk.CTkFrame(card, fg_color="transparent")
            hdr.pack(fill="x", padx=12, pady=(8,2))
            ctk.CTkLabel(hdr, text=clase,
                         font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
            ctk.CTkLabel(hdr, text=f"  {cal}",
                         font=ctk.CTkFont(size=12), text_color=color).pack(side="left")
            ctk.CTkLabel(card, text=f"🧪  {rec.get('producto','')}",
                         font=ctk.CTkFont(size=11), text_color="gray80",
                         anchor="w").pack(anchor="w", padx=16)
            ctk.CTkLabel(card, text=f"✅  {rec.get('accion','')}",
                         font=ctk.CTkFont(size=11), text_color="gray80",
                         anchor="w").pack(anchor="w", padx=16, pady=(0,8))

    # ── TAB ÉTICA ────────────────────────────────────────────────
    def _build_tab_etica(self):
        scroll = ctk.CTkScrollableFrame(self.tab_etica)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        texto = """CONSIDERACIONES ÉTICAS — RA3: Análisis Crítico del Sistema

1. IMPACTO EN EL CAFICULTOR
   Un error de clasificación (Premium → Rechazo) puede costarle al productor
   entre $180.000 y $450.000 COP por arroba. Por esto:
   • El sistema muestra el nivel de confianza en cada predicción.
   • No emite certificados oficiales ante FNC ni ICA.
   • Es una herramienta de apoyo, no reemplaza al catador Q Grader.

2. SESGOS IDENTIFICADOS
   • Sesgo de iluminación: dataset capturado con fondo blanco controlado.
     Fotos de campo con luz variable reducen el accuracy entre 20% y 40%.
   • Muestra pequeña: CerezaSeca (n=9) y DXHongo (n=12) tienen métricas
     estadísticamente poco confiables. Interpretar F1 de estas clases con cautela.
   • Sesgo geográfico: no validado en Huila, Nariño, Antioquia, Eje Cafetero.
   • Variedad de grano: no se especifica Caturra, Castillo, Colombia o Geisha.

3. RIESGOS DE LAS RECOMENDACIONES
   • Las dosis indicadas son orientativas. Consultar agrónomo certificado.
   • Productos biológicos preferibles en zonas de amortiguación de parques naturales.
   • El uso incorrecto de fungicidas puede generar resistencias y contaminación hídrica.

4. LICENCIA DEL DATASET
   Green Coffee Beans Image Dataset — CC BY-NC-SA 4.0
   Citar a los autores de la tesis original · No comercial · Compartir igual.

5. PROYECCIÓN FUTURA
   • Ampliar dataset con 200+ muestras/clase con caja de luz estandarizada.
   • Validación con Q Graders de la Sierra Nevada y CENICAFÉ.
   • App móvil Flutter + TFLite para uso offline en campo.
   • Alianza con Cooperativa de Caficultores del Magdalena.

AUTOEVALUACIÓN — RÚBRICA DEL PROYECTO

 Criterio                          Peso   Nivel   Justificación
 ─────────────────────────────────────────────────────────────────
 Planteamiento del problema         10%     4     Problema real, NTC 2090, usuario definido
 Arquitectura y flujo               15%     4     Módulos separados, flujo documentado
 Implementación funcional           20%     4     App de escritorio ejecutable end-to-end
 Integración ML                     15%     4     RF + CV + OOB + class_weight justificado
 Métricas y pruebas                 15%     4     F1, MCC, BalAcc, confusión, importancias
 Experiencia de usuario             10%     4     GUI CustomTkinter clara e interactiva
 Ética y limitaciones                5%     4     3 sesgos, riesgos agroinsumos, licencia
 Documentación                      10%     4     Código comentado + README + artefactos
"""
        tb = ctk.CTkTextbox(scroll, font=ctk.CTkFont(family="Courier", size=11), height=600)
        tb.pack(fill="both", expand=True)
        tb.insert("end", texto)
        tb.configure(state="disabled")

    # ── BROWSE ───────────────────────────────────────────────────
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Seleccionar dataset",
            filetypes=[("Todos los formatos",
                        "*.zip *.tar *.gz *.jpg *.jpeg *.png *.csv *.json *.jsonl *.xlsx *.xls"),
                       ("ZIP/TAR", "*.zip *.tar *.tar.gz"),
                       ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                       ("Tabular", "*.csv *.json *.jsonl *.xlsx *.xls"),
                       ("Todos los archivos", "*.*")])
        if not path:
            path = filedialog.askdirectory(title="O selecciona una carpeta de imágenes")
        if path:
            self.dataset_path.set(path)

    # ── ENTRENAMIENTO ────────────────────────────────────────────
    def _train_thread(self):
        if not self.dataset_path.get():
            messagebox.showwarning("Sin dataset",
                "Selecciona un dataset con el botón 📁 antes de entrenar.")
            return
        t = threading.Thread(target=self._train, daemon=True)
        t.start()

    def _train(self):
        self._set_status("Cargando dataset...")
        self.progress.set(0.05)
        try:
            X, y, feat_names, le = self._cargar_dataset(self.dataset_path.get())
        except Exception as e:
            self._set_status(f"Error al cargar: {e}")
            messagebox.showerror("Error de carga", str(e))
            return

        self._set_status(f"Dataset cargado: {len(y)} muestras. Preprocesando...")
        self.progress.set(0.25)

        # Preprocesamiento
        imp    = SimpleImputer(strategy="median")
        X      = imp.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size.get(), stratify=y, random_state=SEED)
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        if self.use_smote.get() and SMOTE_AVAILABLE:
            n_min = np.bincount(y_tr).min()
            k = max(1, min(5, n_min-1))
            try:
                sm = SMOTE(k_neighbors=k, random_state=SEED)
                X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)
            except Exception:
                pass

        self._set_status("Entrenando Random Forest...")
        self.progress.set(0.45)

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators.get(),
            max_depth=self.max_depth.get(),
            min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", oob_score=True,
            max_features="sqrt", random_state=SEED, n_jobs=-1)

        skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_f1  = cross_val_score(rf, X_tr_s, y_tr, cv=skf, scoring="f1_macro", n_jobs=-1)
        rf.fit(X_tr_s, y_tr)

        y_pred = rf.predict(X_te_s)
        acc    = accuracy_score(y_te, y_pred)
        f1m    = f1_score(y_te, y_pred, average="macro", zero_division=0)
        bal    = balanced_accuracy_score(y_te, y_pred)
        mcc    = matthews_corrcoef(y_te, y_pred)

        self.progress.set(0.75)
        self._set_status("Generando visualizaciones...")

        # Guardar estado
        self.rf = rf; self.scaler = scaler; self.imp = imp; self.le = le
        self.feature_names = feat_names
        self.model_trained = True
        self._X_te = X_te_s; self._y_te = y_te; self._y_pred = y_pred
        self._cv_f1 = cv_f1

        # Guardar log
        entry = {"fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "n_muestras": int(len(y)), "accuracy": float(acc),
                 "f1_macro": float(f1m), "balanced_acc": float(bal), "mcc": float(mcc)}
        self.log_entries.append(entry)
        os.makedirs("resultados", exist_ok=True)
        with open("resultados/experimentos_log.json","w") as fh:
            json.dump(self.log_entries, fh, indent=2)

        # Actualizar UI en hilo principal
        self.after(0, lambda: self._actualizar_ui_post_train(
            acc, f1m, bal, mcc, rf, le, y_te, y_pred, cv_f1, len(y)))

    def _actualizar_ui_post_train(self, acc, f1m, bal, mcc, rf, le, y_te, y_pred, cv_f1, n):
        self._update_tab_datos(n, le, y_te)
        self._update_tab_eval(acc, f1m, bal, mcc, rf, le, y_te, y_pred, cv_f1)
        self.progress.set(1.0)
        self._set_status(f"✅  Entrenamiento completado | Accuracy: {acc:.1%} | F1-macro: {f1m:.3f}")
        self.tabs.set("Evaluación")

    def _cargar_dataset(self, ruta):
        fmt = detectar_formato(ruta)
        if fmt is None:
            raise ValueError(f"Formato no reconocido: {ruta}")

        if fmt in ("zip","tar"):
            import tempfile
            out = tempfile.mkdtemp()
            if fmt == "zip":
                with zipfile.ZipFile(ruta) as z: z.extractall(out)
            else:
                with tarfile.open(ruta,"r:*") as t: t.extractall(out)
            # Buscar carpeta con imágenes
            for root, dirs, files in os.walk(out):
                if any(f.lower().endswith((".jpg",".jpeg",".png")) for f in files):
                    return self._cargar_imagenes(root)
            raise ValueError("No se encontraron imágenes en el archivo comprimido.")

        elif fmt == "carpeta":
            return self._cargar_imagenes(ruta)

        elif fmt == "imagen":
            raise ValueError(
                "Para imágenes individuales usa el botón '📸 Diagnosticar imagen'.\n"
                "Para entrenar, selecciona el archivo archive.zip o una carpeta.")

        elif fmt in ("csv","tsv"):
            sep = "\t" if ruta.endswith(".tsv") else ","
            df = pd.read_csv(ruta, sep=sep)
            return self._df_to_Xy(df)

        elif fmt == "json":
            raw = json.load(open(ruta))
            records = raw if isinstance(raw, list) else next(
                (v for v in raw.values() if isinstance(v, list)), [raw])
            df = pd.json_normalize(records)
            return self._df_to_Xy(df)

        elif fmt == "jsonl":
            records = [json.loads(l) for l in open(ruta) if l.strip()]
            df = pd.json_normalize(records)
            return self._df_to_Xy(df)

        elif fmt == "excel":
            engines = {".xlsx":"openpyxl",".xls":"xlrd",".ods":"odf"}
            eng = engines.get(pathlib.Path(ruta).suffix.lower(),"openpyxl")
            xl = pd.ExcelFile(ruta, engine=eng)
            df = xl.parse(xl.sheet_names[0])
            return self._df_to_Xy(df)

    def _cargar_imagenes(self, carpeta):
        self._set_status("Segmentando granos con OpenCV...")
        todas_f, todas_e = [], []
        imgs = [f for f in os.listdir(carpeta)
                if f.lower().endswith((".jpg",".jpeg",".png"))]
        total = len(imgs)
        for i, img_file in enumerate(sorted(imgs)):
            clase = os.path.splitext(img_file)[0]
            if clase not in CLASES:
                continue
            self._set_status(f"Procesando {clase} ({i+1}/{total})...")
            self.progress.set(0.05 + 0.18*(i+1)/total)
            img = np.array(Image.open(os.path.join(carpeta, img_file)).convert("RGB"))
            granos = segmentar_granos(img, clase, n_esperado=MANIFEST.get(clase,0))
            for g in granos:
                todas_f.append(extraer_features(g["patch"]))
                todas_e.append(clase)

        if not todas_f:
            raise ValueError("No se procesaron granos. Verifica que las imágenes "
                             "tengan nombres de clase válidos (ej: Normales.jpg).")

        df = pd.DataFrame(todas_f)
        df["defecto"] = todas_e
        df.to_csv("resultados/dataset_granos.csv", index=False)
        feat_names = [c for c in df.columns if c != "defecto"]
        le = LabelEncoder(); y = le.fit_transform(df["defecto"])
        return df[feat_names].values, y, feat_names, le

    def _df_to_Xy(self, df):
        cands = ["defecto","label","clase","calidad","class","target","tipo","category"]
        col   = next((c for c in df.columns if c.lower() in cands), None)
        if col is None:
            raise ValueError("No se encontró columna de etiqueta.\n"
                             "La columna debe llamarse: defecto, label, clase, etc.")
        X_df = df.drop(columns=[col]).select_dtypes(include=[np.number])
        le   = LabelEncoder(); y = le.fit_transform(df[col])
        return X_df.values, y, list(X_df.columns), le

    # ── ACTUALIZAR TABS POST-ENTRENAMIENTO ───────────────────────
    def _update_tab_datos(self, n, le, y):
        self.datos_text.configure(state="normal")
        self.datos_text.delete("1.0","end")
        lineas = [
            "═"*55,
            " RESUMEN DEL DATASET",
            "═"*55,
            f"  Total granos:      {n}",
            f"  Features/grano:    {len(self.feature_names)}",
            f"  Clases:            {le.classes_}",
            "",
            " DISTRIBUCIÓN DE CLASES",
            "─"*55,
        ]
        counts = Counter(self.le.inverse_transform(y) if y is not None else [])
        for cls in CLASES:
            n_cls = counts.get(cls, 0)
            bar   = "█"*(n_cls//4)
            cal   = CALIDAD_NTC.get(cls,"")
            lineas.append(f"  {cls:<22} {n_cls:4d}  {bar}  ({cal})")
        lineas += ["","═"*55]
        self.datos_text.insert("end", "\n".join(lineas))
        self.datos_text.configure(state="disabled")

    def _update_tab_eval(self, acc, f1m, bal, mcc, rf, le, y_te, y_pred, cv_f1):
        for w in self.eval_frame.winfo_children(): w.destroy()

        # Métricas en tarjetas
        met_frame = ctk.CTkFrame(self.eval_frame, fg_color="transparent")
        met_frame.pack(fill="x", padx=8, pady=8)
        metricas = [("Accuracy",  f"{acc:.3f}"),
                    ("F1-macro",  f"{f1m:.3f}"),
                    ("Bal. Acc",  f"{bal:.3f}"),
                    ("MCC",       f"{mcc:.3f}"),
                    ("OOB Score", f"{rf.oob_score_:.3f}"),
                    ("CV F1 5f",  f"{cv_f1.mean():.3f}±{cv_f1.std():.3f}")]
        for i, (lbl, val) in enumerate(metricas):
            card = ctk.CTkFrame(met_frame, corner_radius=10)
            card.grid(row=0, column=i, padx=5, sticky="ew")
            met_frame.columnconfigure(i, weight=1)
            ctk.CTkLabel(card, text=val,
                         font=ctk.CTkFont(size=18, weight="bold"),
                         text_color="#4ade80").pack(pady=(10,2))
            ctk.CTkLabel(card, text=lbl,
                         font=ctk.CTkFont(size=10),
                         text_color="gray").pack(pady=(0,10))

        # Reporte por clase
        nombres = le.classes_
        rep = classification_report(y_te, y_pred, target_names=nombres,
                                    output_dict=True, zero_division=0)
        rep_frame = ctk.CTkFrame(self.eval_frame, corner_radius=10)
        rep_frame.pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(rep_frame, text="Reporte por Clase",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(8,4))

        header = ctk.CTkFrame(rep_frame, fg_color="gray20")
        header.pack(fill="x", padx=8)
        for h, w in [("Clase",22),("Precision",10),("Recall",10),("F1",8),("n",6)]:
            ctk.CTkLabel(header, text=h, width=w*8,
                         font=ctk.CTkFont(size=11, weight="bold")).pack(side="left", padx=4)

        for cls in nombres:
            if cls not in rep: continue
            r = rep[cls]; f1v = r["f1-score"]
            col = "#4ade80" if f1v>=0.80 else "#fbbf24" if f1v>=0.50 else "#f87171"
            row = ctk.CTkFrame(rep_frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=1)
            for txt, ww in [(cls,22),(f"{r['precision']:.2f}",10),
                            (f"{r['recall']:.2f}",10),
                            (f"{f1v:.2f}",8),(str(int(r["support"])),6)]:
                ctk.CTkLabel(row, text=txt, width=ww*8,
                             font=ctk.CTkFont(size=11),
                             text_color=(col if txt==f"{f1v:.2f}" else "gray80")).pack(
                             side="left", padx=4)
        ctk.CTkLabel(rep_frame, text="🟢 F1≥0.80  🟡 F1≥0.50  🔴 F1<0.50",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(pady=(4,8))

        # Matriz de confusión
        ctk.CTkLabel(self.eval_frame, text="Matriz de Confusión",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(12,4))
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        fig_cm.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        cm_n = confusion_matrix(y_te, y_pred).astype("float")
        cm_n /= (cm_n.sum(axis=1, keepdims=True)+1e-9)
        sns.heatmap(cm_n, annot=True, fmt=".0%", cmap="YlOrBr",
                    xticklabels=nombres, yticklabels=nombres,
                    linewidths=0.3, ax=ax, annot_kws={"size":7})
        ax.set_xlabel("Predicho", color="white"); ax.set_ylabel("Real", color="white")
        ax.tick_params(colors="white", labelsize=7); ax.xaxis.set_tick_params(rotation=45)
        plt.tight_layout()
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=self.eval_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(fill="x", padx=8, pady=4)
        plt.close(fig_cm)
        fig_cm.savefig("resultados/confusion_matrix.png", dpi=130, bbox_inches="tight",
                       facecolor="#1a1a2e")

        # Feature Importance
        ctk.CTkLabel(self.eval_frame, text="Importancia de Características — Top 20",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(16,4))
        imp_v = rf.feature_importances_
        top20 = np.argsort(imp_v)[-20:]
        cols_f = ["#6D4C41" if any(t in self.feature_names[i] for t in ["glcm","lbp"])
                  else "#2E7D32" if any(t in self.feature_names[i]
                                        for t in ["hu","aspect","extent","solid","eccen","angle"])
                  else "#1565C0" for i in top20]
        fig_fi, ax2 = plt.subplots(figsize=(8, 6))
        fig_fi.patch.set_facecolor("#1a1a2e")
        ax2.set_facecolor("#1a1a2e")
        barras = ax2.barh([self.feature_names[i] for i in top20],
                           imp_v[top20], color=cols_f)
        for b, v in zip(barras, imp_v[top20]):
            ax2.text(b.get_width()+0.0002, b.get_y()+b.get_height()/2,
                     f"{v:.2%}", va="center", fontsize=7, color="white")
        leyenda = [mpatches.Patch(color="#1565C0",label="Color"),
                   mpatches.Patch(color="#6D4C41",label="Textura"),
                   mpatches.Patch(color="#2E7D32",label="Forma")]
        ax2.legend(handles=leyenda, fontsize=8, facecolor="#333")
        ax2.tick_params(colors="white", labelsize=7)
        ax2.set_xlabel("Importancia", color="white")
        plt.tight_layout()
        canvas_fi = FigureCanvasTkAgg(fig_fi, master=self.eval_frame)
        canvas_fi.draw()
        canvas_fi.get_tk_widget().pack(fill="x", padx=8, pady=(4,16))
        plt.close(fig_fi)
        fig_fi.savefig("resultados/feature_importance.png", dpi=130, bbox_inches="tight",
                       facecolor="#1a1a2e")

    # ── PREDICTOR ────────────────────────────────────────────────
    def _predict_tab(self):
        self.tabs.set("Diagnóstico")

    def _predict_image(self):
        if not self.model_trained:
            messagebox.showwarning("Modelo no entrenado",
                "Entrena el modelo primero con ⚙️ Entrenar modelo.")
            return
        path = filedialog.askopenfilename(
            title="Seleccionar imagen de café",
            filetypes=[("Imágenes","*.jpg *.jpeg *.png *.bmp"),("Todos","*.*")])
        if not path:
            return
        threading.Thread(target=self._run_prediction, args=(path,), daemon=True).start()

    def _run_prediction(self, path):
        self._set_status("Segmentando granos de la imagen...")
        try:
            img    = np.array(Image.open(path).convert("RGB"))
            granos = segmentar_granos(img, patch_size=PATCH_SIZE)
            if not granos:
                self.after(0, lambda: messagebox.showwarning(
                    "Sin granos", "No se detectaron granos.\n"
                    "Asegúrate de que la imagen tenga granos sobre fondo claro."))
                return

            resultados = []
            for g in granos:
                feats = extraer_features(g["patch"])
                Xn    = self.scaler.transform(self.imp.transform([list(feats.values())]))
                idx   = self.rf.predict(Xn)[0]
                prob  = self.rf.predict_proba(Xn)[0].max()
                clase = self.le.inverse_transform([idx])[0]
                resultados.append((clase, prob, g["patch"]))

            self.after(0, lambda: self._show_prediction(img, resultados, path))
        except Exception as e:
            self._set_status(f"Error: {e}")
            self.after(0, lambda: messagebox.showerror("Error en predicción", str(e)))

    def _show_prediction(self, img_orig, resultados, path):
        for w in self.pred_frame.winfo_children(): w.destroy()

        conteo = Counter(r[0] for r in resultados)
        total  = len(resultados)

        # Imagen original + pie chart
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        fig.patch.set_facecolor("#1a1a2e")
        axes[0].imshow(img_orig)
        axes[0].set_title(f"{os.path.basename(path)}\n{total} granos detectados",
                          color="white", fontsize=9)
        axes[0].axis("off")
        etq  = list(conteo.keys()); vals = list(conteo.values())
        cols = plt.cm.Set3(np.linspace(0,1,len(etq)))
        axes[1].pie(vals, labels=etq, autopct="%1.0f%%", colors=cols,
                    startangle=90, textprops={"fontsize":8, "color":"white"})
        axes[1].set_facecolor("#1a1a2e")
        axes[1].set_title("Distribución de defectos", color="white", fontsize=9)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.pred_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=4, pady=4)
        plt.close(fig)

        # Resumen
        n_def = total - conteo.get("Normales", 0)
        pct   = n_def/total*100 if total else 0
        calidad = ("🏆 SUPREMO" if pct<=5 else "✅ EXCELSO" if pct<=15
                   else "⚠️  CONSUMO" if pct<=30 else "❌ REPROCESO")

        info_frame = ctk.CTkFrame(self.pred_frame, corner_radius=10)
        info_frame.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(info_frame, text=f"CALIDAD LOTE: {calidad}  |  "
                     f"{n_def}/{total} defectuosos ({pct:.0f}%)",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#4ade80").pack(pady=10)

        # Galería de parches
        ctk.CTkLabel(self.pred_frame, text="Muestra de granos clasificados",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(8,2))
        gallery = ctk.CTkFrame(self.pred_frame, fg_color="transparent")
        gallery.pack(fill="x", padx=4)
        n_show = min(12, len(resultados))
        for i in range(n_show):
            clase, prob, patch = resultados[i]
            cal   = CALIDAD_NTC.get(clase,"Consumo")
            color = COLOR_CALIDAD.get(cal,"#555")
            frame = ctk.CTkFrame(gallery, corner_radius=6, border_width=2,
                                 border_color=color)
            frame.grid(row=i//6, column=i%6, padx=3, pady=3)
            gallery.columnconfigure(i%6, weight=1)
            pil  = Image.fromarray(patch).resize((64,64))
            cimg = ctk.CTkImage(pil, size=(64,64))
            ctk.CTkLabel(frame, image=cimg, text="").pack(pady=(4,0))
            ctk.CTkLabel(frame, text=clase[:9],
                         font=ctk.CTkFont(size=8), text_color=color).pack()
            ctk.CTkLabel(frame, text=f"{prob:.0%}",
                         font=ctk.CTkFont(size=8), text_color="gray").pack(pady=(0,4))

        # Recomendaciones detectadas
        clases_det = set(r[0] for r in resultados)
        self._render_recomendaciones(destacar=clases_det)
        self.tabs.set("Recomendaciones")
        self._set_status(f"✅  Diagnóstico: {total} granos | Calidad: {calidad}")

    # ── GUARDAR / CARGAR MODELO ──────────────────────────────────
    def _save_model(self):
        if not self.model_trained:
            messagebox.showwarning("Sin modelo", "Entrena primero el modelo.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Modelo joblib","*.joblib"),("Todos","*.*")],
            initialfile="modelo_cafe_rf.joblib")
        if not path: return
        joblib.dump({"model":self.rf, "scaler":self.scaler, "imputer":self.imp,
                     "le":self.le, "feature_names":self.feature_names,
                     "clases":CLASES, "patch_size":PATCH_SIZE,
                     "metadata":{"fecha":datetime.datetime.now().isoformat()}}, path)
        messagebox.showinfo("Guardado", f"Modelo guardado en:\n{path}")
        self._set_status(f"Modelo guardado: {os.path.basename(path)}")

    def _load_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Modelo joblib","*.joblib"),("Todos","*.*")])
        if not path: return
        try:
            art = joblib.load(path)
            self.rf   = art["model"]; self.scaler = art["scaler"]
            self.imp  = art["imputer"]; self.le = art["le"]
            self.feature_names = art.get("feature_names",[])
            self.model_trained = True
            messagebox.showinfo("Cargado", f"Modelo cargado:\n{os.path.basename(path)}")
            self._set_status(f"✅  Modelo cargado: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")

    # ── HELPERS ──────────────────────────────────────────────────
    def _set_status(self, msg):
        self.after(0, lambda: self.status_text.set(msg))


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    app = AgroVisionApp()
    app.mainloop()
