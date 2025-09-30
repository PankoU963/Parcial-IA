# despeckle_sar_inline.py
# Edita SOLO estas variables:
IN_PATH  = r"C:\Users\aleja\Desktop\Parcial IA\ScaledImages\s1c-iw-grd-hv-20250826t102230-20250826t102255-003842-007a99-002_scaled.tiff"      # Archivo .tif/.tiff o carpeta con .tif/.tiff
OUT_PATH = r"C:\Users\aleja\Desktop\Parcial IA\FilteredImages\20250826.tif"  # Archivo de salida o carpeta de salida
MODE     = "file"   # "file" para un solo archivo, "dir" para procesar todos los .tif/.tiff de una carpeta
FILTER   = "lee"    # "lee" o "frost"
WIN      = 7        # Ventana impar: 3,5,7,9...
DAMPING  = 2.0      # Solo para frost

import os, glob
import numpy as np
import rasterio as rio
import cv2

import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

EXT_OK = (".tif", ".tiff")

def box_mean(img, k):
    return cv2.boxFilter(img, ddepth=-1, ksize=(k,k), normalize=True, borderType=cv2.BORDER_REFLECT)

def box_mean_sq(img, k):
    return cv2.boxFilter(img*img, ddepth=-1, ksize=(k,k), normalize=True, borderType=cv2.BORDER_REFLECT)

def local_stats(img, k):
    m = box_mean(img, k)
    ex2 = box_mean_sq(img, k)
    v = np.clip(ex2 - m*m, a_min=0, a_max=None)
    return m, v

def estimate_noise_var(var_map, frac=0.1):
    flat = var_map[np.isfinite(var_map)].ravel()
    if flat.size == 0:
        return np.float32(0.0)
    p = np.percentile(flat, frac*100.0)
    return np.float32(p)

def lee_filter(img, k=7, noise_var=None):
    m, v = local_stats(img, k)
    if noise_var is None:
        noise_var = estimate_noise_var(v, frac=0.1)
    W = v / (v + noise_var + 1e-12)
    return m + W * (img - m)

def frost_filter(img, k=7, damping=2.0):
    m, v = local_stats(img, k)
    sigma = np.sqrt(np.clip(v, 0, None))
    eps = 1e-12
    a = damping * ((sigma + eps) / (m + eps))**2
    diff = np.abs(img - m) / (m + eps)
    w = np.exp(-a * diff)
    num = cv2.boxFilter(w*img, ddepth=-1, ksize=(k,k), normalize=False, borderType=cv2.BORDER_REFLECT)
    den = cv2.boxFilter(w,      ddepth=-1, ksize=(k,k), normalize=False, borderType=cv2.BORDER_REFLECT)
    return num / (den + eps)

def read_as_float(band_arr, nodata):
    arr = band_arr.astype(np.float32)
    if nodata is not None:
        mask = (band_arr == nodata)
        arr = np.where(mask, np.nan, arr)
    return arr

def write_with_nodata(data, nodata):
    out = data.copy()
    if nodata is not None:
        out = np.where(np.isnan(out), nodata, out)
    return out.astype(np.float32)

def process_tif(in_path, out_path, filt="lee", k=7, damping=2.0):
    with rio.open(in_path) as src:
        profile = src.profile.copy()
        nodata = profile.get("nodata", None)
        img = src.read()  # (C,H,W)
        img_f = np.stack([read_as_float(b, nodata) for b in img], axis=0)

        # HeurÃ­stica simple para detectar dB
        p5, p95 = np.nanpercentile(img_f, [5, 95])
        is_db = (-60 < p5 < 30) and (-60 < p95 < 30)
        lin = 10.0 ** (img_f / 10.0) if is_db else img_f

        if k % 2 == 0 or k < 3:
            raise ValueError("WIN debe ser impar y >=3.")

        out_lin = np.empty_like(lin, dtype=np.float32)
        for c in range(lin.shape[0]):
            band = lin[c]
            valid = np.isfinite(band)
            band_safe = np.where(valid, band, 0.0).astype(np.float32)

            if filt == "lee":
                filtered = lee_filter(band_safe, k=k)
            elif filt == "frost":
                filtered = frost_filter(band_safe, k=k, damping=damping)
            else:
                raise ValueError("Filtro no soportado. Use 'lee' o 'frost'.")

            out_lin[c] = np.where(valid, filtered, np.nan).astype(np.float32)

        out = 10.0 * np.log10(np.clip(out_lin, 1e-12, None)) if is_db else out_lin

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        profile.update(dtype="float32", compress="deflate", predictor=3, tiled=True)
        with rio.open(out_path, "w", **profile) as dst:
            for i in range(out.shape[0]):
                dst.write(write_with_nodata(out[i], nodata), i+1)

def run_file():
    if not IN_PATH.lower().endswith(EXT_OK):
        raise SystemExit("IN_PATH debe ser un archivo .tif o .tiff")
    if os.path.isdir(OUT_PATH):
        out_path = os.path.join(OUT_PATH, os.path.basename(IN_PATH).rsplit(".",1)[0] + "_despeckle.tif")
    else:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        out_path = OUT_PATH
    print(f"Procesando archivo: {IN_PATH}")
    process_tif(IN_PATH, out_path, filt=FILTER, k=WIN, damping=DAMPING)
    print(f"OK -> {out_path}")

def run_dir():
    if not os.path.isdir(IN_PATH):
        raise SystemExit("IN_PATH debe ser una carpeta cuando MODE='dir'.")
    os.makedirs(OUT_PATH, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(IN_PATH, "*.tif")) +
        glob.glob(os.path.join(IN_PATH, "*.tiff"))
    )
    if not files:
        raise SystemExit("No se encontraron .tif/.tiff en la carpeta de entrada.")
    for p in files:
        name = os.path.basename(p).rsplit(".",1)[0] + "_despeckle.tif"
        out_path = os.path.join(OUT_PATH, name)
        print(f"Procesando: {p}")
        process_tif(p, out_path, filt=FILTER, k=WIN, damping=DAMPING)
        print(f"OK -> {out_path}")
    print("Listo.")

if __name__ == "__main__":
    try:
        if MODE.lower() == "file":
            run_file()
        elif MODE.lower() == "dir":
            run_dir()
        else:
            raise SystemExit("MODE debe ser 'file' o 'dir'.")
    except Exception as e:
        raise SystemExit(f"Error: {e}")
