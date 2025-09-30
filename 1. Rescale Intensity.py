# safe_scale_sar.py
# === EDITA AQUÍ ===
IN  = r"C:\Users\aleja\Desktop\Parcial IA\S1C_IW_GRDH_1SDH_20250826T102230_20250826T102255_003842_007A99_D784.SAFE\measurement\s1c-iw-grd-hv-20250826t102230-20250826t102255-003842-007a99-002.tiff"
OUT = r"C:\Users\aleja\Desktop\Parcial IA\ScaledImages\s1c-iw-grd-hv-20250826t102230-20250826t102255-003842-007a99-002_scaled.tif"
# ==========================
FILTER   = "lee"   # "lee" o "frost"
WIN      = 7       # impar: 3,5,7,9
DAMPING  = 2.0     # solo frost
TILE     = 1024    # tamaño de bloque (reduce si tienes poca RAM)

import os, numpy as np, rasterio as rio, cv2
from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning
import warnings; warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def _local_stats(img, k):
    m  = cv2.boxFilter(img, ddepth=-1, ksize=(k,k), normalize=True, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(img*img, ddepth=-1, ksize=(k,k), normalize=True, borderType=cv2.BORDER_REFLECT)
    v  = np.clip(m2 - m*m, 0, None).astype(np.float32)
    return m, v

def lee_filter(img, k=7):
    m, v = _local_stats(img, k)
    flat = v[np.isfinite(v)]
    sig2 = np.percentile(flat, 10) if flat.size else 0.0
    W = v / (v + sig2 + 1e-12)
    return m + W*(img - m)

def frost_filter(img, k=7, a=2.0):
    m, v = _local_stats(img, k)
    sigma = np.sqrt(v)
    eps = 1e-12
    alpha = a * ((sigma+eps)/(m+eps))**2
    diff = np.abs(img - m)/(m+eps)
    w = np.exp(-alpha*diff)
    num = cv2.boxFilter(w*img, ddepth=-1, ksize=(k,k), normalize=False, borderType=cv2.BORDER_REFLECT)
    den = cv2.boxFilter(w,     ddepth=-1, ksize=(k,k), normalize=False, borderType=cv2.BORDER_REFLECT)
    return num/(den+eps)

def _detect_db(sample):
    p5, p95 = np.nanpercentile(sample, [5,95])
    return (-60 < p5 < 30) and (-60 < p95 < 30)

def process():
    if WIN % 2 == 0 or WIN < 3: raise SystemExit("WIN debe ser impar y >=3.")
    pad = WIN//2
    if not os.path.isfile(IN): raise SystemExit(f"No existe: {IN}")

    with rio.open(IN) as src:
        prof = src.profile.copy()
        H, W, C = src.height, src.width, src.count
        nodata = prof.get("nodata", None)

        # muestreo pequeño para detectar dB
        sw = Window(int(max(0,W/2-128)), int(max(0,H/2-128)), min(256,W), min(256,H))
        samp = src.read(1, window=sw).astype(np.float32)
        if nodata is not None: samp = np.where(samp==nodata, np.nan, samp)
        is_db = _detect_db(samp)

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        prof.update(dtype="float32", compress="deflate", predictor=3, tiled=True, nodata=nodata)
        with rio.open(OUT, "w", **prof) as dst:
            # recorrer por bloques
            for y0 in range(0, H, TILE):
                for x0 in range(0, W, TILE):
                    h = min(TILE, H - y0)
                    w = min(TILE, W - x0)

                    # ventana extendida con padding para bordes del filtro
                    yb0 = max(0, y0 - pad); yb1 = min(H, y0 + h + pad)
                    xb0 = max(0, x0 - pad); xb1 = min(W, x0 + w + pad)
                    win_big = Window(xb0, yb0, xb1-xb0, yb1-yb0)

                    block = src.read(window=win_big, out_dtype="float32")  # (C, hb, wb)
                    if nodata is not None:
                        block = np.where(block==nodata, np.nan, block)

                    # convertir a lineal si está en dB
                    block = 10.0**(block/10.0) if is_db else block

                    # recorte útil dentro del bloque grande
                    y1 = y0 - yb0; y2 = y1 + h
                    x1 = x0 - xb0; x2 = x1 + w

                    out_block = np.empty((C, h, w), np.float32)
                    for b in range(C):
                        band = block[b]
                        valid = np.isfinite(band)
                        band = np.where(valid, band, 0.0).astype(np.float32)

                        if FILTER=="lee":
                            filt = lee_filter(band, k=WIN)
                        elif FILTER=="frost":
                            filt = frost_filter(band, k=WIN, a=DAMPING)
                        else:
                            raise SystemExit("Filtro no soportado.")

                        filt = filt[y1:y2, x1:x2]
                        if nodata is not None:
                            valid_crop = valid[y1:y2, x1:x2]
                            filt = np.where(valid_crop, filt, np.nan)
                        out_block[b] = filt

                    # regresar a dB si venía en dB
                    if is_db:
                        out_block = 10.0*np.log10(np.clip(out_block, 1e-12, None))

                    dst.write(out_block, window=Window(x0, y0, w, h))
    print("OK ->", OUT)

if __name__ == "__main__":
    try:
        process()
    except Exception as e:
        raise SystemExit(f"Error: {e}")