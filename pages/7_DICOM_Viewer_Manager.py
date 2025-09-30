import streamlit as st
import pydicom
from pydicom.uid import generate_uid
import numpy as np
import pandas as pd
from io import BytesIO
import zipfile
from PIL import Image
import time

# =====================================================
# Radiant‑like DICOM Viewer (Streamlit)
# Core features inspired by RadiAnt: series browser, stack scroll/cine,
# window/level presets, zoom/pan/ROI, basic measurements, MPR (axial/coronal/sagittal),
# anonymize & export. No GPU or external components required.
# =====================================================

st.set_page_config(page_title="🩻 Radiant‑like DICOM Viewer", layout="wide")
st.title("🩻 Radiant‑like DICOM Viewer")
st.caption("Carica serie DICOM, visualizza stack con window/level e preset CT, esegui MPR, misure base, cine e anonimizzazione.")

# --------------------
# Helpers
# --------------------

def read_dicom(file_like):
    try:
        ds = pydicom.dcmread(file_like, force=True)
        return ds, None
    except Exception as e:
        return None, str(e)


def from_zip(upload):
    out = []
    zf = zipfile.ZipFile(upload)
    for info in zf.infolist():
        if info.filename.lower().endswith((".dcm", ".dicom")):
            try:
                with zf.open(info) as f:
                    ds, err = read_dicom(BytesIO(f.read()))
                    if ds is not None:
                        out.append((info.filename, ds))
            except Exception:
                continue
    return out


def get_series_key(ds):
    return (
        str(getattr(ds, "StudyInstanceUID", "")),
        str(getattr(ds, "SeriesInstanceUID", "")),
    )


def get_instance_position(ds):
    # Prefer ImagePositionPatient (z), fallback to InstanceNumber
    ipp = getattr(ds, "ImagePositionPatient", None)
    if isinstance(ipp, (list, tuple)) and len(ipp) == 3:
        return float(ipp[2])
    try:
        return float(getattr(ds, "SliceLocation", float(getattr(ds, "InstanceNumber", 0))))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0) or 0)


def slope_intercept(ds):
    slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
    intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
    return slope, intercept


def pixel_array(ds):
    arr = ds.pixel_array.astype(np.float32)
    s, b = slope_intercept(ds)
    return arr * s + b


def wl_to_uint8(arr, wc, ww):
    lower = wc - ww/2.0
    upper = wc + ww/2.0
    x = np.clip((arr - lower) / (upper - lower), 0, 1)
    return (x * 255).astype(np.uint8)


def default_wc_ww(arr):
    vmin, vmax = np.percentile(arr, [0.5, 99.5]) if np.all(np.isfinite(arr)) else (np.nanmin(arr), np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmin == vmax:
            vmax = vmin + 1
    wc = (vmax + vmin) / 2.0
    ww = (vmax - vmin)
    return float(wc), float(ww)

CT_PRESETS = {
    "Default": None,              # auto from data
    "Brain": (40, 80),            # wc, ww
    "Lung": (-600, 1500),
    "Mediastinum": (40, 400),
    "Bone": (300, 1500),
}


def apply_brightness_contrast(img8, brightness=0, contrast=1.0):
    out = img8.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(out, 0, 255).astype(np.uint8)


def zoom_resize(img8, factor=1.0):
    if factor == 1.0:
        return img8
    h, w = img8.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    return np.array(Image.fromarray(img8).resize((new_w, new_h), Image.BILINEAR))


def crop_roi(img8, cx=0.5, cy=0.5, wp=100, hp=100):
    h, w = img8.shape[:2]
    rw = int(np.clip(wp, 1, 100) / 100.0 * w)
    rh = int(np.clip(hp, 1, 100) / 100.0 * h)
    cx_px = int(np.clip(cx, 0, 1) * w)
    cy_px = int(np.clip(cy, 0, 1) * h)
    x1 = int(np.clip(cx_px - rw//2, 0, w-1))
    y1 = int(np.clip(cy_px - rh//2, 0, h-1))
    x2 = int(np.clip(x1 + rw, 1, w))
    y2 = int(np.clip(y1 + rh, 1, h))
    return img8[y1:y2, x1:x2]


def anonymize(ds):
    ds = ds.copy()
    ds.remove_private_tags()
    for tag in [
        (0x0010,0x0010),(0x0010,0x0020),(0x0010,0x0030),(0x0010,0x0040),
        (0x0008,0x0090),(0x0008,0x0050),(0x0008,0x0080),(0x0008,0x0081),
        (0x0010,0x1000),(0x0010,0x1001),(0x0010,0x2160),(0x0010,0x4000),
        (0x0008,0x1030),(0x0008,0x103E),(0x0008,0x0020),(0x0008,0x0030),
        (0x0008,0x0021),(0x0008,0x0031),(0x0008,0x0022),(0x0008,0x0032),
        (0x0008,0x0023),(0x0008,0x0033),(0x0010,0x1010)
    ]:
        if tag in ds:
            del ds[tag]
    ds.PatientName = "ANONYMIZED"
    ds.PatientID = "000000"
    ds.StudyInstanceUID = getattr(ds, 'StudyInstanceUID', None) or generate_uid()
    ds.SeriesInstanceUID = getattr(ds, 'SeriesInstanceUID', None) or generate_uid()
    ds.SOPInstanceUID = generate_uid()
    return ds


def zip_datasets(datasets, name_prefix="anon"):
    mem = BytesIO()
    with zipfile.ZipFile(mem, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i, ds in enumerate(datasets, start=1):
            bio = BytesIO(); ds.save_as(bio); bio.seek(0)
            zf.writestr(f"{name_prefix}_{i:04d}.dcm", bio.read())
    mem.seek(0)
    return mem

# --------------------
# Load files
# --------------------
with st.sidebar:
    st.header("Caricamento")
    ups = st.file_uploader("DICOM singoli o ZIP di una serie", type=["dcm","dicom","zip"], accept_multiple_files=True)
    st.caption("Suggerimento: carica tutte le immagini di una stessa serie per abilitarne MPR e scorrimento.")

all_files = []
errors = []
if ups:
    for up in ups:
        if up.name.lower().endswith('.zip'):
            all_files.extend(from_zip(up))
        else:
            ds, er = read_dicom(up)
            if ds is not None:
                all_files.append((up.name, ds))
            elif er:
                errors.append((up.name, er))

if errors:
    with st.expander("Errori di lettura", expanded=False):
        for nm, er in errors:
            st.error(f"{nm}: {er}")

if not all_files:
    st.info("Carica dei DICOM per iniziare.")
    st.stop()

# Group by series
series_map = {}
for nm, ds in all_files:
    key = get_series_key(ds)
    series_map.setdefault(key, []).append((nm, ds))

series_keys = list(series_map.keys())

# Sidebar: choose Study/Series
with st.sidebar:
    st.header("Serie")
    sel_idx = st.selectbox("Seleziona serie", options=list(range(len(series_keys))), format_func=lambda i: f"Study={series_keys[i][0][-8:]} | Series={series_keys[i][1][-8:]} ({len(series_map[series_keys[i]])} img)")
    sel_key = series_keys[sel_idx]

# Sort instances by position
pairs = series_map[sel_key]
sorted_pairs = sorted(pairs, key=lambda x: get_instance_position(x[1]))
instances = [ds for _, ds in sorted_pairs]

# Build a 3D volume if same shape
same_size = all((hasattr(d, 'Rows') and hasattr(d, 'Columns') and (d.Rows, d.Columns) == (instances[0].Rows, instances[0].Columns)) for d in instances)
volume = None
spacing = (1.0, 1.0, 1.0)  # (dz, dy, dx)
if same_size and len(instances) >= 2:
    try:
        # Stack along z (instance order)
        stack = [pixel_array(d) for d in instances]
        volume = np.stack(stack, axis=0)  # (z, y, x)
        # spacing from DICOM
        dy, dx = (1.0, 1.0)
        ps = getattr(instances[0], 'PixelSpacing', None)
        if isinstance(ps, (list, tuple)) and len(ps) == 2:
            dy, dx = float(ps[0]), float(ps[1])
        # dz from slice spacing
        zs = [get_instance_position(d) for d in instances]
        dz = np.median(np.diff(sorted(zs))) if len(zs) > 1 else 1.0
        spacing = (abs(dz), dy, dx)
    except Exception:
        volume = None

# --------------------
# Controls
# --------------------
with st.sidebar:
    st.header("Visualizzazione")
    modality = str(getattr(instances[0], 'Modality', ''))
    preset_name = st.selectbox("Preset CT", options=list(CT_PRESETS.keys()), index=0)
    invert = st.checkbox("Inverti", value=False)
    zoom = st.slider("Zoom", 0.25, 4.0, 1.0, 0.05)
    bright = st.slider("Luminosità", -128, 128, 0, 1)
    contr = st.slider("Contrasto", 0.1, 3.0, 1.0, 0.05)

    st.subheader("ROI / Pan")
    enable_roi = st.checkbox("Abilita ROI", value=False)
    if enable_roi:
        c1, c2 = st.columns(2)
        with c1:
            roi_w = st.slider("Larghezza %", 5, 100, 80)
            cx = st.slider("Centro X %", 0, 100, 50) / 100.0
        with c2:
            roi_h = st.slider("Altezza %", 5, 100, 80)
            cy = st.slider("Centro Y %", 0, 100, 50) / 100.0
    else:
        roi_w, roi_h, cx, cy = 100, 100, 0.5, 0.5

    st.subheader("Stack / Cine")
    if 'slice_idx' not in st.session_state:
        st.session_state.slice_idx = max(0, len(instances)//2)
    max_slice = max(0, len(instances) - 1)
    st.session_state.slice_idx = st.slider("Slice", 0, max_slice, st.session_state.slice_idx) if max_slice > 0 else 0
    play = st.toggle("Cine ▶", value=False)
    fps = st.slider("FPS", 1, 30, 10)

    st.subheader("Misure")
    pxsp = getattr(instances[0], 'PixelSpacing', [1.0, 1.0])
    py, px = float(pxsp[0]), float(pxsp[1]) if isinstance(pxsp, (list, tuple)) else (1.0, 1.0)
    st.caption(f"PixelSpacing: {pxsp}")
    x1 = st.number_input("x1 (px)", 0, int(getattr(instances[0],'Columns',512))-1, 10)
    y1 = st.number_input("y1 (px)", 0, int(getattr(instances[0],'Rows',512))-1, 10)
    x2 = st.number_input("x2 (px)", 0, int(getattr(instances[0],'Columns',512))-1, 100)
    y2 = st.number_input("y2 (px)", 0, int(getattr(instances[0],'Rows',512))-1, 100)
    dist_mm = np.sqrt(((x2-x1)*px)**2 + ((y2-y1)*py)**2)
    st.caption(f"📏 Distanza: {dist_mm:.2f} mm")

# Cine autorefresh
if play and len(instances) > 1:
    st.session_state.slice_idx = (st.session_state.slice_idx + 1) % len(instances)
    st.autorefresh(interval=int(1000/fps), key="cine")

# --------------------
# View selection & rendering
# --------------------
view = st.radio("Vista", options=["Axial", "Coronal", "Sagittal"], horizontal=True)

if view == "Axial":
    arr = pixel_array(instances[st.session_state.slice_idx])
elif view == "Coronal" and volume is not None:
    arr = volume[:, :, :]
    arr = arr[:, :, :]  # no-op, clarity
    # coronal slice index from y (rows)
    y_max = volume.shape[1]-1
    if 'y_idx' not in st.session_state:
        st.session_state.y_idx = y_max//2
    st.session_state.y_idx = st.slider("Coronal index", 0, y_max, st.session_state.y_idx)
    arr = volume[:, st.session_state.y_idx, :].T  # shape (x, z) -> transpose to display correctly
elif view == "Sagittal" and volume is not None:
    x_max = volume.shape[2]-1
    if 'x_idx' not in st.session_state:
        st.session_state.x_idx = x_max//2
    st.session_state.x_idx = st.slider("Sagittal index", 0, x_max, st.session_state.x_idx)
    arr = volume[:, :, st.session_state.x_idx]
    arr = arr.T
else:
    # Fallback to axial if no volume
    arr = pixel_array(instances[st.session_state.slice_idx])

# Window/level
if preset_name != "Default" and modality == "CT":
    wc, ww = CT_PRESETS[preset_name]
else:
    wc, ww = default_wc_ww(arr)

img8 = wl_to_uint8(arr, wc, ww)
if invert:
    img8 = 255 - img8

# ROI, brightness/contrast, zoom
roi_img = crop_roi(img8, cx=cx, cy=cy, wp=roi_w, hp=roi_h) if enable_roi else img8
adj = apply_brightness_contrast(roi_img, brightness=bright, contrast=contr)
adj = zoom_resize(adj, factor=zoom)

# Show image
cap = f"{view} | slice {st.session_state.get('slice_idx',0)+1}/{len(instances)} | zoom {zoom:.2f}x | preset {preset_name}"
st.image(adj, caption=cap, use_column_width=True, clamp=True)

# ROI stats (area, mean, std)
with st.expander("📊 ROI stats (sull'immagine finestrata)"):
    roi_stats = {
        "shape": adj.shape,
        "min": int(np.min(adj)),
        "max": int(np.max(adj)),
        "mean": float(np.mean(adj)),
        "std": float(np.std(adj)),
    }
    st.json(roi_stats)

# Metadata table for series
with st.expander("📋 Metadati serie (riassunto)"):
    rows = []
    for ds in instances:
        rows.append({
            'SOP': str(getattr(ds, 'SOPInstanceUID',''))[-12:],
            'Instance': int(getattr(ds, 'InstanceNumber', 0) or 0),
            'Rows': int(getattr(ds, 'Rows', 0) or 0),
            'Cols': int(getattr(ds, 'Columns', 0) or 0),
            'SliceLoc/Z': float(get_instance_position(ds)),
        })
    df = pd.DataFrame(rows).sort_values('Instance')
    st.dataframe(df, use_container_width=True, hide_index=True)

# Export anonymized ZIP of current series
anon = st.checkbox("Prepara ZIP anonimizzato della serie corrente", value=False)
if anon:
    anon_list = [anonymize(ds) for ds in instances]
    zipbio = zip_datasets(anon_list, name_prefix="anon")
    st.download_button("⬇️ Scarica ZIP anonimizzato", data=zipbio, file_name="anon_series.zip", mime="application/zip")

st.caption("Nota: per serie compresse (JPEG2000/RLE) installa: pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg. MPR semplice (nearest) senza ricampionamento isotropico.")
