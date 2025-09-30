import streamlit as st
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import pandas as pd
from io import BytesIO
import zipfile
from PIL import Image
import base64
import datetime as dt

# =============================
# Helpers
# =============================

def read_dicom_bytes(file_like):
    """Return (dataset, errors) for a single DICOM file-like object."""
    try:
        ds = pydicom.dcmread(file_like, force=True)
        return ds, None
    except Exception as e:
        return None, str(e)


def is_multiframe(ds):
    return hasattr(ds, "NumberOfFrames") and int(getattr(ds, "NumberOfFrames", 1)) > 1


def get_instance_number(ds):
    try:
        return int(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0


def get_series_uid(ds):
    return str(getattr(ds, "SeriesInstanceUID", ""))


def get_patient_name(ds):
    try:
        pn = getattr(ds, "PatientName", "")
        return pn.original_string.decode(errors="ignore") if hasattr(pn, 'original_string') else str(pn)
    except Exception:
        return ""


def get_pixel_array(ds, frame_index=0):
    """Returns a numpy array in Hounsfield/linear space (no windowing) handling slope/intercept and VOI LUT option."""
    # Some compressed images require extra libs (pylibjpeg). If unavailable, this may raise.
    arr = ds.pixel_array if not is_multiframe(ds) else ds.pixel_array[frame_index]
    # Apply Rescale Slope/Intercept if present (CT/MR etc.)
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
    intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
    arr = arr * slope + intercept
    return arr


def default_window(ds, arr):
    """Return (center, width) from tags if present, else from data range."""
    def _to_float(x):
        try:
            if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                x = x[0]
            return float(x)
        except Exception:
            return None

    wc = _to_float(getattr(ds, 'WindowCenter', None))
    ww = _to_float(getattr(ds, 'WindowWidth', None))
    if wc is None or ww is None or ww == 0:
        vmin, vmax = np.nanpercentile(arr, [0.5, 99.5]) if np.isfinite(arr).all() else (np.nanmin(arr), np.nanmax(arr))
        wc = (vmax + vmin) / 2.0
        ww = (vmax - vmin)
        if ww == 0:
            ww = 1
    return wc, ww


def window_image(arr, center, width):
    """Window/level to 8-bit image."""
    lower = center - width / 2
    upper = center + width / 2
    arr_clip = np.clip(arr, lower, upper)
    img = ((arr_clip - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return img


def to_png_bytes(img_arr):
    im = Image.fromarray(img_arr)
    bio = BytesIO()
    im.save(bio, format='PNG')
    bio.seek(0)
    return bio


def anonymize_dataset(ds: pydicom.dataset.FileDataset) -> pydicom.dataset.FileDataset:
    ds = ds.copy()
    # Remove private tags & standard PHI
    ds.remove_private_tags()
    # Common identifying elements
    for tag in [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0008, 0x0050),  # AccessionNumber
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x0081),  # InstitutionAddress
        (0x0010, 0x1000),  # OtherPatientIDs
        (0x0010, 0x1001),  # OtherPatientNames
        (0x0010, 0x2160),  # EthnicGroup
        (0x0010, 0x4000),  # PatientComments
        (0x0008, 0x1030),  # StudyDescription (optional)
        (0x0008, 0x103E),  # SeriesDescription (optional)
        (0x0008, 0x0020),  # StudyDate
        (0x0008, 0x0030),  # StudyTime
        (0x0008, 0x0021),  # SeriesDate
        (0x0008, 0x0031),  # SeriesTime
        (0x0008, 0x0022),  # AcquisitionDate
        (0x0008, 0x0032),  # AcquisitionTime
        (0x0008, 0x0023),  # ContentDate
        (0x0008, 0x0033),  # ContentTime
        (0x0010, 0x1010),  # PatientAge
        (0x0010, 0x1020),  # PatientSize
        (0x0010, 0x1030),  # PatientWeight
    ]:
        if tag in ds:
            del ds[tag]
    # Overwrite with generic identifiers
    ds.PatientName = "ANONYMIZED"
    ds.PatientID = "000000"
    ds.StudyInstanceUID = getattr(ds, 'StudyInstanceUID', None) or pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = getattr(ds, 'SeriesInstanceUID', None) or pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    # Optionally shift dates consistently (here: remove)
    for tag in [(0x0008,0x0020),(0x0008,0x0021),(0x0008,0x0022),(0x0008,0x0023),(0x0010,0x0030)]:
        if tag in ds:
            del ds[tag]
    return ds


def datasets_to_zip(datasets, filename_prefix="anon"):
    memzip = BytesIO()
    with zipfile.ZipFile(memzip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i, ds in enumerate(datasets, start=1):
            sop = getattr(ds, 'SOPInstanceUID', None) or f"{i:04d}"
            name = f"{filename_prefix}_{sop}.dcm"
            bio = BytesIO()
            ds.save_as(bio)
            bio.seek(0)
            zf.writestr(name, bio.read())
    memzip.seek(0)
    return memzip


def build_index(datasets, file_names):
    rows = []
    for ds, fname in zip(datasets, file_names):
        try:
            rows.append({
                'file_name': fname,
                'PatientName': get_patient_name(ds),
                'PatientID': str(getattr(ds, 'PatientID', '')),
                'StudyDate': str(getattr(ds, 'StudyDate', '')),
                'Modality': str(getattr(ds, 'Modality', '')),
                'StudyDescription': str(getattr(ds, 'StudyDescription', '')),
                'SeriesDescription': str(getattr(ds, 'SeriesDescription', '')),
                'SeriesInstanceUID': get_series_uid(ds),
                'InstanceNumber': get_instance_number(ds),
                'SOPInstanceUID': str(getattr(ds, 'SOPInstanceUID', '')),
                'Rows': int(getattr(ds, 'Rows', 0) or 0),
                'Columns': int(getattr(ds, 'Columns', 0) or 0),
                'PixelSpacing': str(getattr(ds, 'PixelSpacing', '')),
                'NumberOfFrames': int(getattr(ds, 'NumberOfFrames', 1) or 1),
            })
        except Exception:
            # Best-effort; skip corrupt
            continue
    df = pd.DataFrame(rows)
    # Sort by series then instance
    if not df.empty:
        df = df.sort_values(by=['SeriesInstanceUID', 'InstanceNumber']).reset_index(drop=True)
    return df


# =============================
# App UI
# =============================

st.set_page_config(page_title="DICOM Viewer & Manager", layout="wide")
st.title("🩻 DICOM Viewer & Manager")
st.caption("Carica file DICOM singoli o uno ZIP di una cartella di studio/serie. Visualizza, ispeziona metadati, anonimizza ed esporta immagini.")

with st.sidebar:
    st.header("Caricamento file")
    files = st.file_uploader("Seleziona uno o più file .dcm o uno .zip", type=["dcm", "dicom", "zip"], accept_multiple_files=True)
    st.markdown("Se carichi uno ZIP, verranno letti tutti i file *.dcm al suo interno.")

    st.divider()
    st.header("Visualizzazione")
    invert = st.checkbox("Inverti (bianco/nero)", value=False)
    apply_voi = st.checkbox("Usa VOI LUT se presente", value=True)

    st.divider()
    st.header("Esportazione")
    want_png = st.checkbox("Scarica PNG dell'immagine corrente", value=False)
    want_meta = st.checkbox("Scarica metadati (CSV)", value=False)
    want_anon = st.checkbox("Scarica DICOM anonimizzati (ZIP)", value=False)

# Load datasets
all_datasets = []
all_filenames = []
errors = []

if files:
    for up in files:
        name = up.name
        if name.lower().endswith('.zip'):
            try:
                zf = zipfile.ZipFile(up)
                for info in zf.infolist():
                    if info.filename.lower().endswith(('.dcm', '.dicom')):
                        with zf.open(info) as f:
                            ds, err = read_dicom_bytes(BytesIO(f.read()))
                            if ds is not None:
                                all_datasets.append(ds)
                                all_filenames.append(info.filename)
                            else:
                                errors.append((name, err))
            except Exception as e:
                errors.append((name, str(e)))
        else:
            ds, err = read_dicom_bytes(up)
            if ds is not None:
                all_datasets.append(ds)
                all_filenames.append(name)
            else:
                errors.append((name, err))

if errors:
    with st.expander("Errori di lettura", expanded=False):
        for nm, er in errors:
            st.error(f"{nm}: {er}")

if not all_datasets:
    st.info("Carica dei DICOM per iniziare.")
    st.stop()

# Build and show index
index_df = build_index(all_datasets, all_filenames)

with st.expander("📋 Elenco serie e istanze (filtrabile)", expanded=True):
    if index_df.empty:
        st.warning("Nessun DICOM valido trovato nel caricamento.")
    else:
        # Simple filters
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            f_mod = st.text_input("Filtro Modality", "")
        with c2:
            f_pat = st.text_input("Filtro PatientID/Name", "")
        with c3:
            f_series = st.text_input("Filtro SeriesDescription", "")
        with c4:
            f_study = st.text_input("Filtro StudyDate", "")
        fdf = index_df.copy()
        if f_mod:
            fdf = fdf[fdf['Modality'].str.contains(f_mod, case=False, na=False)]
        if f_pat:
            mask = fdf['PatientID'].str.contains(f_pat, case=False, na=False) | fdf['PatientName'].str.contains(f_pat, case=False, na=False)
            fdf = fdf[mask]
        if f_series:
            fdf = fdf[fdf['SeriesDescription'].str.contains(f_series, case=False, na=False)]
        if f_study:
            fdf = fdf[fdf['StudyDate'].str.contains(f_study, case=False, na=False)]
        st.dataframe(fdf, use_container_width=True, hide_index=True)

# Select series and instance
all_series = index_df['SeriesInstanceUID'].unique().tolist() if not index_df.empty else []
sel_series = st.selectbox("Scegli la Serie", options=all_series, format_func=lambda uid: f"{uid[:12]}..." if len(uid) > 12 else uid)

series_df = index_df[index_df['SeriesInstanceUID'] == sel_series].sort_values('InstanceNumber')
instances = series_df.index.tolist()
if not instances:
    st.warning("Serie vuota.")
    st.stop()

# Map index to dataset
idx_to_ds = {i: all_datasets[i] for i in range(len(all_datasets))}

# Build a list of dataset indices for the selected series
series_ds_list = []
for _, row in series_df.iterrows():
    # locate the dataset matching SOPInstanceUID among all_datasets
    target_uid = row['SOPInstanceUID']
    for i, ds in enumerate(all_datasets):
        if str(getattr(ds, 'SOPInstanceUID', '')) == target_uid:
            series_ds_list.append((i, ds))
            break

# Choose instance and frame
inst_pos = st.slider("Istanze nella serie", 1, len(series_ds_list), 1)
sel_idx, sel_ds = series_ds_list[inst_pos - 1]

n_frames = int(getattr(sel_ds, 'NumberOfFrames', 1) or 1)
frame = 1
if n_frames > 1:
    frame = st.slider("Frame (multiframe)", 1, n_frames, 1)

# Prepare image
try:
    arr = get_pixel_array(sel_ds, frame_index=frame-1)
except Exception as e:
    st.error(f"Impossibile decodificare pixel: {e}\nPotrebbero servire librerie aggiuntive per compressioni (es. pylibjpeg).")
    st.stop()

# Default window from dataset or data
wc_default, ww_default = default_window(sel_ds, arr)

c1, c2, c3 = st.columns([2,2,1])
with c1:
    wc = st.number_input("Window Center", value=float(np.round(wc_default, 2)))
with c2:
    ww = st.number_input("Window Width", value=float(np.round(ww_default, 2)), min_value=0.1)
with c3:
    reset = st.button("Reset window")
    if reset:
        wc, ww = wc_default, ww_default

img8 = window_image(arr, wc, ww)
if invert:
    img8 = 255 - img8

# If RGB photometric interpretation, bypass windowing and show directly
photometric = str(getattr(sel_ds, 'PhotometricInterpretation', '')).upper()
if photometric in ("RGB", "YBR_FULL", "YBR_FULL_422") and len(sel_ds.pixel_array.shape) == 3:
    # Convert to RGB 8-bit
    try:
        rgb = sel_ds.pixel_array
        if rgb.dtype != np.uint8:
            # scale if necessary
            rgb = (255 * (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)).astype(np.uint8)
        disp_img = rgb
    except Exception:
        disp_img = img8
else:
    disp_img = img8

st.image(disp_img, caption=f"Serie {sel_series} • Istanza {inst_pos}/{len(series_ds_list)} • Frame {frame}/{n_frames}", use_column_width=True, clamp=True)

# Metadata viewer
with st.expander("🔎 Metadati DICOM (dataset completo)", expanded=False):
    try:
        items = []
        for elem in sel_ds:
            try:
                val = elem.value
                if isinstance(val, (bytes, bytearray)) and len(val) > 64:
                    val = f"<binary {len(val)} bytes>"
                items.append({"Tag": f"({elem.tag.group:04X},{elem.tag.element:04X})", "Name": elem.name, "VR": elem.VR, "Value": str(val)})
            except Exception:
                continue
        meta_df = pd.DataFrame(items)
        st.dataframe(meta_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Impossibile mostrare metadati: {e}")

# Export downloads
colA, colB, colC = st.columns(3)

if want_png:
    png_bytes = to_png_bytes(disp_img)
    with colA:
        st.download_button(
            label="⬇️ Scarica PNG corrente",
            data=png_bytes,
            file_name=f"dicom_slice_{inst_pos:03d}_frame{frame}.png",
            mime="image/png",
        )

if want_meta:
    csv = index_df.to_csv(index=False).encode("utf-8")
    with colB:
        st.download_button(
            label="⬇️ Scarica metadati (CSV)",
            data=csv,
            file_name="dicom_metadata.csv",
            mime="text/csv",
        )

if want_anon:
    # Anonymize only selected series
    anon_list = []
    for _, ds in series_ds_list:
        try:
            anon_list.append(anonymize_dataset(ds))
        except Exception:
            continue
    if anon_list:
        zipbio = datasets_to_zip(anon_list, filename_prefix="anon")
        with colC:
            st.download_button(
                label="⬇️ Scarica DICOM anonimizzati (ZIP)",
                data=zipbio,
                file_name=f"anon_series_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )
    else:
        with colC:
            st.warning("Nessun file anonimizzato disponibile.")

# Footer
st.caption("Note: per immagini compresse (JPEG2000, RLE, etc.) potrebbe essere necessario installare librerie aggiuntive: 'pylibjpeg', 'pylibjpeg-libjpeg' o 'pylibjpeg-openjpeg'.")

# =============================
# How to run (comment only)
# =============================
# 1) pip install streamlit pydicom pillow numpy pandas
#    (opzionale per formati compressi: pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg)
# 2) streamlit run app.py
