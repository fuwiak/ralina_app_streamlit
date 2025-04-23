import subprocess, sys, importlib, os, tempfile, numpy as np, tensorflow as tf
import streamlit as st

# ─── OpenCV без libGL ────────────────────────────────────────────────────────
def safe_import_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--quiet", "opencv-python-headless==4.7.0.72"])
        return importlib.import_module("cv2")
cv2 = safe_import_cv2()

# ─── константы ───────────────────────────────────────────────────────────────
IMG_SIZE, STEP = 150, 10
COLLAPSE_CLASSES = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER_CLASSES   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

HUMAN_READABLE = {
    # collapse
    "No_Landslide": "Нет обвала",
    "Rockfall": "Камнепад",
    "Earth_Flow": "Земной поток",
    # danger
    "Safe": "Нет опасности",
    "Roads_Damaged": "Разрушены дороги",
    "Houses_Damaged": "Разрушены дома",
}

# ─── загрузка моделей (кэш) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    coll = tf.keras.models.load_model("collapse_model.h5", compile=False) \
           if os.path.exists("collapse_model.h5") else None
    if not os.path.exists("danger_model.h5"):
        st.error("Файл **danger_model.h5** не найден. "
                 "Положите модель рядом с приложением и перезапустите.")
        st.stop()
    dang = tf.keras.models.load_model("danger_model.h5", compile=False)
    return coll, dang
collapse_model, danger_model = load_models()

# ─── классификация одного видео ─────────────────────────────────────────────
def classify_video(model, classes, path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0.0
    votes, fid = np.zeros(len(classes), int), 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step_prog = max(total // 100, 1)
    prog = st.progress(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % STEP == 0:
            fr = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            (IMG_SIZE, IMG_SIZE))
            p = model.predict(np.expand_dims(fr, 0) / 255., verbose=0)[0]
            votes[np.argmax(p)] += 1
        if fid % step_prog == 0:
            prog.progress(min(fid / max(total, 1), 1.0))
        fid += 1
    cap.release(); prog.empty()
    if votes.sum() == 0:
        return None, 0.0
    idx = votes.argmax()
    return classes[idx], votes[idx] / votes.sum()

# ─── rerun для новой загрузки ────────────────────────────────────────────────
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("Обновите страницу вручную.")

# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config("Landslide Detection", "🌋")
st.title("🌋 Landslide Detection")

st.write(
    "1. Загрузите видео (.mp4 /.avi /.mov)\n"
    "2. Нажмите **Классифицировать обвал**\n"
    "3. Если обвал обнаружен — появится кнопка **Оценить опасность**"
)

# ---------- session state ----------
if "video_path" not in st.session_state:
    st.session_state.update(
        video_path=None, collapse_class=None, collapse_conf=None,
        danger_done=False
    )

# ---------- загрузка файла ----------
video_file = st.file_uploader("Видео-файл", type=["mp4", "avi", "mov", "mkv"])
if video_file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read()); tmp.close()
    st.session_state.video_path = tmp.name
    st.video(tmp.name)       # ▶️ показываем превью

# ---------- кнопка 1 — классификация обвала ----------
if st.session_state.video_path and not st.session_state.collapse_class:
    if st.button("🔍 Классифицировать обвал"):
        if collapse_model is None:
            st.warning("Модель collapse_model.h5 отсутствует — "
                       "сразу перейдём к оценке опасности.")
            st.session_state.collapse_class = "No_Landslide"
            st.session_state.collapse_conf = 0
        else:
            c_cls, c_conf = classify_video(
                collapse_model, COLLAPSE_CLASSES, st.session_state.video_path)
            if c_cls is None:
                st.error("Не удалось прочитать видео."); do_rerun()
            st.session_state.collapse_class = c_cls
            st.session_state.collapse_conf = c_conf

# ---------- показываем результат обвала ----------
if st.session_state.collapse_class:
    hr = HUMAN_READABLE[st.session_state.collapse_class]
    st.info(f"**Тип обвала:** {hr}  ({st.session_state.collapse_conf*100:.1f} %)")

# ---------- кнопка 2 — оценка опасности ----------
show_danger_btn = (
    st.session_state.collapse_class and
    st.session_state.collapse_class != "No_Landslide"
)
if show_danger_btn and not st.session_state.danger_done:
    if st.button("⚠️ Оценить опасность"):
        d_cls, d_conf = classify_video(
            danger_model, DANGER_CLASSES, st.session_state.video_path)
        if d_cls is None:
            st.error("Не удалось прочитать видео."); do_rerun()
        hr = HUMAN_READABLE[d_cls]
        st.success(f"**Опасность:** {hr}  ({d_conf*100:.1f} %)")
        st.session_state.danger_done = True

# ---------- если обвала нет ----------
if st.session_state.collapse_class == "No_Landslide":
    st.success("Обвал не обнаружен — опасность не оценивается.")

# ---------- сброс ----------
if st.session_state.video_path:
    st.button("🔄 Загрузить новое видео", on_click=lambda: (
        os.unlink(st.session_state.video_path),
        st.session_state.clear(), do_rerun()
    ))
