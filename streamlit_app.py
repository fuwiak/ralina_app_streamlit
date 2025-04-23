import subprocess, sys, importlib, os, tempfile, numpy as np, tensorflow as tf
import streamlit as st

# ────────── безопасный импорт OpenCV ──────────
def safe_import_cv2():
    try:
        return importlib.import_module("cv2")
    except ImportError:
        # пробуем «headless»-колесо (дружит с контейнерами без libGL)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--quiet", "opencv-python-headless==4.7.0.72"
        ])
        return importlib.import_module("cv2")

cv2 = safe_import_cv2()

# ────────── настройки ──────────
IMG_SIZE, STEP = 150, 10
COLLAPSE_CLASSES = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER_CLASSES   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

# ────────── кэш-загрузка моделей ──────────
@st.cache_resource(show_spinner=False)
def load_models():
    coll = tf.keras.models.load_model("collapse_model.h5", compile=False) \
           if os.path.exists("collapse_model.h5") else None
    if not os.path.exists("danger_model.h5"):
        st.error("Файл **danger_model.h5** не найден. "
                 "Положите модель в каталог приложения и перезапустите.")
        st.stop()
    dang = tf.keras.models.load_model("danger_model.h5", compile=False)
    return coll, dang

collapse_model, danger_model = load_models()

# ────────── классификация видео ──────────
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
            pred = model.predict(np.expand_dims(fr, 0) / 255., verbose=0)[0]
            votes[np.argmax(pred)] += 1
        if fid % step_prog == 0:
            prog.progress(min(fid / max(total, 1), 1.0))
        fid += 1

    cap.release(); prog.empty()
    if votes.sum() == 0:
        return None, 0.0
    idx = votes.argmax()
    return classes[idx], votes[idx] / votes.sum()

# ────────── универсальный rerun ──────────
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("⚠️ Не удалось выполнить rerun; обновите страницу вручную.")

# ────────── Streamlit UI ──────────
st.set_page_config("Landslide Detection", "🌋")
st.title("🌋 Landslide Detection")

st.write(
    "Загрузите видео (.mp4 / .avi / .mov). "
    "Если рядом нет `collapse_model.h5`, приложение выполнит только оценку опасности "
    "(Safe / Roads_Damaged / Houses_Damaged)."
)

video_file = st.file_uploader("Видео-файл", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read()); tmp.close()
    st.success(f"Файл сохранён: {video_file.name}")

    if st.button("🔍 Классифицировать"):
        with st.spinner("Обрабатываем…"):
            # 1) тип обвала
            collapse_cls, collapse_conf = None, None
            if collapse_model:
                collapse_cls, collapse_conf = classify_video(
                    collapse_model, COLLAPSE_CLASSES, tmp.name)
                if collapse_cls is None:
                    st.error("Не удалось прочитать видео."); os.unlink(tmp.name); st.stop()
                st.info(f"**Тип обвала:** {collapse_cls}  ({collapse_conf*100:.1f} %)")

            # 2) уровень опасности
            if (collapse_model is None) or (collapse_cls != "No_Landslide"):
                danger_cls, danger_conf = classify_video(
                    danger_model, DANGER_CLASSES, tmp.name)
                if danger_cls is None:
                    st.error("Не удалось прочитать видео."); os.unlink(tmp.name); st.stop()
                st.success(f"**Опасность:** {danger_cls}  ({danger_conf*100:.1f} %)")
            else:
                st.success("Обвал не обнаружен — опасность не оценивается.")

        os.unlink(tmp.name)
        st.button("🔄 Новый файл", on_click=do_rerun)
else:
    st.caption("⬆️  Выберите видео для начала.")
