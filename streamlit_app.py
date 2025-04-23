
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2, tempfile, os

# ────────── настройки ──────────
IMG_SIZE, STEP = 150, 10         # ресайз кадра и «каждый N-й кадр»
COLLAPSE_CLASSES = ["No_Landslide", "Rockfall", "Earth_Flow"]
DANGER_CLASSES   = ["Safe", "Roads_Damaged", "Houses_Damaged"]

# ────────── кэш-загрузка моделей ──────────
@st.cache_resource(show_spinner=False)
def load_models():
    collapse, danger = None, None
    if os.path.exists("collapse_model.h5"):
        collapse = tf.keras.models.load_model("collapse_model.h5")
    if not os.path.exists("danger_model.h5"):
        st.error("⚠️  Файл danger_model.h5 не найден в каталоге приложения.")
        st.stop()
    danger = tf.keras.models.load_model("danger_model.h5")
    return collapse, danger
collapse_model, danger_model = load_models()

# ────────── функция классификации видео ──────────
def classify_video(model, classes, path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None, 0.0
    votes, frame_id = np.zeros(len(classes), int), 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = max(total // 100, 1)               # progress в 100 шагов
    prog = st.progress(0)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_id % STEP == 0:
            fr = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            p = model.predict(np.expand_dims(fr, 0)/255., verbose=0)[0]
            votes[np.argmax(p)] += 1
        if frame_id % every == 0:
            prog.progress(min(frame_id / max(total,1), 1.0))
        frame_id += 1
    cap.release(); prog.empty()
    if votes.sum()==0: return None, 0.0
    idx = votes.argmax()
    return classes[idx], votes[idx]/votes.sum()

# ────────── UI ──────────
st.set_page_config(page_title="Landslide Detection", page_icon="🌋", layout="centered")
st.title("🌋 Landslide Detection")

st.markdown("""
Загрузите видео *.mp4 / *.avi* и нажмите **Классифицировать**.  
*Если рядом нет `collapse_model.h5`, приложение выполнит только оценку опасности
(`Safe / Roads_Damaged / Houses_Damaged`).*
""")

video_file = st.file_uploader("Выберите видеофайл", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read()); tmp.close()
    st.success(f"✔ Загружено: {video_file.name}")

    if st.button("Классифицировать"):
        with st.spinner("🔍 Анализируем видео…"):
            # шаг 1: тип обвала (если модель есть)
            collapse_cls, collapse_conf = None, None
            if collapse_model is not None:
                collapse_cls, collapse_conf = classify_video(
                    collapse_model, COLLAPSE_CLASSES, tmp.name)
                if collapse_cls is None:
                    st.error("Не удалось прочитать видео."); os.unlink(tmp.name); st.stop()
                st.info(f"**Тип обвала:** {collapse_cls}  ({collapse_conf*100:.1f} %)")

            # решение: нужно ли оценивать опасность?
            if (collapse_model is None) or (collapse_cls != "No_Landslide"):
                danger_cls, danger_conf = classify_video(
                    danger_model, DANGER_CLASSES, tmp.name)
                if danger_cls is None:
                    st.error("Не удалось прочитать видео."); os.unlink(tmp.name); st.stop()
                st.success(f"**Опасность:** {danger_cls}  ({danger_conf*100:.1f} %)")
            else:
                st.success("Опасность не оценивается, т.к. обвал не обнаружен.")

        os.unlink(tmp.name)        # чистим временный файл
        st.button("🔄 Сброс", on_click=lambda: st.experimental_rerun())
else:
    st.caption("⬆️  Видео ещё не выбрано")
