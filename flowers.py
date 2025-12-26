!pip install -q tensorflow tensorflow-datasets

import os, json, numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

print("TF version:", tf.__version__)

# ---------- конфиг ----------
config = {
    "img_height": 180,
    "img_width": 180,
    "channels": 3,
    "batch_size": 32,
    "buffer_size": 2000,
    "num_epochs": 5,
    "learning_rate": 1e-3,
    "use_data_augmentation": True,
    "base_filters": 32,
    "dense_units": 256,
    "dropout_rate": 0.5,
    "random_seed": 42,
}
os.makedirs("models", exist_ok=True)
with open("models/config.json","w") as f:
    json.dump(config, f, indent=2)

# ---------- список имён для 102 классов ----------
pretty_class_names = [
  "pink primrose","hard-leaved pocket orchid","canterbury bells","sweet pea",
  "english marigold","tiger lily","moon orchid","bird of paradise","monkshood",
  "globe thistle","snapdragon","colt's foot","king protea","spear thistle",
  "yellow iris","globe-flower","purple coneflower","peruvian lily",
  "balloon flower","giant white arum lily","fire lily","pincushion flower",
  "fritillary","red ginger","grape hyacinth","corn poppy",
  "prince of wales feathers","stemless gentian","artichoke","sweet william",
  "carnation","garden phlox","love in the mist","mexican aster",
  "alpine sea holly","ruby-lipped cattleya","cape flower","great masterwort",
  "siam tulip","lenten rose","barbeton daisy","daffodil","sword lily",
  "poinsettia","bolero deep blue","wallflower","marigold","buttercup",
  "oxeye daisy","common dandelion","petunia","wild pansy","primula",
  "sunflower","pelargonium","bishop of llandaff","gaura","geranium",
  "orange dahlia","pink-yellow dahlia","cautleya spicata","japanese anemone",
  "black-eyed susan","silverbush","californian poppy","osteospermum",
  "spring crocus","bearded iris","windflower","tree poppy","gazania","azalea",
  "water lily","rose","thorn apple","morning glory","passion flower","lotus",
  "toad lily","anthurium","frangipani","clematis","hibiscus","columbine",
  "desert-rose","tree mallow","magnolia","cyclamen","watercress","canna lily",
  "hippeastrum","bee balm","ball moss","foxglove","bougainvillea","camellia",
  "mallow","mexican petunia","bromelia","blanket flower","trumpet creeper",
  "blackberry lily"
]
assert len(pretty_class_names) == 102

# ---------- загрузка oxford_flowers102 ----------
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    "oxford_flowers102",
    split=["train","validation","test"],
    as_supervised=True,
    with_info=True,
)  # [web:28][web:27]

num_classes = ds_info.features["label"].num_classes
print("num_classes:", num_classes)
assert num_classes == len(pretty_class_names)

def preprocess(image, label):
    image = tf.image.resize(image, [config["img_height"], config["img_width"]])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_val   = ds_val.map(preprocess,   num_parallel_calls=tf.data.AUTOTUNE)
ds_test  = ds_test.map(preprocess,  num_parallel_calls=tf.data.AUTOTUNE)

if config["use_data_augmentation"]:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )
else:
    data_augmentation = tf.keras.Sequential()

AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.shuffle(config["buffer_size"], seed=config["random_seed"]).batch(
    config["batch_size"]
).prefetch(AUTOTUNE)
ds_val  = ds_val.batch(config["batch_size"]).prefetch(AUTOTUNE)
ds_test = ds_test.batch(config["batch_size"]).prefetch(AUTOTUNE)

# ---------- модель ----------
def build_cnn_model(cfg, num_classes):
    inputs = tf.keras.Input(
        shape=(cfg["img_height"], cfg["img_width"], cfg["channels"])
    )
    x = data_augmentation(inputs)
    filters = cfg["base_filters"]
    for _ in range(4):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        filters *= 2
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(cfg["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg["dropout_rate"])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

model = build_cnn_model(config, num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=config["num_epochs"],
)

test_loss, test_acc = model.evaluate(ds_test)
print("Test acc:", test_acc)

# ---------- сохраняем модель и имена ----------
model.save("models/oxford102_final.keras")
with open("models/class_names.json","w") as f:
    json.dump(pretty_class_names, f, indent=2)

print("models:", os.listdir("models"))



%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "oxford102_final.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_meta():
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return class_names, config

model = load_model()
class_names, config = load_meta()

st.set_page_config(page_title="Oxford Flowers", layout="centered")
st.title("🌸 Классификатор цветов")
st.write("Распознаёт 102 вида цветов")

uploaded_file = st.file_uploader("Загрузите фото цветка", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    h, w = config["img_height"], config["img_width"]
    img = image.resize((w, h))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))

    flower_name = class_names[idx]

    st.subheader("Результат")
    st.write(f"🎕 **{flower_name}**")
    st.success("Модель уверена в результате!")

    k = min(5, len(class_names))
    top_idx = np.argsort(preds[0])[::-1][:k]
    st.markdown("### Другие варианты:")
    for i, top_i in enumerate(top_idx):
        if i == 0: continue
        name = class_names[int(top_i)]
        st.write(f"• {name}")
else:
    st.info("Загрузите изображение цветка.")





!pip install -q streamlit pyngrok


from pyngrok import ngrok

ngrok.set_auth_token("37LEgJu5H3eN7gd1sXgcZD5rnms_2dZvpMgr5BCthaYEMAEaV")




from pyngrok import ngrok
import subprocess, time

ngrok.kill()  # закрыть старые туннели

# запускаем streamlit в фоне
process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

time.sleep(8)  # даём приложению стартовать

public_url = ngrok.connect(8501)
print("Публичная ссылка на приложение:", public_url)


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import regularizers

# ---------- 1. Описание 5 алгоритмов ----------
algorithms = [
    # Алгоритм 1: базовая CNN + Adam
    {"name": "Алгоритм 1 (Adam, базовая CNN)",
     "type": "cnn", "optimizer": "adam", "lr": 1e-3,
     "base_filters": 32, "l2": None, "fine_tune": False},

    # Алгоритм 2: глубже + L2
    {"name": "Алгоритм 2 (Adam, L2)",
     "type": "cnn", "optimizer": "adam", "lr": 1e-3,
     "base_filters": 32, "l2": 1e-3, "fine_tune": False},

    # Алгоритм 3: та же CNN, но оптимизатор SGD
    {"name": "Алгоритм 3 (SGD)",
     "type": "cnn", "optimizer": "sgd", "lr": 1e-2,
     "base_filters": 32, "l2": None, "fine_tune": False},

    # Алгоритм 4: MobileNetV2 как feature extractor
    {"name": "Алгоритм 4 (MobileNetV2, frozen)",
     "type": "mobilenet", "optimizer": "adam", "lr": 1e-4,
     "fine_tune": False},

    # Алгоритм 5: MobileNetV2 fine‑tune (как на примере)
    {"name": "Алгоритм 5 (MobileNetV2 Fine‑tune)",
     "type": "mobilenet", "optimizer": "adam", "lr": 1e-4,
     "fine_tune": True},
]


# ---------- 2. Построение моделей ----------
def build_cnn(cfg, num_classes, base_filters=32, l2_reg=None):
    reg = regularizers.l2(l2_reg) if l2_reg is not None else None

    inputs = tf.keras.Input(
        shape=(cfg["img_height"], cfg["img_width"], cfg["channels"])
    )
    x = data_augmentation(inputs)

    filters = base_filters
    for _ in range(3):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
        )(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        filters *= 2

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(cfg["dense_units"], activation="relu")(x)
    x = tf.keras.layers.Dropout(cfg["dropout_rate"])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def build_mobilenet(cfg, num_classes, fine_tune=False):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(cfg["img_height"], cfg["img_width"], cfg["channels"]),
        include_top=False,
        weights="imagenet",
    )
    if fine_tune:
        # размораживаем верхние слои для дообучения
        fine_tune_at = len(base_model.layers) // 2
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(cfg["img_height"], cfg["img_width"], cfg["channels"])
    )
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# ---------- 3. Обучение всех алгоритмов и рисование графиков ----------
histories = {}
final_results = []  # для текста вывода

for i, alg in enumerate(algorithms, start=1):
    print(f"\n======================")
    print(f"{alg['name']}")
    print(f"======================")

    # модель
    if alg["type"] == "cnn":
        model = build_cnn(
            config,
            num_classes,
            base_filters=alg.get("base_filters", 32),
            l2_reg=alg.get("l2", None),
        )
    else:
        model = build_mobilenet(
            config,
            num_classes,
            fine_tune=alg.get("fine_tune", False),
        )

    # оптимизатор
    if alg["optimizer"] == "sgd":
        opt = tf.keras.optimizers.SGD(
            learning_rate=alg["lr"], momentum=0.9
        )
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=alg["lr"])

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=config["num_epochs"],
        verbose=1,
    )

    histories[alg["name"]] = history

    # итоговая метрика
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    final_results.append((alg["name"], train_acc, val_acc))

    # ---- ДВА ГРАФИКА ДЛЯ ЭТОГО АЛГОРИТМА ----
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Обучающая точность")
    plt.plot(epochs, val_acc, label="Валидационная точность")
    plt.title(f"Точность обучения и валидации\n({alg['name']})")
    plt.xlabel("Эпоха")
    plt.ylabel("Точность")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Обучающая функция потерь")
    plt.plot(epochs, val_loss, label="Валидационная функция потерь")
    plt.title(f"Потери обучения и валидации\n({alg['name']})")
    plt.xlabel("Эпоха")
    plt.ylabel("Функция потерь")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------- 4. Краткий вывод по всем алгоритмам ----------
print("\nИТОГИ СРАВНЕНИЯ АЛГОРИТМОВ:")
for name, train_acc, val_acc in final_results:
    print(f"{name}: обучающая точность = {train_acc:.3f}, "
          f"валидационная точность = {val_acc:.3f}")
