import os
import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, Input, SpatialDropout1D)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import regularizers



PATH = '/Users/hamza/Desktop/ЭКГ/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/Needit/'
SAMPLING_RATE = 100
SIGNAL_LENGTH = 1000
BATCH_SIZE = 32
EPOCHS = 50
BALANCE_MULTIPLIER = 1.3
MAX_BALANCED_SIZE = 4000
AUGMENT_FRACTION = 0.4
MAX_TIME_SHIFT = 20
AUGMENT_NOISE_STD = 0.01



# 1) Загрузка данных
Y=pd.read_csv(PATH + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x)) # ast.literal_eval превращает этот текст "{'NORM': 100}" в настоящий словарь Python
agg_df = pd.read_csv(PATH + 'scp_statements.csv', index_col= 0 )
agg_df = agg_df[agg_df.diagnostic_class.notnull()] # сгруппировать их в 5 больших групп






selected_classes = ['NORM', 'MI', 'STTC']  # Норма, Инфаркт, Изменения ST-T

def aggregate_diagnostic(y_dic):
  #Вместо кучи сложных кодов мы получаем простой список: ['MI']
  tmp=[]
  for key in y_dic.keys():
    if key in agg_df.index:
      diagnostic_class = agg_df.loc[key].diagnostic_class
      if diagnostic_class in selected_classes:
                tmp.append(diagnostic_class)
  return list(set(tmp))
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic) #Применяем функцию ко всем пациентам.
Y['counts'] = Y.diagnostic_superclass.apply(len)
df_clean = Y[Y.counts == 1].copy() # Filter - Берем только записи с ОДНИМ диагнозом
df_clean['label'] = df_clean.diagnostic_superclass.apply(lambda x: x[0]) # Вытаскиваем значение из списка. Было ['MI'] (список), стало 'MI' (строка)

# Превращение букв в цифры
le = LabelEncoder()
y_indices = le.fit_transform(df_clean['label'])
classes = le.classes_


print(" МАППИНГ КЛАССОВ (MAPPING)")
for i, class_name in enumerate(classes):
    print(f"  {class_name} <---> {i}")

# Чтение физических файлов
def load_raw_data(df , sampling_rate, path) :
  if sampling_rate == 100 :
    data = [wfdb.rdsamp(path + f)[0] for f in df.filename_lr]
  else :
    data = [wfdb.rdsamp(path + f)[0] for f in df.filename_hr]
  return np.array(data)
X = load_raw_data(df_clean, SAMPLING_RATE, PATH)
print(" СТАТИСТИКА ДАННЫХ")
print(f"Всего записей в базе (Y): {len(Y)}")
print(f"После чистки (только 1 диагноз): {len(df_clean)}")
print(f"Мы берем (sample count): {X.shape[0]} пациентов")
print(f"Длина сигнала: {X.shape[1]} точек (10 секунд)")
print(f"Количество каналов: {X.shape[2]} (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)")
print(f"Размер данных в памяти: {X.nbytes / (1024**3):.2f} GB")
print("="*30 + "\n")

# Визуализация сигналов
def plot_single_ecg_signal(signal, label, sample_rate=100, save_path=None):
    time = np.arange(signal.shape[0]) / sample_rate
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle(f'12-канальная ЭКГ - Диагноз: {label}', fontsize=16, fontweight='bold')
    for ax, lead, channel in zip(axes.flat, leads, signal.T):
        ax.plot(time, channel, linewidth=1.0, color='blue')
        ax.set_title(f'Отведение {lead}', fontweight='bold')
        ax.set_xlabel('Время (сек)')
        ax.set_ylabel('Амплитуда')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_random_sample():
    idx = np.random.randint(0, len(X))
    signal = X[idx]
    label_text = df_clean.iloc[idx]['label']
    plot_single_ecg_signal(signal, label_text, sample_rate=SAMPLING_RATE,
                           save_path=os.path.join(PATH, 'single_ecg_detail.png'))


print("Рисуем случайный сигнал...")
plot_random_sample()


def augment_signal(signal):
    augmented = np.copy(signal)
    if MAX_TIME_SHIFT > 0:
        shift = np.random.randint(-MAX_TIME_SHIFT, MAX_TIME_SHIFT + 1)
        augmented = np.roll(augmented, shift, axis=0)
    scale = np.random.uniform(0.9, 1.1)
    augmented *= scale
    noise = np.random.normal(0, AUGMENT_NOISE_STD, augmented.shape)
    augmented += noise
    return augmented.astype(np.float32)


def augment_batch(signals):
    return np.stack([augment_signal(sig) for sig in signals])


# 2) Разделение данных 
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y_indices, test_size=0.15, random_state=42, stratify=y_indices)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train_raw, y_train_raw, test_size=0.1, random_state=42, stratify=y_train_raw)
print("РАЗДЕЛЕНИЕ ДАННЫХ:")
print(f"Train (raw): {X_train_raw.shape} | Размер: {X_train_raw.nbytes / (1024**3):.2f} GB")
print(f"Validation (raw): {X_val_raw.shape} | Размер: {X_val_raw.nbytes / (1024**3):.2f} GB")
print(f"Test: {X_test.shape} | Размер: {X_test.nbytes / (1024**3):.2f} GB")
print("="*30 + "\n")

# 3) Балансировка классов (Upsampling)
def plot_pie_distribution(y_data, title):
    unique, counts = np.unique(y_data, return_counts=True)
    labels = [classes[i] for i in unique]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    plt.title(title)

#  График ДО балансировки
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_pie_distribution(y_train_raw, "Распределение ДО балансировки (Train)")


# Процесс балансировка
# Создаем DataFrame для удобства обработки индексов
train_df = pd.DataFrame({'label_idx': y_train_raw})
train_df['original_idx'] = range(len(train_df))

# Показываем распределение классов ДО балансировки
print("РАСПРЕДЕЛЕНИЕ КЛАССОВ ДО БАЛАНСИРОВКИ:")
class_counts_before = train_df['label_idx'].value_counts().sort_index()
for class_idx, count in class_counts_before.items():
    print(f"  Класс {class_idx} ({classes[class_idx]}): {count} образцов")
print(f"Минимальный класс: {class_counts_before.min()}")
print(f"Максимальный класс: {class_counts_before.max()}")
print(f"Средний размер класса: {class_counts_before.mean():.0f}")

# Адаптивный таргет для балансировки
class_counts = train_df['label_idx'].value_counts()
min_count = class_counts.min()
target_size = min(int(min_count * BALANCE_MULTIPLIER), MAX_BALANCED_SIZE)
target_size = max(target_size, min_count)
print(f"\nАдаптивная балансировка: min={min_count}, target={target_size}")

balanced_indices = []
for class_idx in range(len(classes)):
    class_data = train_df[train_df['label_idx'] == class_idx]
    current_size = len(class_data)
    replace = current_size < target_size
    data_resampled = resample(
        class_data,
        replace=replace,
        n_samples=target_size,
        random_state=42
    )
    balanced_indices.extend(data_resampled['original_idx'].values)

X_train_balanced = X_train_raw[balanced_indices]
y_train_balanced = y_train_raw[balanced_indices]

# Показываем распределение ПОСЛЕ балансировки
print("\nРАСПРЕДЕЛЕНИЕ КЛАССОВ ПОСЛЕ БАЛАНСИРОВКИ:")
class_counts_after = pd.Series(y_train_balanced).value_counts().sort_index()
for class_idx, count in class_counts_after.items():
    print(f"  Класс {class_idx} ({classes[class_idx]}): {count} образцов")

# Перемешиваем данные
shuffle_idx = np.random.permutation(len(X_train_balanced))
X_train = X_train_balanced[shuffle_idx]
y_train = y_train_balanced[shuffle_idx]

if AUGMENT_FRACTION > 0:
    augment_size = int(len(X_train) * AUGMENT_FRACTION)
    if augment_size > 0:
        sample_indices = np.random.choice(len(X_train), size=augment_size, replace=False)
        X_aug = augment_batch(X_train[sample_indices])
        y_aug = y_train[sample_indices]
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        print(f"🔁 Добавлено аугментированных сигналов: {augment_size}. Новый train размер: {len(X_train)}")

#  График ПОСЛЕ балансировки
plt.subplot(1, 2, 2)
plot_pie_distribution(y_train, "Распределение ПОСЛЕ балансировки (Train)")
plt.savefig('class_distribution_pie.png')
plt.show()

print(f"\n📊 РАЗМЕРЫ ДАННЫХ ПОСЛЕ БАЛАНСИРОВКИ:")
print(f"Train после балансировки: {X_train.shape}")
print(f"Размер в памяти: {X_train.nbytes / (1024**3):.2f} GB")
print(f"Общее количество образцов: {len(X_train)}")
print(f"Увеличение размера: {len(X_train) / len(X_train_raw):.2f}x")
print("="*30 + "\n")

# One-Hot Encoding
y_train_cat = to_categorical(y_train, num_classes=len(classes)) #Было: [0, 1, 2] (где 0=норма, 1=инфаркт, 2=ST-T) Стало: [[1,0,0], [0,1,0], [0,0,1]]
y_val_cat = to_categorical(y_val_raw, num_classes=len(classes))
y_test_cat = to_categorical(y_test, num_classes=len(classes))
X_val = X_val_raw.copy()

# Нормализация (z-score normalization)
#Z-score нормализация: (значение - среднее) / стандартное отклонение
scaler=StandardScaler()
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
scaler.fit(X_train_flat) # fit только на Train:  ОБУЧАЕМСЯ только на тренировочных данных!
X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val = scaler.transform(X_val_flat).reshape(X_val.shape)
X_test = scaler.transform(X_test_flat).reshape(X_test.shape)

# Build the model
def build_model (input_shape, num_classes):
    reg = regularizers.l2(1e-4)
    inputs = Input(shape=input_shape)

    x = Conv1D(32, 7, padding='same', use_bias=False, kernel_regularizer=reg)(inputs) # анализирует участки по 7 отсчетов (70 мс)
    x = BatchNormalization()(x) # # Нормализует данные между слоями
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x) ## Уменьшает размер в 2 раза: 1000 → 500 [500, 32]
    x = SpatialDropout1D(0.1)(x)

    x = Conv1D(64, 5, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x) # [500×64]
    x = Dropout(0.2)(x)

    x = Conv1D(128, 5, padding='same', use_bias=False, kernel_regularizer=reg)(x) # ищет очень сложные паттерны (QRS комплексы, ST сегменты)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x) #  [500×128]
    x = Dropout(0.3)(x)

    x = Conv1D(192, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x) # # [250×192] → [192] Уменьшает количество параметров

    x = Dense(96, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)
    x = Dense(48, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x) # Сильная регуляризация - выключаем 40% нейронов

    outputs = Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model((SIGNAL_LENGTH, 12), len(classes))

print("\n" + "="*50)
print("🏗️  АРХИТЕКТУРА МОДЕЛИ (MODEL ARCHITECTURE)")
print("="*50)
model.summary()
print("="*50 + "\n")

model.compile(optimizer=Adam(learning_rate=5e-4),
              loss='categorical_crossentropy', #Функция потерь для многоклассовой классификации
              metrics=['accuracy'])

print("✅ Модель скомпилирована успешно!")
print(f"Оптимизатор: Adam (learning_rate=0.0005)")
print(f"Функция потерь: categorical_crossentropy")
print(f"Метрики: accuracy")
print("="*50 + "\n")

# ==========================================
# ШАГ 5: ОБУЧЕНИЕ (БЕЗ CLASS_WEIGHT)
# ==========================================
# Создаем callbacks для обучения
class ValidationF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, class_names):
        super().__init__()
        self.validation_data = validation_data # (X_val, y_val_cat)
        self.class_names = class_names # ['NORM', 'MI', 'STTC']

    def on_epoch_end(self, epoch, logs=None): #Вызов после каждой эпохи
        X_val, y_val = self.validation_data
        preds = self.model.predict(X_val, verbose=0) # # Получаем вероятности
        y_pred = np.argmax(preds, axis=1)
        y_true = np.argmax(y_val, axis=1)
        _, _, f1_scores, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names)))
        scores = {f'f1_{cls}': float(score) for cls, score in zip(self.class_names, f1_scores)}
        print(f"\nValidation F1 per class: {scores}")


callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(PATH, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        verbose=1,
        mode='min',
        min_lr=1e-6
    ),
    CSVLogger('training_log.csv', append=False),
    ValidationF1Callback(validation_data=(X_val, y_val_cat), class_names=classes)
]

# ВАЖНО: class_weight убрали, так как данные теперь физически сбалансированы
history = model.fit(
    X_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks,
    verbose=1
)

# ==========================================
# ШАГ 6: ОЦЕНКА
# ==========================================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=classes))

# Графики обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.show()

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (After Balancing)')
plt.show()

