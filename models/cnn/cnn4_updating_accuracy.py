import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import json
import matplotlib.pyplot as plt

# Генератор данных
class NPYDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Архитектура модели
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.001)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def main(args):
    print("\U0001F4E6 Загрузка данных...")
    X = np.load(args.x_path).astype('float32')
    y = np.load(args.y_path)
    label_classes = np.load(args.classes_path)

    print("✅ Формы массивов: X =", X.shape, ", y =", y.shape)

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Генераторы
    train_gen = NPYDataGenerator(X_train, y_train, batch_size=args.batch_size)
    val_gen = NPYDataGenerator(X_test, y_test, batch_size=args.batch_size)

    print("\U0001F9E0 Строим модель...")
    model = build_model(input_shape=X.shape[1:], num_classes=y.shape[1])
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("🚀 Обучение...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[early_stop]
    )

    print("📀 Сохраняем модель в", args.model_path)
    model.save(args.model_path)

    print("💾 Сохраняем историю обучения в training_history.json")
    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    print("\U0001F5C3 Строим графики...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('📈 Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('📉 Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png")
    print("✅ Графики сохранены в training_plots.png")

    print("✅ Готово! Точность на тесте:")
    loss, acc = model.evaluate(val_gen)
    print(f"🎯 Accuracy: {acc:.4f}")

    # Пример предсказания
    sample = X_test[0:1]
    pred = model.predict(sample)
    predicted_index = np.argmax(pred)
    print("🔍 Пример предсказания: ", label_classes[predicted_index])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение CNN для распознавания акцента")
    parser.add_argument("--x_path", type=str, default="X_melspec.npy", help="Путь к X (мелспектры)")
    parser.add_argument("--y_path", type=str, default="y_onehot.npy", help="Путь к y (one-hot метки)")
    parser.add_argument("--classes_path", type=str, default="label_classes.npy", help="Путь к файлу с названиями классов")
    parser.add_argument("--model_path", type=str, default="cnn_accent_model.h5", help="Куда сохранить модель")
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    main(args)

