
"""
CNN para Reconhecimento de Emoções (FER2013)
------------------------------------------------------------
- Metodologia: Imagens 48x48, Grayscale, 7 Classes 
- Estrutura esperada:
  dataset/
    ├── train/ (angry, disgust, fear, happy, neutral, sad, surprise)
    └── test/  (angry, disgust, fear, happy, neutral, sad, surprise)

- Como compilar
  "python CNN_emocoes.py --data_dir "caminho/para/FE-2013"

  """

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# 1) Configurações
# -------------------------
NUM_CLASSES = 7
EMOCOES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def fixar_semente(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -------------------------
# 2) Modelo (Arquitetura)
# -------------------------
def construir_modelo(img_size: int = 48):
    """
    CNN adaptada para imagens pequenas (48x48) e complexas (faces).
    Baseado na hierarquia de aprendizado: Bordas -> Expressões -> Emoção
    """
    # Entrada: 48x48, 1 canal (escala de cinza)
    inputs = layers.Input(shape=(img_size, img_size, 1))

    # Bloco 1 - Extração de bordas e formas básicas
    x = layers.Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Bloco 2 - Padrões intermediários
    x = layers.Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Bloco 3 - Expressões mais complexas
    x = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Bloco 4 - Aprofundando (opcional, mas ajuda em faces)
    x = layers.Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Classificação
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Saída: Softmax para 7 classes (probabilidade para cada emoção) 
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="CNN_Emocoes_FER2013")
    return model

# -------------------------
# 3) Geradores de Dados
# -------------------------
def criar_generators(data_dir, img_size, batch_size):
    """
    Carrega imagens em Grayscale e aplica Data Augmentation no treino.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Data Augmentation: Rotações e zoom conforme metodologia [cite: 59]
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0, # Normalização [cite: 58]
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Validação/Teste: Apenas normaliza
    test_val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    print(f"Carregando Treino de: {train_dir}")
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",  # IMPORTANTE: Tons de cinza
        batch_size=batch_size,
        class_mode="categorical", # IMPORTANTE: Multiclasse
        shuffle=True
    )

    print(f"Carregando Teste/Validação de: {test_dir}")
    val_gen = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen

# -------------------------
# 4) Treinamento
# -------------------------
def treinar(model, train_gen, val_gen, epochs, out_dir):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy", # Loss para multiclasse
        metrics=["accuracy"]
    )

    os.makedirs(out_dir, exist_ok=True)
    
    # Callbacks
    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "melhor_modelo_fer.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=cbs
    )
    return history

# -------------------------
# 5) Avaliação e Gráficos
# -------------------------
def avaliar(model, test_gen, out_dir):
    print("\n--- Iniciando Avaliação Final ---")
    y_true = test_gen.classes
    # Predição
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1) # Pega o índice da maior probabilidade

    # Relatório
    report = classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys()))
    cm = confusion_matrix(y_true, y_pred)
    
    print(report)
    
    with open(os.path.join(out_dir, "relatorio_final.txt"), "w") as f:
        f.write(report)
        f.write("\n\nMatriz de Confusão:\n")
        f.write(str(cm))

def plotar_historia(history, out_dir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Gráfico Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Treino Acurácia')
    plt.plot(epochs_range, val_acc, label='Validação Acurácia')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treino e Validação')

    # Gráfico Perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Treino Loss')
    plt.plot(epochs_range, val_loss, label='Validação Loss')
    plt.legend(loc='upper right')
    plt.title('Perda (Loss) de Treino e Validação')
    
    plt.savefig(os.path.join(out_dir, "graficos_treinamento.png"))
    plt.show()

# -------------------------
# Principal (Main)
# -------------------------
if __name__ == "__main__":
    # Configuração da leitura de argumentos via terminal
    parser = argparse.ArgumentParser(description="Treinamento da CNN para Emoções (FER2013)")
    
    # Argumento caminho da pasta
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Caminho raiz do dataset (onde estão as pastas train e test)")
    
    # Argumentos OPCIONAIS (com valores padrão se não informar)
    parser.add_argument("--epochs", type=int, default=12,
                        help="Número de épocas de treinamento (padrão: 12)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Tamanho do lote de imagens (padrão: 64)")
    parser.add_argument("--out_dir", type=str, default="resultados_emocoes",
                        help="Pasta onde salvar o modelo e gráficos (padrão: resultados_emocoes)")

    # Processa os argumentos
    args = parser.parse_args()

    # Configurações fixas do FER2013
    IMG_SIZE = 48
    
    # Garante reprodutibilidade
    fixar_semente()

    print(f"--- Iniciando treinamento ---")
    print(f"Dataset: {args.data_dir}")
    print(f"Épocas: {args.epochs} | Batch: {args.batch_size}")

    # 1. Carregar Dados (Usando o caminho que veio do terminal: args.data_dir)
    train_gen, val_gen = criar_generators(args.data_dir, IMG_SIZE, args.batch_size)

    # 2. Criar Modelo
    model = construir_modelo(IMG_SIZE)
    # model.summary() # Descomente se quiser ver a estrutura no terminal

    # 3. Treinar (Usando as épocas e pasta de saída do terminal)
    history = treinar(model, train_gen, val_gen, args.epochs, args.out_dir)

    # 4. Avaliar e Salvar
    plotar_historia(history, args.out_dir)
    avaliar(model, val_gen, args.out_dir)
    
    print(f"\nConcluído! Resultados salvos em: {args.out_dir}")