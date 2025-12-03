#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN para Detecção de Pneumonia em Raios-X
------------------------------------------------------------
- Baseado em abordagens clássicas de CNN para o dataset Chest X-Ray (Pneumonia)
- Foco em código simples, legível e com comentários linha a linha
- Usa Keras (TensorFlow) + ImageDataGenerator para carregar e aumentar os dados
- Métricas: Acurácia, AUC, Precisão, Revocação e F1 (via sklearn)
- Salva: melhor modelo (.keras), gráficos de treino (.png) e relatório (.txt)

Estrutura de pastas esperada (como no Kaggle):
chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/

Uso:
    python cnn_pneumonia.py --data_dir "caminho/para/chest_xray" --epochs 20 --batch_size 32 --img_size 224
"""

import os
import argparse
import random
import numpy as np

# Força o TensorFlow a ser importado somente quando necessário
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# -------------------------
# 1) Controle de aleatoriedade
# -------------------------
def fixar_semente(seed: int = 42) -> None:
    """Garante reprodutibilidade simples (tanto quanto possível)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------
# 2) Construtor do modelo CNN
# -------------------------
def construir_modelo(img_size: int = 224) -> tf.keras.Model:
    """
    Cria uma CNN simples e eficiente para classificação binária.
    - Blocos Conv2D + BatchNorm + ReLU + MaxPool
    - GlobalAveragePooling para reduzir parâmetros
    - Camada densa final com Dropout
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # Bloco 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloco 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bloco 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Pooling global para reduzir para 1 vetor por imagem
    x = layers.GlobalAveragePooling2D()(x)

    # Densa com Dropout para generalização
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Saída binária (sigmoid)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="CNN_Pneumonia")
    return model


# -------------------------
# 3) Geradores de dados (Data Augmentation)
# -------------------------
def criar_generators(data_dir: str, img_size: int, batch_size: int):
    """
    Cria ImageDataGenerators para treino/validação/teste.
    - Treino usa augmentation leve para ajudar generalização.
    - Val/Test apenas reescala para 0..1.
    """
    # Caminhos esperados
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Verificações básicas de estrutura de pastas
    for p in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"Pasta não encontrada: {p}\n"
                "A estrutura esperada é data_dir/train, data_dir/val e data_dir/test."
            )

    # Gerador de treino com aumento de dados (rotação/zoom/shift/flip)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # Validação e teste apenas normalizam
    test_val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # flow_from_directory espera subpastas (NORMAL, PNEUMONIA)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    test_gen = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# -------------------------
# 4) Treinamento do modelo
# -------------------------
def treinar(model: tf.keras.Model,
            train_gen,
            val_gen,
            epochs: int,
            out_dir: str):
    """
    Compila, configura callbacks e treina o modelo.
    - Usa Adam com LR padrão 1e-3
    - EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
    """
    # Compilação com métricas úteis para binário
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    # Calcula pesos de classe para lidar com desbalanceamento
    y_indices = train_gen.classes  # rótulos inteiros do gerador
    classes = np.unique(y_indices)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_indices
    )
    class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}

    # Callbacks para melhorar generalização e evitar overfitting
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best_model.keras")
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=6, mode="max", restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=3, mode="max", min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1
        ),
    ]

    # Ajuste do número de passos por época
    steps_per_epoch = len(train_gen)
    val_steps = len(val_gen)

    # Treinamento
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=cbs,
        verbose=1,
    )
    return history, ckpt_path


# -------------------------
# 5) Avaliação no conjunto de teste
# -------------------------
def avaliar(model: tf.keras.Model, test_gen, out_dir: str) -> None:
    """
    Avalia o modelo no conjunto de teste e gera relatório.
    - Salva classification_report e confusion_matrix em TXT.
    """
    # Avaliação direta (perda + métricas compiladas)
    results = model.evaluate(test_gen, verbose=0)
    metric_names = model.metrics_names
    resultados_txt = os.path.join(out_dir, "test_metrics.txt")
    with open(resultados_txt, "w", encoding="utf-8") as f:
        for name, value in zip(metric_names, results):
            f.write(f"{name}: {value:.4f}\n")

    # Predições para relatório detalhado
    y_true = test_gen.classes
    y_prob = model.predict(test_gen, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Relatório de classificação e F1 (macro)
    report = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"])
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    # Salva em TXT
    relatorio_txt = os.path.join(out_dir, "classification_report.txt")
    with open(relatorio_txt, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nF1 (macro): {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print("=== Resultados de Teste ===")
    print(report)
    print(f"F1 (macro): {f1:.4f}")
    print("Matriz de confusão:\n", cm)


# -------------------------
# 6) Gráficos de perda e métricas
# -------------------------
def plotar_historia(history: tf.keras.callbacks.History, out_dir: str) -> None:
    """Gera gráficos de treino/validação e salva como PNG."""
    hist = history.history

    def salvar_plot(chave: str, titulo: str, ylabel: str):
        if chave not in hist or ("val_" + chave) not in hist:
            return
        plt.figure()
        plt.plot(hist[chave], label=chave)
        plt.plot(hist["val_" + chave], label="val_" + chave)
        plt.title(titulo)
        plt.xlabel("Época")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        caminho = os.path.join(out_dir, f"{chave}.png")
        plt.savefig(caminho, dpi=150)
        plt.close()

    os.makedirs(out_dir, exist_ok=True)
    salvar_plot("loss", "Perda (Train vs Val)", "Perda")
    salvar_plot("accuracy", "Acurácia (Train vs Val)", "Acurácia")
    salvar_plot("auc", "AUC (Train vs Val)", "AUC")
    salvar_plot("precision", "Precisão (Train vs Val)", "Precisão")
    salvar_plot("recall", "Revocação (Train vs Val)", "Revocação")


# -------------------------
# 7) Função principal / CLI
# -------------------------
def main():
    # Parser dos argumentos da linha de comando
    parser = argparse.ArgumentParser(description="CNN para Detecção de Pneumonia")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Pasta raiz do dataset (com subpastas train/val/test).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Número de épocas para treinar (padrão: 20).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamanho do batch (padrão: 32).")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Tamanho do lado da imagem quadrada (padrão: 224).")
    parser.add_argument("--out_dir", type=str, default="resultados_cnn_pneumonia",
                        help="Pasta de saída para salvar modelo e gráficos.")
    args = parser.parse_args()

    # Semente para reprodutibilidade
    fixar_semente(42)

    # Cria geradores de dados
    train_gen, val_gen, test_gen = criar_generators(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    # Constrói o modelo
    model = construir_modelo(img_size=args.img_size)

    # Treina o modelo
    history, ckpt_path = treinar(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=args.epochs,
        out_dir=args.out_dir
    )

    # Salva gráficos do histórico
    plotar_historia(history, args.out_dir)

    # Recarrega o melhor checkpoint (garantia extra)
    if os.path.isfile(ckpt_path):
        model = tf.keras.models.load_model(ckpt_path)

    # Avalia no conjunto de teste e salva relatórios
    avaliar(model, test_gen, args.out_dir)

    print("\nArquivos gerados em:", os.path.abspath(args.out_dir))
    print("- best_model.keras (modelo)")
    print("- *.png (gráficos de treino)")
    print("- test_metrics.txt, classification_report.txt")

if __name__ == "__main__":
    main()
