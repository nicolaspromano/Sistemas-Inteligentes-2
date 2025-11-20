import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam

# --- 1. Carregar e Normalizar o CIFAR-10 ---
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# --- 2. Adicionar Ruído Gaussiano ---
#para que o algoritmo não só "decore" padrões
noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# --- 3. Definir a Arquitetura do dAE ---

input_img = Input(shape=(32, 32, 3))

# --- Encoder ---
# 32x32x3 -> 16x16x32
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
# 16x16x32 -> 8x8x64
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# 8x8x64 -> 8x8x128
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x) # 4x4x128

# --- Decoder ---
# 4x4x128 -> 8x8x128
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
# 8x8x128 -> 16x16x64
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
# 16x16x64 -> 32x32x32
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
# 32x32x32 -> 32x32x3
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# criar o modelo do autoencoder
autoencoder = Model(input_img, decoded)

# mean_squared_error para reconstruir pixels
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

autoencoder.summary()

# --- 4. Treinar o dAE ---
# Treinamos usando todo o conjunto de treinamento 
print("Iniciando o treinamento do dAE...")
history = autoencoder.fit(x_train_noisy, x_train,
                          epochs=12,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test_noisy, x_test))

print("Treinamento concluído.")


# --- 5. Verificar a Reconstrução ---
# Usar o dAE treinado para "limpar" as imagens de teste
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10  # número de imagens
plt.figure(figsize=(20, 6))
for i in range(n):
    # Imagem Ruidosa
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i])
    plt.title("Ruidosa")
    plt.axis("off")

    # Imagem Reconstruída
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstruída")
    plt.axis("off")

    # Imagem Original
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.axis("off")
plt.show()

from keras.layers import Flatten, Dense
from keras.utils import to_categorical

# --- 6. Preparação para Classificação ---

num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Selecionar 500 imagens aleatórias para o treinamento 
np.random.seed(42) 
indices_aleatorios = np.random.choice(len(x_train), 500, replace=False)

x_train_sub = x_train[indices_aleatorios]
y_train_sub = y_train_cat[indices_aleatorios]

print(f"Subconjunto de treino criado: {x_train_sub.shape}")


# --- 7. Experimento 1: Classificador COM Pré-treinamento (Usando o dAE) ---

# camadas de classificação
flat = Flatten()(encoded)
out = Dense(num_classes, activation='softmax')(flat)

clf_pretrained = Model(input_img, out)

# Congelar as camadas do Encoder para usar o conhecimento que o dAE já adquiriu, não treinar tudo de novo.
for layer in clf_pretrained.layers[:7]:
    layer.trainable = False

clf_pretrained.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTreinando Classificador COM Pré-treinamento...")
hist_pretrained = clf_pretrained.fit(x_train_sub, y_train_sub,
                                     epochs=12, 
                                     batch_size=32, 
                                     validation_data=(x_test, y_test_cat),
                                     verbose=0) # verbose=0 para não encher a tela
print(f"Acurácia Final (Pré-treinado): {hist_pretrained.history['val_accuracy'][-1]:.4f}")


# --- 8. Experimento 2: Classificador SEM Pré-treinamento (Do Zero) ---
# pesos iniciados aleatoriamente

input_scratch = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_scratch)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x) 

x = Flatten()(x)
out_scratch = Dense(num_classes, activation='softmax')(x)

clf_scratch = Model(input_scratch, out_scratch)
clf_scratch.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTreinando Classificador SEM Pré-treinamento (Do Zero)...")
hist_scratch = clf_scratch.fit(x_train_sub, y_train_sub,
                               epochs=12, 
                               batch_size=32, 
                               validation_data=(x_test, y_test_cat),
                               verbose=0)
print(f"Acurácia Final (Do Zero):      {hist_scratch.history['val_accuracy'][-1]:.4f}")

# --- 9. Gráfico ---
plt.figure(figsize=(10, 5))
plt.plot(hist_pretrained.history['val_accuracy'], label='Pré-treinado')
plt.plot(hist_scratch.history['val_accuracy'], label='Do Zero')
plt.title('Comparação de Acurácia no Teste')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()
