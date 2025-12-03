# salvar como: webcam_emocoes.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo treinado
modelo = load_model("melhor_modelo_fer.keras")

mapa_emocoes = {
    0: "Raiva",
    1: "Nojo",
    2: "Medo",
    3: "Feliz",
    4: "Triste",
    5: "Surpreso",
    6: "Neutro"
}

# Classificador de faces do OpenCV (Haar Cascade)
detector_rosto = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Abrir webcam (0 = câmera padrão)
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Converter para cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos
    rostos = detector_rosto.detectMultiScale(
        frame_cinza,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in rostos:
        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # Recortar o rosto
        roi_cinza = frame_cinza[y:y+h, x:x+w]

        # Redimensionar para 48x48 (mesmo do treino)
        roi_redimensionado = cv2.resize(roi_cinza, (48, 48))
        roi_normalizado = roi_redimensionado / 255.0
        roi_normalizado = roi_normalizado.reshape(1, 48, 48, 1)

        # Prever emoção
        probabilidades = modelo.predict(roi_normalizado, verbose=0)[0]
        indice_emocao = np.argmax(probabilidades)
        nome_emocao = mapa_emocoes[indice_emocao]

        # Escrever emoção no frame
        cv2.putText(
            frame,
            nome_emocao,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    cv2.imshow("Reconhecimento de Emocoes - Webcam", frame)

    # Tecla 'Esc' para sair
    if cv2.waitKey(1) & 0xFF == 27: # 27: Valor ASCII para tecla ESC
        break

camera.release()
cv2.destroyAllWindows()
