"""
Como rodar:
    python MLP_autoencoder.py --hidden 128 --epochs 20 --lr 0.05 --log-every 1
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Definição do modelo
# -------------------------
class Modelo:
    __slots__ = ("Wh", "Wo", "E", "H", "O")
    def __init__(self, Wh, Wo, sizes):
        self.Wh = Wh
        self.Wo = Wo
        self.E, self.H, self.O = sizes


def initialization(inputs, hidden, outputs, rng=None, wscale=0.5):
    if rng is None:
        rng = np.random.default_rng(1234)
    Wh = rng.uniform(-wscale, wscale, size=(hidden, inputs + 1))
    Wo = rng.uniform(-wscale, wscale, size=(outputs, hidden + 1))
    return Modelo(Wh, Wo, (inputs, hidden, outputs))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid_from_output(s):
    return s * (1.0 - s)


def forward(example, model: Modelo):
    inp = np.concatenate([np.array([1.0]), example])   
    net_hidden = model.Wh @ inp                         
    hidden_out = sigmoid(net_hidden)                    
    h_bias = np.concatenate([np.array([1.0]), hidden_out]) 
    net_out = model.Wo @ h_bias                        
    O_output = sigmoid(net_out)                        
    return O_output, net_hidden, hidden_out, net_out, inp, h_bias


# -------------------------
# MLPTrain agora treina AE (X -> X) sem mexer no nome
# -------------------------
def MLPTrain(X, Y_unused, model: Modelo, epochs=1500, lr=0.1, tol=1e-6, verbose=True, log_every=100):
    N = X.shape[0]
    losses = []

    for ep in range(epochs):
        epoch_loss = 0.0

        # Embaralha índices por época
        for i in np.random.permutation(N):
            x = X[i]

            # FORWARD
            x_hat, net_h, h_out, net_o, inp, h_bias = forward(x, model)

            # ERRO e loss (MSE/2)
            err = x - x_hat
            epoch_loss += 0.5 * np.sum(err**2)

            # BACKPROP
            dO = err * d_sigmoid_from_output(x_hat)         # (O,)
            Wo_no_bias = model.Wo[:, 1:]                    # (O, H)
            dh = (Wo_no_bias.T @ dO) * d_sigmoid_from_output(h_out)

            # ATUALIZAÇÃO
            model.Wo += lr * np.outer(dO, h_bias)
            model.Wh += lr * np.outer(dh, inp)

        epoch_loss /= N
        losses.append(epoch_loss)

        if verbose and (((ep + 1) % log_every == 0) or ep == 0 or ep == epochs - 1):
            print(f"Epoch {ep+1}/{epochs} - Loss: {epoch_loss:.6f}")

        if epoch_loss < tol:
            if verbose:
                print(f"Parada por tolerância na época {ep+1} (loss={epoch_loss:.6f})")
            break

    return np.array(losses)


def reconstruct(X, model: Modelo):
    X_hat = np.zeros_like(X)
    for i in range(X.shape[0]):
        out, *_ = forward(X[i], model)
        X_hat[i] = out
    return X_hat


def main():
    # 1) Argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=128, help="Neurônios na camada oculta (default: 128).")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas (default: 20).")
    parser.add_argument("--lr", type=float, default=0.05, help="Taxa de aprendizado (default: 0.05).")
    parser.add_argument("--no-plot", action="store_true", help="Não abrir janela de gráfico (apenas salvar loss.png).")
    parser.add_argument("--log-every", type=int, default=1, help="Frequência de logs de época (default: 1).")
    args = parser.parse_args()

    # 2) Carregar MNIST (sklearn) e split 60k/10k
    print("Carregando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float64)

    X_train, X_test = X[:60000], X[60000:]

    # 3) Normalização 0..1 (MinMax)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4) Inicializa AE: E->H->E
    E = X_train.shape[1]
    H = args.hidden
    O = E
    model = initialization(E, H, O)

    # 5) Treino (target = X_train).
    losses = MLPTrain(X_train, None, model, epochs=args.epochs, lr=args.lr, tol=1e-6, verbose=True, log_every=args.log_every)

    # 6) Avaliação por MSE no teste + imagens
    X_hat_test = reconstruct(X_test, model)
    mse_test = float(np.mean(0.5 * np.sum((X_test - X_hat_test)**2, axis=1)))
    print(f"MSE médio (teste): {mse_test:.6f}")

    # 7) Curva de loss
    plt.figure()
    plt.plot(losses)
    plt.title("Curva de Loss (MSE) - AE simples (MNIST)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)

    # 8) Reconstruções (10 primeiras)
    n = 10
    plt.figure(figsize=(n*1.2, 2.4))
    for i in range(n):
        # original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28,28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_ylabel("Original", rotation=90, size=10)
        # reconstrução
        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(X_hat_test[i].reshape(28,28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_ylabel("Reconstr.", rotation=90, size=10)
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)

    if not args.no_plot:
        plt.show()
    else:
        print("Figuras salvas: loss.png e reconstructions.png")

if __name__ == "__main__":
    main()
