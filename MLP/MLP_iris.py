"""
MLP (NumPy) para classificar o dataset Iris
-------------------------------------------
Este script implementa do zero um Perceptron Multicamadas (MLP) com:
 - 1 camada oculta
 - função de ativação sigmóide
 - backpropagation "online" (atualiza após cada exemplo)
 - métricas de avaliação e curva de loss


 pode usar **seu CSV do Iris** no formato UCI (4 floats + 1 rótulo de texto)


Como executar:
    # Usando um CSV local :
    python MLP_iris.py --csv "C:\\caminho\\para\\iris.data.csv" --no-plot
    
    # Sem CSV (usa o load_iris do scikit-learn):
    python MLP_iris.py


Requisitos:
    numpy, scikit-learn, matplotlib
"""

# Bibliotecas padrão
import os           # manipulação de caminhos/arquivos
import argparse     # parse de argumentos de linha de comando

# Numérico
import numpy as np  # arrays e operações vetorizadas

# Métricas e pré-processamento
from sklearn.model_selection import train_test_split     # split treino/teste
from sklearn.preprocessing import StandardScaler         # padronização (média 0, desvio 1)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Visualização
import matplotlib.pyplot as plt                          # gráfico da curva de loss


# Carregamento opcional do Iris via CSV
def load_iris_from_csv(path: str):
    # Carrega tudo como string para poder tratar linhas vazias e rótulos
    raw = np.genfromtxt(path, delimiter=",", dtype=str)

    # Remove linhas completamente vazias (às vezes há uma linha final em branco)
    raw = raw[~np.all(raw == "", axis=1)]

    # Primeiras 4 colunas -> float64 (features)
    X = raw[:, 0:4].astype(np.float64)

    # Última coluna -> rótulos de classe (strings)
    labels = raw[:, 4]

    # Mapeia nomes únicos para índices (ordem alfabética por padrão do np.unique)
    classes = np.unique(labels)                 # ex.: ['Iris-setosa','Iris-versicolor','Iris-virginica']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Converte cada rótulo para seu índice [0,1,2]
    y_idx = np.array([class_to_idx[l] for l in labels], dtype=int)

    # One-hot a partir dos índices
    Y = np.eye(len(classes))[y_idx]

    return X, Y, y_idx


# Definição do modelo MLP (NumPy puro)
class Modelo:
    """
    Estrutura simples para guardar:
      - Wh : pesos da camada oculta (shape: H x (E+1))  -> inclui bias como 1ª coluna
      - Wo : pesos da camada de saída (shape: O x (H+1)) -> inclui bias como 1ª coluna
      - E, H, O : tamanhos (entradas, oculto, saídas)
    """
    __slots__ = ("Wh", "Wo", "E", "H", "O")

    def __init__(self, Wh, Wo, sizes):
        self.Wh = Wh
        self.Wo = Wo
        self.E, self.H, self.O = sizes


def initialization(inputs, hidden, outputs, rng=None):
    """
    Inicializa pesos com valores aleatórios pequenos ao redor de 0.
    Adicionamos +1 na dimensão de entrada e da oculta por causa do bias.

    Parâmetros
    ----------
    inputs  : int  -> E (nº de atributos)
    hidden  : int  -> H (neurônios da camada oculta)
    outputs : int  -> O (neurônios de saída / nº de classes)
    rng     : np.random.Generator (opcional) para reprodutibilidade

    Retorna
    -------
    model : Modelo
    """
    if rng is None:
        rng = np.random.default_rng(1234)  # semente fixa p/ resultados reprodutíveis

    # Wh: pesos da oculta. Cada linha é um neurônio da oculta.
    #     Cada coluna corresponde a um atributo da entrada + 1 coluna extra p/ bias.
    Wh = rng.uniform(-0.5, 0.5, size=(hidden, inputs + 1))

    # Wo: pesos da saída. Idéia análoga: cada linha é um neurônio da saída.
    #     Colunas: ativações da oculta + 1 coluna extra de bias.
    Wo = rng.uniform(-0.5, 0.5, size=(outputs, hidden + 1))

    return Modelo(Wh, Wo, (inputs, hidden, outputs))


def sigmoid(x):
    """
    Função de ativação sigmóide.
    Recebe um array e retorna 1/(1+exp(-x)) elemento a elemento.
    """
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid_from_output(s):
    """
    Derivada da sigmóide em função da **saída** já sigmoidada.
    Se s = sigmoid(x), então d/dx sigmoid(x) = s * (1 - s)

    Usar esta forma evita recomputar sigmoid(x) duas vezes.
    """
    return s * (1.0 - s)


def forward(example, model: Modelo):
    # Concatena o bias (=1) na frente do vetor de entrada -> shape (E+1,)
    inp = np.concatenate([np.array([1.0]), example])

    # Camada oculta: soma ponderada (Wh @ inp) -> shape (H,)
    net_hidden = model.Wh @ inp

    # Aplica a ativação sigmóide na oculta -> shape (H,)
    hidden_out = sigmoid(net_hidden)

    # Concatena bias na frente das ativações da oculta -> shape (H+1,)
    h_bias = np.concatenate([np.array([1.0]), hidden_out])

    # Camada de saída: soma ponderada (Wo @ h_bias) -> shape (O,)
    net_out = model.Wo @ h_bias

    # Saída final (após sigmóide) -> "probabilidades" por classe
    O_output = sigmoid(net_out)

    return O_output, net_hidden, hidden_out, net_out, inp, h_bias


def MLPTrain(X, Y, model: Modelo, epochs=1500, lr=0.1, tol=1e-6, verbose=True):
   
    N = X.shape[0]       # nº de exemplos
    losses = []          # histórico de loss por época

    for ep in range(epochs):
        epoch_loss = 0.0

        # Embaralha a ordem dos exemplos a cada época (boa prática)
        for i in np.random.permutation(N):
            x, y = X[i], Y[i]

            # 1) FORWARD
            O_out, net_h, h_out, net_o, inp, h_bias = forward(x, model)

            # 2) ERRO no espaço da saída e loss (MSE/2 por exemplo)
            err = y - O_out
            epoch_loss += 0.5 * np.sum(err**2)

            # 3) BACKPROP - Gradientes
            # 3a) Saída: dO = (y - O_out) * d(sigmoid)/dnet
            dO = err * d_sigmoid_from_output(O_out)          # shape (O,)

            # 3b) Oculta: propaga o gradiente da saída para a oculta
            #     Atenção: a 1ª coluna de Wo é do bias — NÃO propague por ela.
            Wo_no_bias = model.Wo[:, 1:]                     # shape (O, H)
            dh = (Wo_no_bias.T @ dO) * d_sigmoid_from_output(h_out)  # shape (H,)

            # 4) ATUALIZAÇÃO DOS PESOS (gradiente ascend. no erro detalhado)
            # Saída: Wo += lr * (dO outer h_bias)
            model.Wo += lr * np.outer(dO, h_bias)            # (O, H+1)

            # Oculta: Wh += lr * (dh outer inp)
            model.Wh += lr * np.outer(dh, inp)               # (H, E+1)

        # Loss médio da época (boa métrica para acompanhar a convergência)
        epoch_loss /= N
        losses.append(epoch_loss)

        # Logs amigáveis
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"Epoch {ep+1}/{epochs} - Loss: {epoch_loss:.6f}")

        # Critério de parada opcional por tolerância
        if epoch_loss < tol:
            if verbose:
                print(f"Parada por tolerância na época {ep+1} (loss={epoch_loss:.6f})")
            break

    return np.array(losses)


def predict(X, model: Modelo):
    """
    Gera a predição de classe (argmax) para cada linha de X.
    Retorna um vetor de inteiros (0,1,2).
    """
    preds = np.zeros((X.shape[0],), dtype=int)
    for i in range(X.shape[0]):
        O_out, *_ = forward(X[i], model)
        preds[i] = int(np.argmax(O_out))
    return preds


# Função principal (pipeline completo)
def main():
   
    # 1) Argumentos de CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Caminho para iris.data.csv (formato UCI).")
    parser.add_argument("--xor", action="store_true", 
                        help="Usar dataset XOR com MLP implementado.")
    parser.add_argument("--xor-sklearn", action="store_true",
                        help="Rodar XOR usando o MLPClassifier do scikit-learn")
    parser.add_argument("--hidden", type=int, default=8,
                        help="Neurônios na camada oculta (default: 8).")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="Número de épocas de treino (default: 1500).")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Taxa de aprendizado (default: 0.1).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Não abrir janela de gráfico (apenas salvar loss.png).")
    args = parser.parse_args()

  
    # 2) Carregar os dados
    if args.xor:
        X = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ], dtype=np.float64)

        y = np.array([0, 1, 1, 0]) # saída esperada
        Y = np.eye(2)[y] # 2 classes

        X_train, X_test = X, X
        Y_train, Y_test = Y, Y
        y_train, y_test = y, y
    
    elif args.xor_sklearn:
        X = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ], dtype=np.float64)
        y = np.array([0, 1, 1, 0])

        # Scikit-learn MLP
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(
            hidden_layer_sizes=(3,),
            activation="logistic",
            solver="sgd",
            learning_rate_init=0.3,
            max_iter=5000,
            random_state=42
        )
        mlp.fit(X,y)
        print("Predições:", mlp.predict(X))
        print("Saídas reais:", y)
        print("Acurácia",  mlp.score(X,y))
        return

    else:
        if args.csv and os.path.exists(args.csv):
            # Caminho CSV fornecido -> usar nosso loader
            print(f"Lendo CSV: {args.csv}")
            X, Y, y = load_iris_from_csv(args.csv)
        else:
            # Sem CSV -> fallback: usar versão empacotada no scikit-learn
            from sklearn.datasets import load_iris
            data = load_iris()
            X = data["data"].astype(np.float64)      # (150, 4)
            y = data["target"].astype(int)           # (150,)
            Y = np.eye(len(np.unique(y)))[y]         # one-hot a partir dos índices

        # 3) Split + padronização
        # Stratify=y mantém a proporção de classes no treino e teste
        X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
            X, Y, y, test_size=0.25, random_state=42, stratify=y
        )

        # Padroniza cada atributo para média 0 e desvio 1 (com base no TREINO)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)  # usa os mesmos parâmetros do treino

    # 4) Criar e treinar MLP
    E = X_train.shape[1]       # nº de atributos (4 no Iris)
    O = Y_train.shape[1]       # nº de classes (3 no Iris)
    H = args.hidden            # neurônios na oculta (ajustável por argumento)

    # Inicializa pesos
    model = initialization(E, H, O)

    # Treina e guarda o histórico de loss
    losses = MLPTrain(
        X_train, Y_train, model,
        epochs=args.epochs, lr=args.lr, tol=1e-6, verbose=True
    )

    # ----------------------
    # 5) Avaliação do modelo
    # ----------------------
    y_pred_train = predict(X_train, model)
    y_pred_test  = predict(X_test, model)

    print("Acurácia (treino):", round(accuracy_score(y_train, y_pred_train), 4))
    print("Acurácia (teste):",  round(accuracy_score(y_test,  y_pred_test),  4))
    print("\nMatriz de confusão (teste):")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nRelatório de classificação (teste):")
    print(classification_report(y_test, y_pred_test, digits=4))

    # 6) Curva de loss (MSE)
    plt.figure()
    plt.plot(losses)
    plt.title("Curva de Loss (MSE) - Treino")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)   # salva sempre (útil para anexar em relatório)

    # Por padrão abrimos o gráfico; com --no-plot evitamos bloquear a execução
    if not args.no_plot:
        plt.show()
    else:
        print("Figura salva em loss.png (sem abrir janela).")


if __name__ == "__main__":
    main()
