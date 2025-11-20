import numpy as np
from sklearn.datasets import load_iris #dataset do iris
import matplotlib.pyplot as plt

def sinal_degrau(V):
    return 1 if V >= 0 else 0  

def perceptron_train(logica, classes, W, eta=0.1, n_iter=100):
    epoch = 0
    error = True
    error_history = []

    while error and epoch<n_iter:
        epoch += 1
        error = False
        total_error = 0

        print(f"\n--- Época {epoch} ---")

        for n in range(len(logica)):
            X_n = logica[n]
            classes_n = classes[n]

            V=np.dot(W, X_n)
            y = sinal_degrau(V)

            #Se houver erro
            if y != classes_n:
                error = True
                W = W + eta * (classes_n - y) * X_n
                print(f" -> Erro! W atualizado para {np.round(W, 2)}")
                total_error += (classes_n - y)**2
            else:
                print(" -> OK!")

        avg_error = total_error / len(entradas)
        error_history.append(avg_error)
    
    if epoch==n_iter:
        print("ERRO! Excedeu as 100 épocas")
    else:
        print(f"Convergiu em {epoch} épocas.")  
            
    print(f"\n* Treinamento finalizado.")
    return {'weights': W, 'error_history': error_history, 'epochs': epoch}

def perceptron_test(X, W):
    V = np.dot(W, X)
    return sinal_degrau(V)

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    
    # Dados para as portas lógicas
    tabela_verdade_logica = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    classe_or = np.array([0, 1, 1, 1])
    classe_and = np.array([0, 0, 0, 1])
    classe_xor = np.array([0, 1, 1, 0])

    while True:
        print("   Selecione o problema para treinar")
        print("1. Lógica OR")
        print("2. Lógica AND")
        print("3. Lógica XOR")
        print("4. Teste Iris (Setosa vs. Outras)")
        print("5. Sair")
        escolha = input("Digite sua opção (1-5): ")

        if escolha == '1':
            logica_nome = "OR"
            entradas = tabela_verdade_logica
            classes_selecionadas = classe_or
            w_inicial = [0.5, 0.5, 0.5]
        elif escolha == '2':
            logica_nome = "AND"
            entradas = tabela_verdade_logica
            classes_selecionadas = classe_and
            w_inicial = [0.5, 0.5, 0.5]
        elif escolha == '3':
            logica_nome = "XOR"
            entradas = tabela_verdade_logica
            classes_selecionadas = classe_xor
            w_inicial = [0.5, 0.5, 0.5]
        elif escolha == '4':
            logica_nome = "Iris"
            
            #carrega o dataset
            iris = load_iris()
    
            #correspondendo ao código do mantovani em R
            X = iris.data[:, [1, 3]] 
            y = iris.target
            
            #classe 1 para Setosa e classe 0 para as outras
            classes_selecionadas = np.where(y == 0, 1, 0)
            
            #adiciona o bias e pesos
            entradas = np.c_[np.ones(X.shape[0]), X]
            w_inicial = [0.5, 0.5, 0.5]

        elif escolha == '5':
            print("Encerrando o programa. Até mais!")
            break
        else:
            print("Opção inválida! Por favor, escolha um número de 1 a 5.")
            continue
        
        # --- Bloco de Treinamento e Teste ---
        print(f"\n--- TREINANDO PERCEPTRON PARA: {logica_nome} ---")
        
        resultado = perceptron_train(entradas, classes_selecionadas, W=w_inicial, eta=0.1, n_iter=100)
        w_final = resultado['weights']
        
        print("\nPesos finais:", np.round(w_final, 2))
    
        #grafico
        if escolha == '4': #plota apenas para iris
            #curva de convergência do erro
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, resultado['epochs'] + 1), resultado['error_history'], marker='o')
            plt.title('Curva de Convergência do Erro')
            plt.xlabel('Época')
            plt.ylabel('Erro Quadrático Médio')
            plt.grid(True)
            plt.show()

            #hiperplano separador
            plt.figure(figsize=(8, 6))
            
            plt.scatter(X[classes_selecionadas == 1, 0], X[classes_selecionadas == 1, 1], color='red', marker='o', label='Setosa (Classe 1)')
            plt.scatter(X[classes_selecionadas == 0, 0], X[classes_selecionadas == 0, 1], color='blue', marker='x', label='Outras (Classe 0)')

            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            m, n = -w_final[1] / w_final[2], -w_final[0] / w_final[2]
            
            x1_reta = np.array([x1_min, x1_max])
            x2_reta = m * x1_reta + n
            
            plt.plot(x1_reta, x2_reta, 'g-', label='Hiperplano Separador')
            plt.title('Dataset Iris com Hiperplano do Perceptron')
            # MUDANÇA: Legenda do eixo X atualizada
            plt.xlabel('Sepal Width (cm)')
            plt.ylabel('Petal Width (cm)')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        input("\nPressione Enter para voltar ao menu...")
