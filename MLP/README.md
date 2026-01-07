# MLP from Scratch: Iris Dataset Classification

Este repositório contém uma implementação completa de um **Perceptron Multicamadas (MLP)** desenvolvida utilizando apenas **NumPy** para as operações matemáticas fundamentais da rede neural. O projeto foi aplicado para classificar as variedades da planta *Iris*.

## 🚀 Sobre o Projeto

O objetivo deste script é demonstrar o funcionamento interno de uma rede neural artificial, incluindo os processos de **Forward Propagation** e **Backpropagation (Online)**.

### Principais Características:
- **Arquitetura:** 1 camada oculta com número de neurônios ajustável.
- **Função de Ativação:** Sigmóide ($$f(x) = \frac{1}{1 + e^{-x}}$$).
- **Otimização:** Gradiente Descendente com atualização online (exemplo por exemplo).
- **Flexibilidade:** Suporta o dataset Iris (via Scikit-Learn ou CSV externo) e o clássico problema lógico **XOR**.
- **Métricas:** Gera relatórios de precisão, revocação, F1-score e matriz de confusão.

---

## 🛠️ Tecnologias e Requisitos

Para rodar o projeto, você precisará de Python 3.x e das seguintes bibliotecas:
* **NumPy**: Processamento numérico.
* **Scikit-Learn**: Utilizado apenas para pré-processamento (StandardScaler) e carregamento de dados.
* **Matplotlib**: Geração da curva de Loss.

Instale as dependências com:
```bash
pip install numpy scikit-learn matplotlib
```
## Como Executar

O script é configurado via linha de comando para facilitar testes com diferentes hiperparâmetros.

* Execução Padrão (Iris Dataset)
```bash
python MLP_iris.py
```

* Testando com o Problema XOR
```bash
python MLP_iris.py --xor
```

* Usando um CSV próprio
```bash
python MLP_iris.py --csv "caminho/para/seu/arquivo.csv"
```

* Ajustando Hiperparâmetros
```bash
python MLP_iris.py --lr 0.05 --epochs 2000 --hidden 12
```
