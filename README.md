# Treinando PyTorch

Coleção de notebooks em Python que ilustram conceitos-chave de redes neurais com PyTorch, desenvolvidos a partir das aulas do curso Treinando uma Rede Neural, da formação de Deep Learning com PyTorch da Alura. O repositório cobre funções de perda, algoritmos de otimização e fluxos completos de carregamento, treinamento e validação de modelos em problemas de classificação e regressão.

## Notebooks
- **01-FuncoesDePerda.ipynb**: Apresenta o papel das funções de perda em classificação (CrossEntropyLoss aplicada ao dataset Wine) e regressão (MSELoss e L1Loss com o dataset Diabetes), incluindo preparo de dados, envio para GPU e definição de redes MLP simples.
- **03-Otimizacao.ipynb**: Explora a descida do gradiente estocástica (`optim.SGD`), a importância da padronização de atributos e a visualização da fronteira de decisão de um classificador treinado em duas características do dataset Wine.
- **04-CarregamentoDeDados.ipynb**: Demonstra o uso dos datasets e transforms do `torchvision` com o MNIST, a construção de `DataLoader` com mini-batches, além de um loop completo de treinamento em GPU usando um MLP com `CrossEntropyLoss` e `Adam`.
- **05-CarregamentoDeDados2.ipynb**: Constrói um pipeline completo de regressão para o Bike Sharing Dataset, abordando download e preparação dos dados, implementação de um `Dataset` customizado, treino e validação com `model.train()`/`model.eval()` e monitoramento de perdas usando `L1Loss`.

## Datasets Utilizados
- Dataset Wine (`sklearn.datasets.load_wine`) para classificação multiclasse.
- Dataset Diabetes (`sklearn.datasets.load_diabetes`) para regressão.
- MNIST (via `torchvision.datasets.MNIST`) para classificação de dígitos escritos à mão.
- Bike Sharing Dataset (UCI) com engenharia de um `Dataset` próprio para regressão de demanda de bicicletas.

## Requisitos
- Python 3.8+ recomendado.
- Dependências listadas em `requirements.txt`:
  - Jupyter Notebook
  - PyTorch e Torchvision
  - Scikit-learn, Pandas, NumPy, Matplotlib, Torchsummary
- GPU CUDA opcional, mas utilizada nos notebooks 04 e 05 para acelerar o treinamento.

Para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Como Executar
- Clone este repositório e instale as dependências.
- Inicie o servidor local:

```bash
jupyter notebook
```

- Abra o notebook de interesse pela interface do Jupyter, execute as células em ordem e ajuste hiperparâmetros conforme desejar. Caso não possua GPU, altere o dispositivo para CPU nas células indicadas.

## Aprofundamentos Recomendados
- Experimente variações de hiperparâmetros (taxa de aprendizado, arquitetura e batch size) para observar impactos na convergência.
- Substitua o MLP por arquiteturas convolucionais para o MNIST e compare resultados.
- Explore métricas adicionais (MAE, RMSE, acurácia) para complementar a análise de perdas durante a validação.
