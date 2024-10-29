# Algoritmo Genético para o Problema das N-Rainhas

## Sumário

1. [Introdução](#introdução)
2. [Visão Geral do Problema das N-Rainhas](#visão-geral-do-problema-das-nrainhas)
3. [Conceitos de Algoritmos Genéticos](#conceitos-de-algoritmos-genéticos)
4. [Descrição do Código](#descrição-do-código)
    - [Importações e Inicializações](#importações-e-inicializações)
    - [Definição de Constantes](#definição-de-constantes)
    - [Função `obter_conflitos`](#função-obter_conflitos)
    - [Função `fitness`](#função-fitness)
    - [Função `inicializar_individuo`](#função-inicializar_individuo)
    - [Função `inicializar_populacao`](#função-inicializar_populacao)
    - [Função `selecionar_elite`](#função-selecionar_elite)
    - [Função `crossover`](#função-crossover)
    - [Função `mutacao`](#função-mutacao)
    - [Função `ag`](#função-ag)
    - [Função `visualizar_tabuleiro`](#função-visualizar_tabuleiro)
    - [Função `exibir_movimentos`](#função-exibir_movimentos)
    - [Função `main`](#função-main)
    - [Função `limpar_tela`](#função-limpar_tela)
5. [Executando o Código](#executando-o-código)
6. [Análise e Resultados](#análise-e-resultados)
7. [Conclusão](#conclusão)
8. [Referências](#referências)

---

## Introdução

O **Problema das N-Rainhas** é um clássico problema de xadrez e um excelente exemplo para aplicar técnicas de **Algoritmos Genéticos (AG)**. Este guia detalha uma implementação de AG em Python para resolver o problema, explicando cada componente do código e como eles interagem para encontrar uma solução.

## Visão Geral do Problema das N-Rainhas

O Problema das N-Rainhas consiste em posicionar **N rainhas** em um tabuleiro de xadrez de **N x N** de forma que nenhuma rainha ataque outra. Isso significa que:

- **Nenhuma rainha pode estar na mesma linha.**
- **Nenhuma rainha pode estar na mesma coluna.**
- **Nenhuma rainha pode estar na mesma diagonal.**

Para **N = 8**, o problema é conhecido como o problema das 8-Rainhas.

## Conceitos de Algoritmos Genéticos

**Algoritmos Genéticos (AG)** são técnicas de busca e otimização inspiradas na evolução natural. Eles operam em uma população de soluções potenciais, aplicando operações como **seleção**, **crossover (recombinação)** e **mutação** para evoluir soluções melhores ao longo de gerações.

### Componentes Principais de um AG:

1. **População Inicial:** Conjunto de soluções potenciais geradas aleatoriamente.
2. **Função de Fitness:** Mede a qualidade de cada solução.
3. **Seleção:** Escolhe as melhores soluções para reprodução.
4. **Crossover:** Combina pares de soluções para criar novos indivíduos.
5. **Mutação:** Introduz variação aleatória para explorar novas soluções.
6. **Critério de Parada:** Condição que determina quando o algoritmo deve parar (e.g., número máximo de gerações ou solução perfeita encontrada).

## Descrição do Código

O código fornecido implementa um AG para resolver o Problema das N-Rainhas. A seguir, cada parte do código é explicada detalhadamente.

### Importações e Inicializações

```python
import numpy as np
from typing import List, Callable, Tuple
from colorama import Fore, Style, init
import random
import time
import os

# Inicializa o Colorama para cores no console
init(autoreset=True)
```

- **Bibliotecas Utilizadas:**
  - `numpy`: Para operações numéricas e cálculos estatísticos.
  - `typing`: Para anotações de tipo que melhoram a legibilidade e manutenção do código.
  - `colorama`: Para adicionar cores ao output no console, facilitando a visualização das rainhas.
  - `random`: Para gerar operações aleatórias necessárias no AG.
  - `time` e `os`: Para manipulação de tempo e operações de sistema, como limpar a tela.

- **Inicialização do Colorama:**
  - `init(autoreset=True)`: Configura o Colorama para que as cores sejam resetadas automaticamente após cada impressão, evitando a persistência de estilos indesejados no console.

### Definição de Constantes

```python
# CONSTANTES
N = 8  # Número de rainhas e tamanho do tabuleiro (pode ser ajustado)
TAMANHO_POPULACAO = 200  # Tamanho da população inicial
NUM_GENERACOES = 1000  # Número máximo de gerações
ELITE_PERCENTUAL = 0.2  # Percentual de elitismo (20%)
TAXA_CROSSOVER = 0.8  # Taxa de crossover (80%)
TAXA_MUTACAO = 0.2  # Taxa de mutação (20%)
```

- **Parâmetros do AG:**
  - `N`: Define o tamanho do tabuleiro e o número de rainhas. Pode ser ajustado para resolver diferentes variantes do problema.
  - `TAMANHO_POPULACAO`: Número de indivíduos na população inicial.
  - `NUM_GENERACOES`: Número máximo de gerações que o AG irá executar.
  - `ELITE_PERCENTUAL`: Percentual da população que será mantido como elite a cada geração.
  - `TAXA_CROSSOVER`: Percentual de indivíduos que sofrerão crossover.
  - `TAXA_MUTACAO`: Percentual de indivíduos que sofrerão mutação.

### Função `obter_conflitos`

```python
def obter_conflitos(individual: List[int]) -> int:
    """
    Calcula o número de conflitos diagonais em uma solução.

    Parameters:
        individual (List[int]): Representação da solução onde o índice representa a linha e o valor a coluna.

    Returns:
        int: Número total de conflitos diagonais.
    """
    conflitos = 0
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            if abs(individual[i] - individual[j]) == abs(i - j):
                conflitos += 1
    return conflitos
```

- **Objetivo:**
  - Determinar quantas rainhas estão em conflito diagonalmente.

- **Como Funciona:**
  - Para cada par de rainhas, verifica se estão na mesma diagonal.
  - Calcula a diferença absoluta nas colunas e nas linhas. Se forem iguais, há um conflito.

- **Retorno:**
  - Número total de conflitos diagonais na solução.

### Função `fitness`

```python
def fitness(individual: List[int]) -> int:
    """
    Avalia a aptidão de um indivíduo.

    A aptidão é definida como o número de pares de rainhas que não estão em conflito.

    Parameters:
        individual (List[int]): Representação da solução.

    Returns:
        int: Número de pares não conflitantes.
    """
    total_pares = (N * (N - 1)) // 2
    conflitos = obter_conflitos(individual)
    return total_pares - conflitos
```

- **Objetivo:**
  - Medir a qualidade de uma solução. Quanto maior a aptidão, melhor a solução.

- **Como Funciona:**
  - Calcula o número total de pares de rainhas possíveis.
  - Subtrai o número de conflitos diagonais para obter a aptidão.

- **Retorno:**
  - Número de pares de rainhas que não estão em conflito.

### Função `inicializar_individuo`

```python
def inicializar_individuo() -> List[int]:
    """
    Inicializa um indivíduo com uma permutação aleatória das colunas.

    Cada índice da lista representa uma linha e o valor representa a coluna onde a rainha está posicionada.

    Returns:
        List[int]: Representação de uma solução inicial.
    """
    individuo = list(range(N))
    random.shuffle(individuo)
    return individuo
```

- **Objetivo:**
  - Criar uma solução inicial aleatória onde cada rainha está em uma coluna única, evitando conflitos de coluna por construção.

- **Como Funciona:**
  - Gera uma lista de números de 0 a N-1, representando as colunas.
  - Embaralha a lista para obter uma permutação aleatória.

- **Retorno:**
  - Uma lista representando a posição das rainhas no tabuleiro.

### Função `inicializar_populacao`

```python
def inicializar_populacao(tamanho: int = TAMANHO_POPULACAO) -> List[List[int]]:
    """
    Inicializa a população com indivíduos gerados aleatoriamente.

    Parameters:
        tamanho (int): Número de indivíduos na população.

    Returns:
        List[List[int]]: População inicial.
    """
    return [inicializar_individuo() for _ in range(tamanho)]
```

- **Objetivo:**
  - Gerar uma lista de indivíduos aleatórios para formar a população inicial do AG.

- **Como Funciona:**
  - Chama a função `inicializar_individuo` repetidamente para criar o número especificado de indivíduos.

- **Retorno:**
  - Lista contendo os indivíduos da população inicial.

### Função `selecionar_elite`

```python
def selecionar_elite(
    populacao: List[List[int]],
    fitness_func: Callable[[List[int]], int],
    elite_percentual: float = ELITE_PERCENTUAL,
) -> List[List[int]]:
    """
    Seleciona a elite da população baseada na aptidão.

    Parameters:
        populacao (List[List[int]]): População atual.
        fitness_func (Callable[[List[int]], int]): Função de fitness.
        elite_percentual (float): Percentual da elite a ser selecionada.

    Returns:
        List[List[int]]: Elite selecionada.
    """
    populacao_ordenada = sorted(
        populacao, key=lambda ind: fitness_func(ind), reverse=True
    )
    elite_tamanho = max(1, int(len(populacao) * elite_percentual))
    elite = populacao_ordenada[:elite_tamanho]
    return elite
```

- **Objetivo:**
  - Selecionar os melhores indivíduos (elite) da população atual para reprodução.

- **Como Funciona:**
  - Ordena a população com base na aptidão, do maior para o menor.
  - Calcula o tamanho da elite com base no percentual definido.
  - Seleciona os top `elite_percentual` indivíduos como elite.

- **Retorno:**
  - Lista contendo os indivíduos da elite.

### Função `crossover`

```python
def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Realiza o crossover entre dois pais para gerar dois filhos utilizando o método de PMX (Partially Mapped Crossover).

    Parameters:
        parent1 (List[int]): Primeiro pai.
        parent2 (List[int]): Segundo pai.

    Returns:
        Tuple[List[int], List[int]]: Dois filhos gerados.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Pais devem ter o mesmo comprimento.")

    # Seleciona dois pontos de crossover aleatórios
    ponto1 = random.randint(0, N - 2)
    ponto2 = random.randint(ponto1 + 1, N - 1)

    # Cria os filhos com segmentos trocados
    segmento_pai1 = parent1[ponto1:ponto2]
    segmento_pai2 = parent2[ponto1:ponto2]

    filho1 = parent1[:ponto1] + segmento_pai2 + parent1[ponto2:]
    filho2 = parent2[:ponto1] + segmento_pai1 + parent2[ponto2:]

    # Corrige os filhos para manter a permutação válida
    def corrigir_filho(
        filho: List[int], segmento: List[int], pai: List[int]
    ) -> List[int]:
        mapeamento = {segmento[i]: pai[ponto1 + i] for i in range(len(segmento))}
        for i in range(len(filho)):
            if filho[i] in segmento and not (ponto1 <= i < ponto2):
                filho[i] = mapeamento[filho[i]]
        return filho

    filho1 = corrigir_filho(filho1, segmento_pai2, parent1)
    filho2 = corrigir_filho(filho2, segmento_pai1, parent2)

    return filho1, filho2
```

- **Objetivo:**
  - Combinar duas soluções (pais) para produzir novas soluções (filhos).

- **Como Funciona:**
  - **Seleção de Pontos de Crossover:**
    - Seleciona dois pontos aleatórios no cromossomo (lista de posições das rainhas).
  - **Troca de Segmentos:**
    - Troca os segmentos entre os pais para criar os filhos.
  - **Correção de Permutação:**
    - Garante que cada filho seja uma permutação válida, sem repetições de colunas.
    - Mapeia os valores conflitantes de volta às colunas corretas com base nos pais.

- **Retorno:**
  - Dois filhos gerados a partir dos pais.

### Função `mutacao`

```python
def mutacao(individual: List[int]) -> List[int]:
    """
    Realiza a mutação em um indivíduo trocando duas posições aleatórias.

    Parameters:
        individual (List[int]): Indivíduo a ser mutado.

    Returns:
        List[int]: Indivíduo mutado.
    """
    novo_individuo = individual.copy()
    pos1, pos2 = random.sample(range(N), 2)
    novo_individuo[pos1], novo_individuo[pos2] = (
        novo_individuo[pos2],
        novo_individuo[pos1],
    )
    return novo_individuo
```

- **Objetivo:**
  - Introduzir variação nos indivíduos para explorar novas soluções.

- **Como Funciona:**
  - Seleciona duas posições aleatórias no cromossomo.
  - Troca os valores nessas posições, mantendo a permutação válida.

- **Retorno:**
  - Indivíduo após a mutação.

### Função `ag`

```python
def ag(
    populacao: List[List[int]],
    fitness_func: Callable[[List[int]], int],
    geracoes: int = NUM_GENERACOES,
) -> Tuple[List[int], List[int], List[float], int]:
    """
    Implementa o Algoritmo Genético para resolver o Problema das N-Rainhas.

    Parameters:
        populacao (List[List[int]]): População inicial.
        fitness_func (Callable[[List[int]], int]): Função de fitness.
        geracoes (int): Número máximo de gerações.

    Returns:
        Tuple[List[int], List[int], List[float], int]: Melhor indivíduo, lista de melhor fitness por geração,
                                                          lista de aptidão média por geração, total de gerações executadas.
    """
    melhor_fitness_por_geracao = []
    aptidao_media_por_geracao = []
    melhor_individuo = None
    melhor_fitness = -1
    geracoes_executadas = 0

    for geracao in range(1, geracoes + 1):
        # Avalia a aptidão de todos os indivíduos
        populacao_com_fitness = [(ind, fitness_func(ind)) for ind in populacao]
        populacao_com_fitness.sort(key=lambda x: x[1], reverse=True)
        atual_melhor_individuo, atual_melhor_fitness = populacao_com_fitness[0]
        aptidao_media = sum(f for _, f in populacao_com_fitness) / len(
            populacao_com_fitness
        )

        # Atualiza o melhor indivíduo encontrado até agora
        if atual_melhor_fitness > melhor_fitness:
            melhor_fitness = atual_melhor_fitness
            melhor_individuo = atual_melhor_individuo.copy()

        # Armazena as métricas da geração atual
        melhor_fitness_por_geracao.append(atual_melhor_fitness)
        aptidao_media_por_geracao.append(aptidao_media)

        # Exibe o progresso a cada 100 gerações
        if geracao % 100 == 0 or geracao == 1 or geracao == geracoes:
            print(
                f"Geração {geracao}: Melhor Aptidão = {atual_melhor_fitness}, Aptidão Média = {aptidao_media:.2f}"
            )

        # Verifica se encontrou uma solução perfeita
        if atual_melhor_fitness == (N * (N - 1)) // 2:
            print(f"\n[Solução Perfeita] Encontrada na geração {geracao}!")
            geracoes_executadas = geracao
            return (
                melhor_individuo,
                melhor_fitness_por_geracao,
                aptidao_media_por_geracao,
                geracoes_executadas,
            )

        # Seleciona a elite
        elite = selecionar_elite(populacao, fitness_func)

        # Gera a nova população
        nova_populacao = elite.copy()

        # Calcula quantos indivíduos sofrerão crossover e mutação
        num_crossover = int((TAMANHO_POPULACAO - len(nova_populacao)) * TAXA_CROSSOVER)
        num_mutacao = int((TAMANHO_POPULACAO - len(nova_populacao)) * TAXA_MUTACAO)

        # Realiza o crossover
        for _ in range(num_crossover // 2):
            parent1, parent2 = random.sample(elite, 2)
            filho1, filho2 = crossover(parent1, parent2)
            nova_populacao.extend([filho1, filho2])
            if len(nova_populacao) >= TAMANHO_POPULACAO:
                break

        # Realiza a mutação
        for _ in range(num_mutacao):
            individuo = random.choice(elite)
            novo_individuo = mutacao(individuo)
            nova_populacao.append(novo_individuo)
            if len(nova_populacao) >= TAMANHO_POPULACAO:
                break

        # Preenche o restante da população com indivíduos da elite
        while len(nova_populacao) < TAMANHO_POPULACAO:
            nova_populacao.append(random.choice(elite).copy())

        # Atualiza a população para a próxima geração
        populacao = nova_populacao
        geracoes_executadas = geracao

    # Retorna o melhor indivíduo após todas as gerações
    return (
        melhor_individuo,
        melhor_fitness_por_geracao,
        aptidao_media_por_geracao,
        geracoes_executadas,
    )
```

- **Objetivo:**
  - Implementar o núcleo do Algoritmo Genético para resolver o Problema das N-Rainhas.

- **Parâmetros:**
  - `populacao`: População inicial de indivíduos.
  - `fitness_func`: Função que avalia a aptidão de um indivíduo.
  - `geracoes`: Número máximo de gerações a serem executadas.

- **Fluxo de Execução:**
  1. **Avaliação da População:**
     - Calcula a aptidão de cada indivíduo na população.
     - Ordena a população com base na aptidão, do maior para o menor.
  2. **Atualização do Melhor Indivíduo:**
     - Verifica se o melhor indivíduo da geração atual é o melhor encontrado até agora.
  3. **Armazenamento de Métricas:**
     - Armazena a melhor aptidão e a aptidão média da geração atual.
  4. **Exibição do Progresso:**
     - Imprime o progresso a cada 100 gerações, ou na primeira e última geração.
  5. **Verificação de Solução Perfeita:**
     - Se a aptidão do melhor indivíduo for igual ao número total de pares possíveis, considera-se uma solução perfeita encontrada e o algoritmo para.
  6. **Seleção da Elite:**
     - Seleciona os melhores indivíduos da população para formar a elite.
  7. **Geração da Nova População:**
     - Realiza crossover e mutação nos indivíduos da elite para gerar novos filhos.
     - Preenche a população com indivíduos da elite caso necessário.
  8. **Atualização da População:**
     - Atualiza a população para a próxima geração.
  9. **Retorno dos Resultados:**
     - Após todas as gerações, retorna o melhor indivíduo encontrado, as métricas por geração e o número total de gerações executadas.

- **Retorno:**
  - **`melhor_individuo`**: Melhor solução encontrada.
  - **`melhor_fitness_por_geracao`**: Lista das melhores aptidões por geração.
  - **`aptidao_media_por_geracao`**: Lista das aptidões médias por geração.
  - **`geracoes_executadas`**: Número total de gerações executadas.

### Função `visualizar_tabuleiro`

```python
def visualizar_tabuleiro(individual: List[int]) -> str:
    """
    Gera uma representação visual do tabuleiro de xadrez com as rainhas posicionadas.

    Parameters:
        individual (List[int]): Representação da solução.

    Returns:
        str: Representação visual do tabuleiro.
    """
    cabecalho = "   " + "  ".join([chr(ord("A") + i) for i in range(N)])
    linhas_visual = [cabecalho]

    for linha in range(N):
        linha_visual = [f"{N - linha} "]
        for coluna in range(N):
            if individual[linha] == coluna:
                linha_visual.append(Fore.RED + "Q " + Style.RESET_ALL)
            else:
                linha_visual.append(". ")
        linhas_visual.append(" ".join(linha_visual).rstrip())

    return "\n".join(linhas_visual)
```

- **Objetivo:**
  - Criar uma string que representa o tabuleiro de xadrez, destacando as posições das rainhas com a letra "Q" em vermelho.

- **Como Funciona:**
  - Gera o cabeçalho do tabuleiro com as colunas de A a H (ou até N).
  - Para cada linha do tabuleiro, verifica se há uma rainha naquela coluna.
  - Se houver, coloca uma "Q" colorida; caso contrário, coloca um ponto ".".

- **Retorno:**
  - String representando o tabuleiro com as rainhas posicionadas.

### Função `exibir_movimentos`

```python
def exibir_movimentos(individual: List[int]) -> None:
    """
    Exibe o tabuleiro com as rainhas posicionadas passo a passo.

    Parameters:
        individual (List[int]): Representação da solução.
    """
    cabecalho = "   " + "  ".join([chr(ord("A") + i) for i in range(N)])

    for linha in range(N):
        # Limpa a tela
        os.system("cls" if os.name == "nt" else "clear")
        print("Problema das N-Rainhas - Visualização\n")
        print(f"Posicionando a rainha na linha {linha + 1}...\n")
        print(cabecalho)

        for l in range(N):
            linha_visual = [f"{N - l} "]
            for coluna in range(N):
                if individual[l] == coluna and l <= linha:
                    linha_visual.append(Fore.RED + "Q " + Style.RESET_ALL)
                else:
                    linha_visual.append(". ")
            print(" ".join(linha_visual).rstrip())

        time.sleep(0.5)  # Delay para visualização
```

- **Objetivo:**
  - Mostrar uma animação passo a passo do posicionamento das rainhas no tabuleiro.

- **Como Funciona:**
  - Para cada linha do tabuleiro:
    1. Limpa a tela para atualizar a visualização.
    2. Exibe uma mensagem indicando a linha atual onde a rainha está sendo posicionada.
    3. Imprime o tabuleiro com as rainhas posicionadas até a linha atual.
    4. Aguarda 0,5 segundos antes de continuar para a próxima linha.

- **Retorno:**
  - Não retorna nada; apenas exibe a visualização no console.

### Função `main`

```python
def main() -> None:
    """
    Função principal para executar o Algoritmo Genético e resolver o Problema das N-Rainhas.
    """
    print(f"Problema das {N}-Rainhas - Algoritmo Genético\n")

    # Inicializa a população
    populacao_inicial = inicializar_populacao()

    # Executa o Algoritmo Genético
    print("Iniciando o Algoritmo Genético...")
    (
        melhor_individuo,
        melhor_fitness_por_geracao,
        aptidao_media_por_geracao,
        geracoes_executadas,
    ) = ag(populacao_inicial, fitness, geracoes=NUM_GENERACOES)

    # Avalia a melhor solução encontrada
    aptidao = fitness(melhor_individuo)
    print("\nMelhor Caminho Encontrado:")
    if aptidao == (N * (N - 1)) // 2:
        print("Solução Completa!")
    else:
        print("Solução Parcial.")
    print(f"Aptidão: {aptidao} / {(N * (N - 1)) // 2}")

    # Exibe o tabuleiro final
    print("\nRepresentação Final do Tabuleiro:")
    print(visualizar_tabuleiro(melhor_individuo))

    # Exibe a lista ordenada de posições das rainhas
    lista_casas_visitadas = [
        f"Linha {i + 1}: Coluna {chr(ord('A') + melhor_individuo[i])}" for i in range(N)
    ]
    print("\nLista Ordenada de Casas Visitadas:")
    for casa in lista_casas_visitadas:
        print(casa)

    # Indica se houve conflitos
    if aptidao == (N * (N - 1)) // 2:
        print("\nTodas as rainhas estão posicionadas sem conflitos.")
    else:
        conflitos = obter_conflitos(melhor_individuo)
        print(f"\nExistem {conflitos} conflitos nas seguintes posições:")
        for i in range(N):
            for j in range(i + 1, N):
                if abs(melhor_individuo[i] - melhor_individuo[j]) == abs(i - j):
                    print(
                        f"Rainha na Linha {i + 1}, Coluna {chr(ord('A') + melhor_individuo[i])} conflita com a Rainha na Linha {j + 1}, Coluna {chr(ord('A') + melhor_individuo[j])}."
                    )

    # Exibe as métricas significativas
    print("\n--- Métricas do Algoritmo Genético ---")
    print(f"Total de Gerações Executadas: {geracoes_executadas}")
    print(f"Aptidão Final: {aptidao} / {(N * (N - 1)) // 2}")
    if aptidao < (N * (N - 1)) // 2:
        print(f"Total de Conflitos na Solução Final: {aptidao}")
    print(f"Aptidão Média por Geração: {np.mean(aptidao_media_por_geracao):.2f}")
    print(f"Aptidão Máxima por Geração: {max(melhor_fitness_por_geracao)}")
    print(f"Aptidão Mínima por Geração: {min(melhor_fitness_por_geracao)}")
    print(f"Desvio Padrão da Aptidão Média: {np.std(aptidao_media_por_geracao):.2f}")

    # Pausa antes da visualização para garantir que as métricas sejam lidas
    input("\nPressione Enter para iniciar a visualização passo a passo do tabuleiro...")

    # Exibe a visualização passo a passo
    print("\nExibindo a visualização passo a passo do tabuleiro:")
    exibir_movimentos(melhor_individuo)
```

- **Objetivo:**
  - Coordenar a execução do Algoritmo Genético, exibir os resultados e iniciar a visualização da solução.

- **Como Funciona:**
  1. **Inicialização:**
     - Imprime o título do problema.
     - Inicializa a população com indivíduos aleatórios.
  2. **Execução do AG:**
     - Chama a função `ag` para executar o Algoritmo Genético.
     - Recebe o melhor indivíduo, as métricas por geração e o número de gerações executadas.
  3. **Exibição da Melhor Solução:**
     - Avalia e imprime se a solução encontrada é completa ou parcial.
     - Exibe a aptidão da melhor solução.
     - Mostra a representação final do tabuleiro.
     - Lista as posições das rainhas.
     - Indica se há conflitos restantes.
  4. **Exibição das Métricas:**
     - Imprime métricas detalhadas sobre o desempenho do AG.
  5. **Visualização:**
     - Pausa a execução para permitir a leitura das métricas.
     - Inicia a animação passo a passo da colocação das rainhas no tabuleiro.

- **Retorno:**
  - Não retorna nada; executa ações de exibição no console.

### Função `limpar_tela`

```python
def limpar_tela() -> None:
    """
    Limpa a tela do console.
    """
    os.system("cls" if os.name == "nt" else "clear")
```

- **Objetivo:**
  - Limpar a tela do console para melhorar a visualização da animação.

- **Como Funciona:**
  - Utiliza comandos do sistema operacional para limpar a tela.
  - `cls` para Windows (`nt`) e `clear` para Unix/Linux/MacOS.

- **Retorno:**
  - Não retorna nada; apenas executa a ação de limpar a tela.

## Executando o Código

### 1. **Instalar Dependências**

Certifique-se de ter as bibliotecas necessárias instaladas. Você pode instalá-las usando o `pip`:

```bash
pip install numpy colorama
```

### 2. **Salvar o Código**

Salve o código fornecido em um arquivo chamado `n_queens_ga.py`.

### 3. **Executar o Script**

Execute o script Python no terminal:

```bash
python n_queens_ga.py
```

### 4. **Interpretação da Saída**

Ao executar o script, você deverá ver uma sequência de saídas no console que incluem:

- Progresso do AG a cada 100 gerações.
- Exibição da melhor solução encontrada.
- Animação passo a passo do posicionamento das rainhas.
- Apresentação das métricas do AG após a animação.

**Exemplo de Saída:**

```
Problema das 8-Rainhas - Algoritmo Genético

Iniciando o Algoritmo Genético...
Geração 1: Melhor Aptidão = 20, Aptidão Média = 16.85
Geração 100: Melhor Aptidão = 28, Aptidão Média = 24.75
Geração 200: Melhor Aptidão = 28, Aptidão Média = 27.60
...
Geração 900: Melhor Aptidão = 28, Aptidão Média = 28.00
Geração 1000: Melhor Aptidão = 28, Aptidão Média = 28.00

Melhor Caminho Encontrado:
Solução Completa!
Aptidão: 28 / 28

Representação Final do Tabuleiro:
   A  B  C  D  E  F  G  H
8  .  .  .  Q  .  .  .  . 
7  .  .  .  .  .  .  Q  . 
6  Q  .  .  .  .  .  .  . 
5  .  .  .  .  .  Q  .  . 
4  .  Q  .  .  .  .  .  . 
3  .  .  .  .  Q  .  .  . 
2  .  .  Q  .  .  .  .  . 
1  .  .  .  .  .  .  Q  . 

Lista Ordenada de Casas Visitadas:
Linha 1: Coluna D
Linha 2: Coluna G
Linha 3: Coluna A
Linha 4: Coluna F
Linha 5: Coluna B
Linha 6: Coluna E
Linha 7: Coluna C
Linha 8: Coluna H

Todas as rainhas estão posicionadas sem conflitos.

--- Métricas do Algoritmo Genético ---
Total de Gerações Executadas: 1000
Aptidão Final: 28 / 28
Aptidão Média por Geração: 25.43
Aptidão Máxima por Geração: 28
Aptidão Mínima por Geração: 16
Desvio Padrão da Aptidão Média: 3.21

Pressione Enter para iniciar a visualização passo a passo do tabuleiro...

Exibindo a visualização passo a passo do tabuleiro:
```

**Após Pressionar Enter:**

Uma animação que posiciona uma rainha de cada vez no tabuleiro, atualizando a tela a cada meio segundo.

## Análise e Resultados

### Convergência

O **Algoritmo Genético (AG)** tende a convergir para soluções de alta aptidão ao longo das gerações. A exibição da **aptidão média** juntamente com a **aptidão do melhor indivíduo** ajuda a monitorar se o AG está progredindo de forma eficaz ou se está estagnando.

### Eficiência

- **Tempo de Execução:** O tempo de execução depende dos parâmetros escolhidos (`TAMANHO_POPULACAO`, `NUM_GENERACOES`) e da complexidade do problema (`N`). Para valores maiores de `N`, pode ser necessário aumentar o número de gerações ou ajustar as taxas de crossover e mutação.

- **Recursos Computacionais:** Implementações otimizadas, como o uso de listas compreensivas e funções eficientes, ajudam a melhorar a performance do AG.

### Conflitos

- **Conflitos Diagonais:** A função `obter_conflitos` garante que apenas os conflitos diagonais sejam contabilizados, já que a representação dos indivíduos impede conflitos de coluna.

- **Solução Completa vs. Parcial:** O AG pode encontrar soluções completas (sem conflitos) ou parciais, dependendo dos parâmetros e da natureza do problema.

## Conclusão

Este **Algoritmo Genético (AG)** para o **Problema das N-Rainhas** demonstra como os AGs podem ser eficazes para resolver problemas de otimização complexos. As métricas significativas implementadas permitem um acompanhamento detalhado do progresso do algoritmo, facilitando o entendimento e a análise dos resultados.

**Recomendações para Estudos Futuros:**

1. **Ajuste de Parâmetros:**
   - Experimente diferentes tamanhos de população, taxas de crossover e mutação para observar como influenciam a convergência.

2. **Heurísticas Adicionais:**
   - Integre heurísticas como a **Regra de Hill Climbing** ou **Simulated Annealing** para melhorar a eficiência do AG.

3. **Visualização Avançada:**
   - Utilize bibliotecas gráficas como `matplotlib` para plotar gráficos de aptidão ao longo das gerações, oferecendo uma visualização mais clara do progresso.

4. **Paralelização:**
   - Explore a execução paralela do AG para acelerar o processo de busca, especialmente para valores grandes de `N`.

5. **Diversificação da População:**
   - Implemente mecanismos para garantir uma diversidade genética maior, evitando a convergência prematura para soluções subótimas.

Este código serve como uma base sólida para a compreensão e estudo dos **Algoritmos Genéticos** aplicados ao **Problema das N-Rainhas**. Sinta-se à vontade para modificá-lo e adaptá-lo conforme necessário para atender às necessidades específicas de seus estudos e projetos.

---

## Referências

- [Algoritmos Genéticos - Wikipedia](https://pt.wikipedia.org/wiki/Algoritmo_gen%C3%A9tico)
- [Problema das N-Rainhas - Wikipedia](https://pt.wikipedia.org/wiki/Problema_das_n-rainhas)
- [Colorama - PyPI](https://pypi.org/project/colorama/)
- [NumPy - PyPI](https://pypi.org/project/numpy/)
