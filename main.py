# n_queens_ga.py

import numpy as np
from typing import List, Callable, Tuple
from colorama import Fore, Style, init
import random
import time
import os

# Inicializa o Colorama para cores no console
init(autoreset=True)

# CONSTANTES
N = 8  # Número de rainhas e tamanho do tabuleiro (pode ser ajustado)
TAMANHO_POPULACAO = 200  # Tamanho da população inicial
NUM_GENERACOES = 1000    # Número máximo de gerações
ELITE_PERCENTUAL = 0.2   # Percentual de elitismo (20%)
TAXA_CROSSOVER = 0.8     # Taxa de crossover (80%)
TAXA_MUTACAO = 0.2       # Taxa de mutação (20%)

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

def inicializar_populacao(tamanho: int = TAMANHO_POPULACAO) -> List[List[int]]:
    """
    Inicializa a população com indivíduos gerados aleatoriamente.

    Parameters:
        tamanho (int): Número de indivíduos na população.

    Returns:
        List[List[int]]: População inicial.
    """
    return [inicializar_individuo() for _ in range(tamanho)]

def selecionar_elite(populacao: List[List[int]], fitness_func: Callable[[List[int]], int], elite_percentual: float = ELITE_PERCENTUAL) -> List[List[int]]:
    """
    Seleciona a elite da população baseada na aptidão.

    Parameters:
        populacao (List[List[int]]): População atual.
        fitness_func (Callable[[List[int]], int]): Função de fitness.
        elite_percentual (float): Percentual da elite a ser selecionada.

    Returns:
        List[List[int]]: Elite selecionada.
    """
    populacao_ordenada = sorted(populacao, key=lambda ind: fitness_func(ind), reverse=True)
    elite_tamanho = max(1, int(len(populacao) * elite_percentual))
    elite = populacao_ordenada[:elite_tamanho]
    return elite

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
    def corrigir_filho(filho: List[int], segmento: List[int], pai: List[int]) -> List[int]:
        mapeamento = {segmento[i]: pai[ponto1 + i] for i in range(len(segmento))}
        for i in range(len(filho)):
            if filho[i] in segmento and not (ponto1 <= i < ponto2):
                filho[i] = mapeamento[filho[i]]
        return filho

    filho1 = corrigir_filho(filho1, segmento_pai2, parent1)
    filho2 = corrigir_filho(filho2, segmento_pai1, parent2)

    return filho1, filho2

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
    novo_individuo[pos1], novo_individuo[pos2] = novo_individuo[pos2], novo_individuo[pos1]
    return novo_individuo

def ag(populacao: List[List[int]], fitness_func: Callable[[List[int]], int], geracoes: int = NUM_GENERACOES) -> Tuple[List[int], List[int], List[float], int]:
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
        aptidao_media = sum(f for _, f in populacao_com_fitness) / len(populacao_com_fitness)

        # Atualiza o melhor indivíduo encontrado até agora
        if atual_melhor_fitness > melhor_fitness:
            melhor_fitness = atual_melhor_fitness
            melhor_individuo = atual_melhor_individuo.copy()

        # Armazena as métricas da geração atual
        melhor_fitness_por_geracao.append(atual_melhor_fitness)
        aptidao_media_por_geracao.append(aptidao_media)

        # Exibe o progresso a cada 100 gerações
        if geracao % 100 == 0 or geracao == 1 or geracao == geracoes:
            print(f"Geração {geracao}: Melhor Aptidão = {atual_melhor_fitness}, Aptidão Média = {aptidao_media:.2f}")

        # Verifica se encontrou uma solução perfeita
        if atual_melhor_fitness == (N * (N - 1)) // 2:
            print(f"\n[Solução Perfeita] Encontrada na geração {geracao}!")
            geracoes_executadas = geracao
            return melhor_individuo, melhor_fitness_por_geracao, aptidao_media_por_geracao, geracoes_executadas

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
    return melhor_individuo, melhor_fitness_por_geracao, aptidao_media_por_geracao, geracoes_executadas

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

def exibir_movimentos(individual: List[int]) -> None:
    """
    Exibe o tabuleiro com as rainhas posicionadas passo a passo.

    Parameters:
        individual (List[int]): Representação da solução.
    """
    cabecalho = "   " + "  ".join([chr(ord("A") + i) for i in range(N)])

    for linha in range(N):
        # Limpa a tela
        os.system('cls' if os.name == 'nt' else 'clear')
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

def main() -> None:
    """
    Função principal para executar o Algoritmo Genético e resolver o Problema das N-Rainhas.
    """
    print(f"Problema das {N}-Rainhas - Algoritmo Genético\n")

    # Inicializa a população
    populacao_inicial = inicializar_populacao()

    # Executa o Algoritmo Genético
    print("Iniciando o Algoritmo Genético...")
    melhor_individuo, melhor_fitness_por_geracao, aptidao_media_por_geracao, geracoes_executadas = ag(populacao_inicial, fitness, geracoes=NUM_GENERACOES)

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
    lista_casas_visitadas = [f"Linha {i + 1}: Coluna {chr(ord('A') + melhor_individuo[i])}" for i in range(N)]
    print("\nLista Ordenada de Casas Visitadas:")
    for casa in lista_casas_visitadas:
        print(casa)

    # Indica se houve conflitos
    if aptidao == (N * (N - 1)) // 2:
        print("\nParabéns! Todas as rainhas estão posicionadas sem conflitos.")
    else:
        conflitos = obter_conflitos(melhor_individuo)
        print(f"\nExistem {conflitos} conflitos nas seguintes posições:")
        for i in range(N):
            for j in range(i + 1, N):
                if abs(melhor_individuo[i] - melhor_individuo[j]) == abs(i - j):
                    print(f"Rainha na Linha {i + 1}, Coluna {chr(ord('A') + melhor_individuo[i])} conflita com a Rainha na Linha {j + 1}, Coluna {chr(ord('A') + melhor_individuo[j])}.")

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

def limpar_tela() -> None:
    """
    Limpa a tela do console.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()
