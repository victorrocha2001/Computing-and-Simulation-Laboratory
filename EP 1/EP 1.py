'''
EP 1 - MAP2212 - 2024
    Victor Rocha Cardoso Cruz
    NUSP: 11223757

    ENUNCIADO:
    Neste EP, você deve implementar um programa para estimar π.
        - Utilize a função randou similar em seu ambiente de programação para gerar pontos uniformemente distribuídos xi∈[−1,1]² , i∈{1,…,n};
        - Estime π pela proporção p=1n∑i=1nT(xi), onde T(x)=Ind(||x||2≤1) testa se x está dentro do círculo unitário;
        - Determine n de modo a obter uma estimativa que seja precisa com margem de erro de até 0.05%.
'''
# INÍCIO

import numpy as np


def main():
    print("____________________")
    print("** EP 1 INICIADO **")
    print("--------------------")
    
    # Definindo as variáveis do par ordenado (x,y) a ser testado, 
    #  bem como o número total de pontos sorteados Nt (n) e a soma de pontos 
    #  dentro do círculo unitário Nin (T(x))
    x, y, Nin, Nt = 0, 0, 0, 0
    # Definindo o valor de Pi da biblioteca numpy (PINp) e a variável para o PI calculado pela proporção (PIcalc)
    PINp, PIcalc = np.pi, 4
    # A variável sigma será responsável pelo cáculo do erro entre os Pis
    sigma = abs(PINp - PIcalc)

    # Executa a validação pelo método de Monte Carlo até que o erro seja menor que 0,05%
    while sigma > 0.0005:
        # Popula as variáveis do par ordenado, sorteando aleatóriamente pontos, tal qual xi, yi ∈ [-1,1], i ∈ {1,…,n}
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)

        # Para que o ponto (x,y) sorteado esteja dentro da circunferência unitária, ele deve ter distância r
        # menor ou igual a 1, o raio da circunferência. Portanto, será testado a igualdade entre r (a soma dos 
        # quadrados das coordenadas do ponto) e o raio da circunferência.
        if x**2 + y**2 <= 1:
            # O número de pontos dentro do circulo é incrementado
            Nin = Nin + 1
        
        # O número de pontos sorteados é incrementado
        Nt = Nt + 1
        
        # O valor PI é calculado, à cada loop, pela proporção p multiplicada por 4, que é a área do quadrado de lado 2
        PIcalc = 4*Nin/Nt
        sigma = abs(PINp - PIcalc)

    print("O número π foi calculado com precisão de %f%% e tem valor de %f" %(sigma*100, PIcalc))
    print("π_calc = ", PIcalc)
    print("π_numpy =", PINp)
    print("Foram sorteados n =", Nt ,"pontos e" , Nin, "deles estavam contidos no círculo")
    print()

main()

# FINAL