'''
EP 2 - MAP2212 - 2024
    Victor Rocha Cardoso Cruz
    NUSP: 11223757

    ENUNCIADO:
    Neste EP, você deverá utilizar as quatro variantes do método de Monte Carlo estudadas para estimar a integral definida de uma função real em um intervalo.
    - Encontre, em seu a ambiente computacional, funções para gerar amostras de variáveis aleatórias com distribuições Uniforme, Beta, Gamma e Weibull.
    - Implemente as quatro variantes de integração por Monte Carlo estudadas para integrar a função f(x)=exp(−ax)cos(bx) em [0,1], onde a=0.RG, b=0.CPF, e RG
        e CPF correspondem aos dígitos de seus documentos de identificação.
    -Escolha parâmetros para cada distribuição amostral via inspeção visual ou outro método que quiser. Escolha uma função polinomial para a variável de controle. 
        Escolha n para obter um erro relativo |γ−γ^|/γ<0.0005 (sem conheçer γ!).
'''
import numpy as np
from scipy.stats import uniform, beta, gamma, weibull_min

# Definição dos parâmetros a e b
a = 0.564263989  # RG
b = 0.51197762809  # CPF

# Função alvo f(x)
def f(x):
    fx = np.exp(-a * x) * np.cos(b * x)
    return fx

print("*** EP 2 INICIANDO ***")
print()
print("*** EXPERIMENTO PILOTO ***")
# EXPERIMENTO PILOTO #
# Método Cru para definição da integral esperada
# Definição a largura da distribuição
n = 1000000
uniform_samples_x = uniform.rvs(size=n, loc=0, scale=1)
uniform_samples_y = f(uniform_samples_x)
# Aproximação através de Distribuição Uniforme 
AreaPilot = np.mean(uniform_samples_y)
print(f"Área Piloto {AreaPilot}")
print()

#Definição da margem de erro e quantil da normal
epsilon = 0.05/100 * AreaPilot
z_alfa = 1.64

# ********************* MÉTODO DE MONTE CARLO CRU ********************* #
print("*** MÉTODO DE MONTE CARLO CRU ***")
# Distribuição Uniforme
n = 100000
uniform_samples_x = uniform.rvs(size=n, loc=0, scale=1)
uniform_samples_y = f(uniform_samples_x)
# Aproximação através de Distribuição Uniforme 
area_cru = np.mean(uniform_samples_y)
variancia = np.sum((uniform_samples_y - area_cru)**2)/n
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Área estimada: {area_cru}")
print(f"Variância do estimador: {variancia}")
print(f"Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Largura da Distr.: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()

# ******************************************************************************* #
# ********************* MÉTODO DE MONTE CARLO 'HIT OR MISS' ********************* #
print("*** MÉTODO DE MONTE CARLO 'HIT OR MISS' ***")
# Aproximação através de Distribuição Uniforme 
n = 100000
PontosDentro = 0
for i in range(n): 
    xi = uniform.rvs()
    yi = uniform.rvs()
    # Checa se o ponto sorteado está abaixo da curva
    if yi <= f(xi):
        PontosDentro += 1
area_hitmiss = PontosDentro/n
# Variancia do estimador
variancia = area_hitmiss*(1 - area_hitmiss)/n
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Área estimada: {area_hitmiss}")
print(f"Variância do estimador: {variancia:.10f}")
print(f"Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Largura da Distr.: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()

# ******************************************************************************************** #
# ********************* MÉTODO DE MONTE CARLO AMOSTRAGEM POR IMPORTÂNCIA ********************* #
print("*** MÉTODO DE MONTE CARLO AMOSTRAGEM POR IMPORTÂNCIA ***")
# Distribuição Uniforme
n = 100000
pesos = []
pdfx = []
for i in range(n):
    xi = uniform.rvs()
    # Função Densidade de probabilidade da Distribuição Uniforme
    pdf_value = uniform.pdf(xi)
    pdfx.append(pdf_value)
    yi = f(xi)
    # Calculo dos pesos, como o quociente entre os valores f(xi) da função pela FDP da distribuição uniforme
    peso = yi/pdf_value
    pesos.append(peso)
    i+=1
area_uniform = np.mean(np.array(pesos))
# Variancia do estimador
variancia = np.sum(np.array(pdfx)*(np.array(pesos) - area_uniform)**2)/n
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Uniforme: Área estimada: {area_uniform}")
print(f"Uniforme: Variância do estimador: {variancia:.10f}")
print(f"Uniforme: Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Uniforme: Quantidade de iterações: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()

# Distribuição Beta
# Calculo dos pesos, como o quociente entre os valores f(xi) da função pela FDP da distribuição Beta
pesos = []
pdfx = []
n = 100000
for i in range(n):
    xi = beta.rvs(size=1, a = 1, b = 1)
    # Obtenção da função densidade de probabilidade, para a distribuição Beta
    pdf_value = beta.pdf(xi, a = 1, b = 1)
    pdfx.append(pdf_value)
    yi = f(xi)
    peso = yi/pdf_value
    pesos.append(peso)
    i+=1
area_beta = np.mean(pesos)
# Variancia do estimador
variancia = np.sum(np.array(pdfx)*(np.array(pesos) - area_beta)**2)/n
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Beta: Área estimada: {area_beta}")
print(f"Beta: Variância do estimador: {variancia:.10f}")
print(f"Beta: Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Beta: Quantidade de iterações: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()

# Distribuição Gamma
# Normalização da distribuição Gamma a partir da função cumulativa nos pontos de interesse
cdf_lower = gamma.cdf(0, a=1, scale=2)
cdf_upper = gamma.cdf(1, a=1, scale=2) 
const_normalizacao = 1 / (cdf_upper - cdf_lower)
# Função criada para obter apenas valores entre 0 e 1
def generate_n_gamma_samples(num_samples):
    samples_no_intervalo = []
    while len(samples_no_intervalo) < num_samples:
        # Gerar amostras aleatórias pela distribuição Gama
        gamma_samples_x = gamma.rvs(a=1, scale=2)
        # Checa se a amostra está dentro do intervalo
        if 0 <= gamma_samples_x <= 1:
            samples_no_intervalo.append(gamma_samples_x)
    return np.array(samples_no_intervalo)
pesos = []
pdfx = []
n = 100000
for i in range(n):
    xi = generate_n_gamma_samples(1)
    # Obtenção da função densidade de probabilidade, para a distribuição Gamma, e normalizada à 0 e 1
    pdf_value = gamma.pdf(xi, a = 1, scale = 2)
    pdf_no_intervalo = pdf_value * const_normalizacao
    pdfx.append(pdf_no_intervalo)
    yi = f(xi)
    peso = yi/pdf_no_intervalo
    pesos.append(peso)
    i+=1
area_gama = np.mean(pesos)
# Variancia do estimador
variancia = np.sum(np.array(pdfx)*(np.array(pesos) - area_gama)**2)/n
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Gama: Área estimada: {area_gama}")
print(f"Gama: Variância do estimador: {variancia}")
print(f"Gama: Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Gama: Quantidade de iterações: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()

# Distribuição Weibull
# Normalização da distribuição Weibull a partir da função cumulativa nos pontos de interesse
cdf_lower = weibull_min.cdf(0, c=1)
cdf_upper = weibull_min.cdf(1, c=1) 
const_normalizacao = 1 / (cdf_upper - cdf_lower)
# Função criada para obter apenas valores entre 0 e 1
def generate_n_weibull_samples(num_samples):
    samples_no_intervalo = []
    while len(samples_no_intervalo) < num_samples:
        # Gerar amostras aleatórias pela distribuição Weibull
        weibull_samples_x = weibull_min.rvs(c=1)
        # Checa se a amostra está dentro do intervalo
        if 0 <= weibull_samples_x <= 1:
            samples_no_intervalo.append(weibull_samples_x)
    return np.array(samples_no_intervalo)
pesos = []
pdfx = []
n = 100000
for i in range(n):
    xi = generate_n_weibull_samples(1)
    # Obtenção da função densidade de probabilidade, para a distribuição Weibull, e normalizada à 0 e 1
    pdf_value = weibull_min.pdf(xi, c=1)
    pdf_no_intervalo = pdf_value * const_normalizacao
    pdfx.append(pdf_no_intervalo)
    yi = f(xi)
    peso = yi/pdf_no_intervalo
    pesos.append(peso)
    i+=1
area_weibull = np.mean(pesos)
# Variancia do estimador
variancia = np.sum(np.array(pdfx)*(np.array(pesos) - area_weibull)**2)/n
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Weibull: Área estimada: {area_weibull}")
print(f"Weibull: Variância do estimador: {variancia:.10f}")
print(f"Weibull: Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Weibull: Quantidade de iterações: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()

# *************************************************************************************** #
# ********************* MÉTODO DE MONTE CARLO VARIÁVEIS DE CONTROLE ********************* #
print("*** MÉTODO DE MONTE CARLO VARIÁVEIS DE CONTROLE ***")
# Definição dos limites de integração para a função f e φ
limInfer, limSuper= 0, 1
# Função polinominal φ que aproxima-se à curva da função f(x), encontrada a partir dos pontos (1, f(1)) e (0, f(0))
def phi(x):
    phix = ((f(1)-f(0))/(1 - 0))*x + 1
    #phix = 1 - 0.564263989*x + 0.02814*x**(2) + 0.04400*x**(3)- 0.01377*x**(4)
    return phix
# Função primitiva de φ
def phiPrim(x):
    phiPrim = x + ((f(1)-f(0))/(1 - 0))*x**(2)/2
    #phiPrim = x - 0.564263989*x**(2)/2 + 0.02814*x**(3)/3 + 0.04400*x**(4)/4 - 0.01377*x**(5)/5
    return phiPrim

termo = 0
fxn, phixn = [], []
n = 100000
for i in range(n):
    xi = uniform.rvs()
    # Armezando os valores de f(xi) e g(xi) para o cálculo final da variância
    fxn.append(f(xi))
    phixn.append(phi(xi))
    # Termo do somatório para calculo de gama chapeu
    termo += f(xi) - phi(xi) + (phiPrim(limSuper)-phiPrim(limInfer))
    i+=1
area_control = termo/n
var_f = np.var(np.array(fxn))
var_phi = np.var(np.array(phixn))
correlacao = np.cov(fxn, phixn)[0,1]
variancia = (1/n)*(var_f + var_phi - 2*correlacao*np.sqrt(var_f)*np.sqrt(var_phi))
erro_padrao = np.sqrt(variancia)/np.sqrt(n)
print(f"Área estimada:: {area_control}")
print(f"Variância do estimador: {variancia:.10f}")
print(f"Erro padrão do estimador: {erro_padrao:.10f}")
print(f"Quantidade de iterações: {n}")
print()
n = round(((z_alfa * np.sqrt(variancia))/epsilon)**2)
print(f"Valor de n calculado: {n}")
print()