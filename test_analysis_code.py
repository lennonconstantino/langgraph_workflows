from parallelization import code_analysis_workflow

# codigo de exemplo para analise
codigo_teste = """
def calcular_media(lista):
    soma = 0
    for i in range(len(lista)):
        soma = soma + lista[i]
    media = som / len(lista)
    return media
    
# testando a funcao
numeros = [1,2,3,4,5]
resultado = calcular_media(numeros)
print(f'A media eh: {resultado}')
"""

# executando workflow
resultado = code_analysis_workflow.invoke({
    "query": codigo_teste
})

# exibir resultados
print("\n=== Análise do Gemini ===")
print(resultado["llm1"])
print("\n=== Análise do o4 Mini ===")
print(resultado["llm2" ])
print("\n=== Avaliação Final ===")
print(resultado["best_llm" ])
