from pyexpat.errors import messages
from typing import Annotated
from typing_extensions import TypedDict
from operator import or_
import random

from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from models import models

# Estados
class State(TypedDict):
    detective: str
    crime: str
    location: str
    clue: str
    story: Annotated[dict[str, str], or_]

# Nodes
def first_act(state: State):
    """Escreve o primeiro ato"""
    detective = random.choice([
        "Um detetive excêntrico com memória fotográfica",
        "Um investigador que resolve casos através de análises de padrões",
        "Um detetive aposentado que não consegue resistir a um bom mistério"
    ])
    crime = random.choice([
        "O desaparecimento de um objeto valioso",
        "Uma mensagem codificada deixada na cena do crime",
        "Um assassinato aparentemente impossível"
    ])
    location = random.choice([
        "Uma mansão vitoriana isolada"
        "Um museu de antiguidades",
        "Um clube exclusivo de alta sociedade"
    ])
    clue = random.choice([
        "Uma marca de cigarro rara",
        "Um relógio parado em um horário específico",
        "Uma carta de baralho manchada"
    ])
    msg = f"""
    Você é um escritor de mistério experiente
    encarregado de escrever o Primeiro Ato de um história de detetive.
    
    **Instruções:**
    
    Baseado nas seguintes informações iniciais:
    *  **Detetive:** {detective}
    *  **Crime/Mistério:** {crime}
    *  **Local:** {location}
    *  **Pista Inicial:** {clue}
    
    Escreva o **Primeiro Ato** desta história. Este ato deve:
    1. **Apresentar o Detetive: ** Mostre quem é o detetive e suas caracteristicas unicas.
    2. **Estabelecer o Contexto:** Descreva o local e a situação antes do crime ser descoberto
    3. **Introduzir o Mistério:** O momento em que o detetive é chamado para investigar {crime}.
    4. **Terminar com o Primeiro Ponto de Virada:** 0 detetive encontra {clue} e decide aceitar o caso.
    
    O primeiro Ato de ter entre 1 e 2 parágrafos.
    Ao final, sinalize claramente:
    --- FIM DO PRIMEIRO ATO ---
    --- PRIMEIRO PONTO DE VIRADA: [Descreva brevemente o ponto de virada] ---
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4o"].invoke(messages)

    return {
        "story": {"act_1": response.content},
        "detective": detective,
        "crime": crime,
        "location": location,
        "clue": clue
    }

def second_act(state: State):
    """Escreve o Segundo Ato"""
    msg = f"""
    Você é um escritor de mistério continuando uma história de detetive.
        Abaixo está o Primeiro Ato da história:
        --- INÍCIO DO PRIMEIRO ATO ---
        {state["story"]["act_1"]}
        --- FIM DO PRIMEIRO ATO ---
    
        **Instruções:**
        
        Agora, escreve o **Segundo Ato** desta história, continuando
        diretemanete de onde o Primeiro Ato parou. Este ato deve ter:
        1. **Investigações Iniciais:** Como o detetive começa a coletar evidências e entrevistar suspeitos.
        2. **Novas Pistas:** Descobertas que parecem levar em direções diferentes.
        3. **Complicações:** Suspeitos que mentem, pistas que se contradizem.
        4. **Ponto Médio:** Uma revelação surpreendente que muda a direção da investigação.
        
        O Segundo Ato deve ter entre 1 e 2 parágrafos. Ao final, sinalize claramente:
        --- FIM DO SEGUNDO ATO ---
        --- PONTO MÉDIO: [Descreva brevemente o ponto médio] ---
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4o"].invoke(messages)

    return {
        "story": {"act_2": response.content}
    }

def third_act(state: State):
    """Escreve o Terceiro Ato"""
    msg = f"""
    Você é um escritor de mistério continuando uma história de detetive.
        Abaixo está o Primeiro e o Segundo Ato da história:
        --- INÍCIO DO CONTEXTO PREEXISTENTE ---'
        {state["story"]["act_1"]}
        
        {state["story"]["act_2"]}
        --- FIM DO CONTEXTO PREEXISTENTE ---

        **Instruções:**

        Agora, escreve o **Terceiro Ato** desta história, continuando
        diretemanete de onde o Segundo Ato parou. Este ato deve ter:
        1. **Crise:** O momento em que todas as teorias parecem estar erradas.
        2. **Revelação:** Uma conexão inesperada entre as pistas.
        3. **Decisão Final:** O detetive percebe a verdade e se prepara para o confronto.

        O Terceiro Ato deve ter entre 1 e 2 parágrafos. Ao final, sinalize claramente:
        --- FIM DO TERCEIRO ATO ---
        --- SEGUNDO PONTO DE VIRADA: [Descreva brevemente o segundo ponto virada] ---
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4o"].invoke(messages)

    return {
        "story": {"act_3": response.content}
    }

def fourth_act(state: State):
    """Escreve o Quarto Ato"""
    msg = f"""
    Você é um escritor de mistério continuando uma história de detetive.
        Abaixo estão os primeiros atos da história:
        --- INÍCIO DO CONTEXTO PREEXISTENTE ---'
        {state["story"]["act_1"]}

        {state["story"]["act_2"]}
        
        {state["story"]["act_3"]}
        --- FIM DO CONTEXTO PREEXISTENTE ---

        **Instruções:**

        Agora, escreve o **Quarto Ato** desta história, continuando
        diretemanete de onde o Terceiro Ato parou. Este ato deve ter:
        1. **Clímax:** O momento em que o detetive revela a solução do mistério.
        2. **Explicação:** Como todas as pistas se encaixam na solução final.
        3. **Conclusão:** O impacto da resolução do caso e o que o detetive aprendeu.

        O Quarto Ato deve ter entre 1 e 2 parágrafos. Ao final, sinalize claramente:
        --- FIM DO QUARTO ATO ---
        --- CLÍMAX: [Texto do Clímax] ---
        --- RESOLUÇÃO: [Texto do Resolução] ---
        --- FIM DA HISTÓRIA ---
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4o"].invoke(messages)

    return {
        "story": {"act_4": response.content}
    }

# contruindo o workflow
detective_builder = StateGraph(State)

detective_builder.add_node("first_act", first_act)
detective_builder.add_node("second_act", second_act)
detective_builder.add_node("third_act", third_act)
detective_builder.add_node("fourth_act", fourth_act)

detective_builder.add_edge(START, "first_act")
detective_builder.add_edge("first_act", "second_act")
detective_builder.add_edge("second_act", "third_act" )
detective_builder.add_edge("third_act", "fourth_act")
detective_builder.add_edge("fourth_act", END)

detective_workflow = detective_builder.compile()
