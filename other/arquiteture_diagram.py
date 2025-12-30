import streamlit as st
from graphviz import Digraph


def get_diagrama():
    # Criar o objeto do gráfico
    dot = Digraph(comment="Arquitetura do Sistema de Fake News")

    # --- Configurações Visuais (Idênticas ao teu script original) ---
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")

    # 1. Entrada
    dot.node(
        "Input", "Input\n(Notícia: Título + Corpo)", fillcolor="#E0F7FA", shape="note"
    )

    # 2. Pré-processamento
    dot.node(
        "Pre", "Pré-processamento Geral\n(Limpeza, Tokenização)", fillcolor="#F5F5F5"
    )

    # 3. Camada dos Modelos Especialistas (Level 1)
    with dot.subgraph(name="cluster_level1") as c:
        c.attr(
            label="Modelos Especialistas (Extração de Features)",
            style="dashed",
            color="grey",
        )
        c.attr("node", fillcolor="#FFF9C4")  # Amarelo claro

        c.node("M1", "M1: Classificação\nde Tópicos\n(All The News)")
        c.node("M2", "M2: Deteção de\nAnomalias\n(ISOT - Reuters)")
        c.node("M3", "M3: Stance\nDetection\n(FNC-1)")
        c.node("M4", "M4: Deteção de\nClickbait\n(Clickbait Dataset)")

    # 54. Meta-Classificador
    dot.node(
        "Meta",
        "Meta-Classificador\n(Modelo Final - FakeNewsNet)",
        shape="box3d",
        fillcolor="#B2EBF2",
    )

    # 5. Saída
    dot.node(
        "Output",
        "Classificação Final:\nReal vs Fake",
        shape="ellipse",
        fillcolor="#C8E6C9",
        style="filled,bold",
    )

    # --- Ligações ---
    dot.edge("Input", "Pre")

    dot.edge("Pre", "M1")
    dot.edge("Pre", "M2")
    dot.edge("Pre", "M3")
    dot.edge("Pre", "M4")

    dot.edge("M1", "Meta", label="Tópico")
    dot.edge("M2", "Meta", label="Score Anomalia")
    dot.edge("M3", "Meta", label="Stance")
    dot.edge("M4", "Meta", label="Score Clickbait")

    dot.edge("Meta", "Output")

    return dot


# --- Interface Streamlit ---
st.title("Arquitetura Detalhada do Sistema")

# Chamamos a função e passamos o resultado para o componente do Streamlit
grafico = get_diagrama()
try:
    grafico.render("arquitetura_fake_news", format="png", cleanup=True)
    print("Imagem gerada: arquitetura_fake_news.png")
except Exception as e:
    print("Erro ao gerar ficheiro (provavelmente falta o Graphviz no PATH).")
    print("Sugestão: Corre no Streamlit e tira um Print Screen da imagem gerada.")

st.graphviz_chart(grafico)
