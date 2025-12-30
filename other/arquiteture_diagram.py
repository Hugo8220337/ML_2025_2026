from graphviz import Digraph


def gerar_diagrama():
    # Criar o objeto do gráfico
    dot = Digraph(comment="Arquitetura do Sistema de Fake News")

    # Configurações globais para ficar com aspeto académico/limpo
    dot.attr(rankdir="TB")  # TB = Top to Bottom (De cima para baixo)
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
    # Usamos um 'subgraph' para os alinhar horizontalmente
    with dot.subgraph(name="cluster_level1") as c:
        c.attr(
            label="Nível 1: Modelos Especialistas (Extração de Features)",
            style="dashed",
            color="grey",
        )
        c.attr("node", fillcolor="#FFF9C4")  # Amarelo claro

        c.node("M1", "M1: Classificação\nde Tópicos\n(All The News)")
        c.node("M2", "M2: Deteção de\nAnomalias\n(ISOT - Reuters)")
        c.node("M3", "M3: Stance\nDetection\n(FNC-1)")
        c.node("M4", "M4: Deteção de\nClickbait\n(Clickbait Dataset)")

    # 4. Vetor de Features (Opcional, mas ajuda a explicar)
    dot.node(
        "Features",
        "Vetor de Meta-Features\n(Probabilidades M1, Score M2, Classe M3, Score M4)",
        shape="parallelogram",
        fillcolor="#E1BEE7",
    )

    # 5. Meta-Classificador (Level 2)
    dot.node(
        "Meta",
        "Meta-Classificador\n(Modelo Final - FakeNewsNet)",
        shape="box3d",
        fillcolor="#B2EBF2",
    )

    # 6. Saída
    dot.node(
        "Output",
        "Classificação Final:\nReal vs Fake",
        shape="ellipse",
        fillcolor="#C8E6C9",
        style="filled,bold",
    )

    # --- Definição das Ligações (Edges) ---
    dot.edge("Input", "Pre")

    # Do pré-processamento para cada modelo
    dot.edge("Pre", "M1")
    dot.edge("Pre", "M2")
    dot.edge("Pre", "M3")
    dot.edge("Pre", "M4")

    # Dos modelos para o vetor de features (ou direto para o meta)
    dot.edge("M1", "Features", label="Tópico")
    dot.edge("M2", "Features", label="Score Anomalia")
    dot.edge("M3", "Features", label="Stance")
    dot.edge("M4", "Features", label="Score Clickbait")

    # Do vetor para o meta e saída
    dot.edge("Features", "Meta")
    dot.edge("Meta", "Output")

    # Renderizar o gráfico para um ficheiro
    # Isto cria um ficheiro 'arquitetura_fake_news.png' na mesma pasta
    dot.render("arquitetura_fake_news", view=True, format="png")
    print("Diagrama gerado com sucesso!")


if __name__ == "__main__":
    gerar_diagrama()
