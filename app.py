import streamlit as st
import pandas as pd
import plotly.express as px
from dados import data_set
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np 

# Configuração da página
st.set_page_config(page_title="Análise de Jogos Steam", layout="centered")

# Título e descrição
st.title("🎮 Análise de Jogos Steam - Dashboard Interativo")
st.markdown("""
Explore estatísticas e tendências dos jogos na plataforma Steam.
Cada gráfico possui seus próprios filtros para análise detalhada.
""")

# ======================================================================
# GRÁFICO 1: Quantidade de Jogos por Ano (Indie vs Não-Indie)
# ======================================================================
st.header("📊 Quantidade de Jogos por Ano")

# Adicionar coluna 'indie' baseada nas tags
data_set['indie'] = data_set['tags'].apply(lambda tags: 'Indie' in tags)

# Range slider para seleção de anos (acima do gráfico 1)
min_year = 2000
max_year = 2024
default_start = 2013
default_end = 2023

selected_years = st.slider(
    "Selecione o intervalo de anos para o Gráfico 1:",
    min_value=min_year,
    max_value=max_year,
    value=(default_start, default_end),
    step=1,
    key="year_slider_1"
)

# Filtrar dados
games_by_year = data_set[(data_set['ano_lancamento'] >= selected_years[0]) & 
                         (data_set['ano_lancamento'] <= selected_years[1])]
games_by_year = games_by_year.groupby(['ano_lancamento', 'indie']).size().reset_index(name='quantidade')

# Criar gráfico
fig1 = px.bar(
    games_by_year,
    x='ano_lancamento',
    y='quantidade',
    color='indie',
    color_discrete_map={True: '#FF7F0E', False: '#1F77B4'},
    labels={'ano_lancamento': 'Ano de Lançamento', 'quantidade': 'Quantidade de Jogos', 'indie': 'Jogo Indie'},
    template='plotly_dark',
    barmode='group',
    height=400
)

# Personalizar layout
fig1.update_layout(
    hovermode="x unified",
    legend_title_text='Tipo de Jogo',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(tickmode='linear'),
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig1, use_container_width=True)

# ======================================================================
# GRÁFICO 2: Popularidade de Tags por Ano com Multi-Seleção
# ======================================================================
st.header("📈 Popularidade de Tags")

# Range slider para seleção de anos
selected_years_2 = st.slider(
    "Selecione o intervalo de anos para o Gráfico 2:",
    min_value=min_year,
    max_value=max_year,
    value=(default_start, default_end),
    step=1,
    key="year_slider_2"
)

# Extrair todas as tags únicas
all_tags = list(set(tag for sublist in data_set['tags'] for tag in sublist))

# Widget de seleção múltipla com opção de remoção
selected_tags = st.multiselect(
    "Selecione uma ou mais Tags:",
    sorted(all_tags),
    default=["Action", "Adventure"],
    key="tag_selector"
)

# Filtrar e processar dados
if selected_tags:
    tag_data = []
    for year in range(selected_years_2[0], selected_years_2[1] + 1):
        yearly_games = data_set[data_set['ano_lancamento'] == year]
        total_games = len(yearly_games)
        if total_games > 0:
            for tag in selected_tags:
                tag_count = sum(tag in tags for tags in yearly_games['tags'])
                tag_percentage = (tag_count / total_games) * 100
                tag_data.append({'Ano': year, 'Tag': tag, 'Popularidade (%)': tag_percentage})
    
    tag_df = pd.DataFrame(tag_data)
    
    # Criar gráfico
    fig2 = px.line(
        tag_df,
        x='Ano',
        y='Popularidade (%)',
        color='Tag',
        markers=True,
        labels={'Popularidade (%)': 'Popularidade da Tag (%)'},
        template='plotly_dark',
        height=400
    )
    
    # Personalizar layout
    fig2.update_layout(
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickmode='linear'),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# ======================================================================
# GRÁFICO 3: Jogos Mais Bem Avaliados Não-Indie (Gráfico de Pizza)
# ======================================================================
st.header("🏆 Ranking de Jogos Mais Bem Avaliados (Não-Indie)")

# Filtrar jogos não indie (que não têm a tag 'Indie')
non_indie_games = data_set[~data_set['tags'].apply(lambda tags: 'Indie' in tags)]

# Range slider para seleção de anos
selected_years_3 = st.slider(
    "Selecione o intervalo de anos para o Gráfico 3:",
    min_value=min_year,
    max_value=max_year,
    value=(default_start, default_end),
    step=1,
    key="year_slider_3"
)

# Filtros de avaliação e quantidade
col1, col2 = st.columns(2)
with col1:
    min_rating = st.slider(
        "Avaliação Mínima (%):",
        min_value=0,
        max_value=100,
        value=75,
        step=1,
        key="rating_slider"
    )
with col2:
    num_games = st.slider(
        "Quantidade de Jogos no Ranking:",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="num_games_slider"
    )

# Criar score combinado (porcentagem * quantidade de avaliações)
non_indie_games['score'] = non_indie_games['porcentagem_final'] * non_indie_games['avaliacao_total_final']

# Filtrar por ano e avaliação
filtered_games = non_indie_games[
    (non_indie_games['ano_lancamento'] >= selected_years_3[0]) &
    (non_indie_games['ano_lancamento'] <= selected_years_3[1]) &
    (non_indie_games['porcentagem_final'] >= min_rating)
]

# Pegar os top jogos mais bem avaliados (considerando avaliação e quantidade)
top_games = filtered_games.nlargest(num_games, 'score')

if not top_games.empty:
    # Adicionar ranking
    top_games['Ranking'] = range(1, len(top_games) + 1)
    
    # Criar gráfico de pizza
    fig3 = px.pie(
        top_games,
        names='titulo',
        values='score',
        template='plotly_dark',
        hole=0.3,
        height=500
    )
    
    # Personalizar layout
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,  # Ajuste para posicionar a legenda abaixo do gráfico
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=80, b=100),  # Ajuste de margens para o título
        title={
            'text': f"Top {num_games} Jogos com Avaliação ≥ {min_rating}%",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Mostrar tabela com detalhes
    st.subheader("Detalhes do Ranking")
    display_cols = ['titulo', 'ano_lancamento', 'porcentagem_final', 'avaliacao_total_final', 'preco_dolar_desconto', 'preco_real_desconto']
    st.dataframe(
        top_games[display_cols].rename(columns={
            'titulo': 'Jogo',
            'ano_lancamento': 'Ano',
            'porcentagem_final': 'Avaliação (%)',
            'avaliacao_total_final': 'Total Avaliações',
            'preco_dolar_desconto': 'Preço (US$)',
            'preco_real_desconto': 'Preço (R$)'
        }).style.format({
            'Preço (US$)': '${:.2f}',
            'Preço (R$)': 'R$ {:.2f}'
        }),
        height=min(300, 50 + num_games * 35),
        use_container_width=True
    )
else:
    st.warning("Nenhum jogo encontrado com os critérios selecionados.")

# ======================================================================
# GRÁFICO 4: Busca Detalhada de Jogos
# ======================================================================
st.header("🔍 Busca Detalhada de Jogos")

# Campo de busca
search_term = st.text_input("Digite termos para buscar (nome, tag, descrição, etc.):", key="search_input")

# Filtros adicionais
col1, col2, col3 = st.columns(3)
with col1:
    min_price = st.number_input("Preço Mínimo (US$):", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
with col2:
    max_price = st.number_input("Preço Máximo (US$):", min_value=0.0, max_value=200.0, value=100.0, step=5.0)
with col3:
    min_rating_search = st.slider("Avaliação Mínima (%):", 0, 100, 70, key="search_rating")

# Filtrar resultados
if search_term:
    search_term = search_term.lower()
    results = data_set[
        data_set.apply(lambda row: 
            (search_term in str(row['titulo']).lower()) |
            (search_term in str(row['descricao']).lower()) |
            (any(search_term in tag.lower() for tag in row['tags'])) |
            (str(search_term) in str(row['preco_dolar_desconto'])) |
            (str(search_term) in str(row['ano_lancamento'])),
        axis=1
    )]
    
    # Aplicar filtros adicionais
    results = results[
        (results['preco_dolar_desconto'] >= min_price) &
        (results['preco_dolar_desconto'] <= max_price) &
        (results['porcentagem_final'] >= min_rating_search)
    ]
    
    if not results.empty:
        st.success(f"🎮 {len(results)} jogos encontrados!")
        
        # Mostrar cards para cada jogo
        container = st.container()
        with container:
            for idx, row in results.iterrows():
                with st.expander(f"### {row['titulo']} ({row['ano_lancamento']}) ⭐ {row['porcentagem_final']}%", expanded=False):
                    st.markdown(f"**{row['descricao'][:200]}...**")
                    
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.markdown(f"**Tags:** {', '.join(row['tags'][:5])}")
                        st.markdown(f"**Desenvolvedora:** {', '.join(row['desenvolvedora'])}")
                    with cols[1]:
                        st.markdown("**Preços:**")
                        st.markdown(f"- Original: ${row['preco_dolar_lancamento']:.2f} | R$ {row['preco_real_lancamento']:.2f}")
                        st.markdown(f"- Desconto: ${row['preco_dolar_desconto']:.2f} | R$ {row['preco_real_desconto']:.2f}")
                    with cols[2]:
                        st.markdown("**Avaliações:**")
                        st.markdown(f"- Positivas: {row['porcentagem_final']}%")
                        st.markdown(f"- Total: {row['avaliacao_total_final']}")
                    
                    if row['link_do_jogo']:
                        st.markdown(f"[🔗 Acessar na Steam]({row['link_do_jogo']})")
    else:
        st.warning("Nenhum jogo encontrado com os critérios de busca.")

# ======================================================================
# GRÁFICO 5: Comparação de Preços em Dólar e Real
# ======================================================================
st.header("💰 Comparação de Preços")

# Filtro de preço e quantidade
col1, col2 = st.columns(2)
with col1:
    price_range = st.slider(
        "Selecione a faixa de preço (US$):",
        min_value=0,
        max_value=200,
        value=(0, 100),
        step=5,
        key="price_filter"
    )
with col2:
    num_prices = st.slider(
        "Quantidade de Jogos por Gráfico:",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        key="num_prices"
    )

# Range slider para seleção de anos
selected_years_5 = st.slider(
    "Selecione o intervalo de anos para o Gráfico 5:",
    min_value=min_year,
    max_value=max_year,
    value=(default_start, default_end),
    step=1,
    key="year_slider_5"
)

# Filtrar jogos dentro da faixa de preço
filtered_prices = data_set[
    (data_set['preco_dolar_desconto'] >= price_range[0]) &
    (data_set['preco_dolar_desconto'] <= price_range[1]) &
    (data_set['ano_lancamento'] >= selected_years_5[0]) &
    (data_set['ano_lancamento'] <= selected_years_5[1])
].nlargest(num_prices, 'preco_dolar_desconto')

# Gráfico de preços em dólar
st.subheader(f"Top {len(filtered_prices)} Jogos por Preço (US$)")
fig_dolar = px.bar(
    filtered_prices,
    y='titulo',
    x='preco_dolar_desconto',
    orientation='h',
    labels={'preco_dolar_desconto': 'Preço (US$)', 'titulo': ''},
    color='preco_dolar_desconto',
    color_continuous_scale='Bluered',
    template='plotly_dark',
    height=400
)
fig_dolar.update_layout(
    xaxis_range=[0, 200],
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    margin=dict(l=20, r=20, t=50, b=20),
    yaxis={'categoryorder':'total ascending'}
)
st.plotly_chart(fig_dolar, use_container_width=True)

# Gráfico de preços em real
st.subheader(f"Top {len(filtered_prices)} Jogos por Preço (R$)")
fig_real = px.bar(
    filtered_prices,
    y='titulo',
    x='preco_real_desconto',
    orientation='h',
    labels={'preco_real_desconto': 'Preço (R$)', 'titulo': ''},
    color='preco_real_desconto',
    color_continuous_scale='Tealrose',
    template='plotly_dark',
    height=400
)
fig_real.update_layout(
    xaxis_range=[0, 1000],
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    margin=dict(l=20, r=20, t=50, b=20),
    yaxis={'categoryorder':'total ascending'}
)               
st.plotly_chart(fig_real, use_container_width=True)

# ======================================================================
# COMPONENTE DE MACHINE LEARNING (ÁRVORE DE DECISÃO)
# ======================================================================
st.header("🤖 Machine Learning - Árvore de Decisão")

# Filtrar jogos com preços razoáveis
df_filtered = data_set[
    (data_set['preco_dolar_desconto'] <= 50) & 
    (data_set['preco_dolar_lancamento'] <= 200)
].copy()

if len(df_filtered) < 50:
    st.warning("⚠️ Poucos jogos após filtragem de preço. Ajustando critérios...")
    df_filtered = data_set[
        (data_set['preco_dolar_desconto'] <= 100) & 
        (data_set['preco_dolar_lancamento'] <= 250)
    ].copy()

st.subheader("🔮 Prever se o Jogo terá Avaliação ≥75%")

# Adicionar novas features
df_filtered['qtd_linguagens'] = df_filtered['linguagens_suportadas'].apply(len)
df_filtered['qtd_funcionalidades'] = df_filtered['funcionalidades'].apply(len)

# Definir target (1 se avaliação >= 75%, senão 0)
df_filtered['avaliacao_positiva'] = (df_filtered['porcentagem_final'] >= 75).astype(int)

# Codificar variáveis categóricas
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_filtered['sistema_operacional_encoded'] = le.fit_transform(df_filtered['sistema_operacional'].astype(str))
df_filtered['tipo_armazenamento_encoded'] = le.fit_transform(df_filtered['tipo_armazenamento'].astype(str))

# Seleção de features
features = st.multiselect(
    "Selecione as features para previsão:",
    options=['preco_dolar_desconto', 'ano_lancamento', 'memoria_ram', 
            'sistema_operacional_encoded', 'tipo_armazenamento_encoded',
            'qtd_linguagens', 'qtd_funcionalidades'],
    default=['preco_dolar_desconto', 'ano_lancamento', 'memoria_ram', 'qtd_linguagens'],
    key="features_decision_tree"
)

if features and len(features) >= 2:
    X = df_filtered[features].fillna(0)
    y = df_filtered['avaliacao_positiva']
    
    # Treinar modelo
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt
    
    model = DecisionTreeClassifier(max_depth=4, random_state=42)  # Aumentei a profundidade para 4
    model.fit(X, y)
    
    # Plotar árvore
    st.write("### Estrutura da Árvore de Decisão")
    fig, ax = plt.subplots(figsize=(20, 12))  # Aumentei o tamanho da figura
    plot_tree(model, feature_names=features, class_names=["<75%", "≥75%"], 
             filled=True, rounded=True, ax=ax, fontsize=10, proportion=True)
    st.pyplot(fig)
    
    # Exibir importância das features
    st.write("### Importância das Features")
    importance = pd.DataFrame({
        'Feature': features,
        'Importância': model.feature_importances_
    }).sort_values('Importância', ascending=False)
    
    # Gráfico de importância melhorado
    fig_importance = px.bar(importance, x='Feature', y='Importância', 
                           color='Importância', text_auto='.2f',
                           title='Importância Relativa das Features',
                           color_continuous_scale='Bluered')
    fig_importance.update_layout(showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Estatísticas descritivas
    st.write("### Estatísticas das Features Selecionadas")
    st.dataframe(df_filtered[features].describe().T[['mean', 'std', 'min', 'max']])
    
    # Correlação com o target
    st.write("### Correlação com Avaliação Positiva (≥75%)")
    correlation = df_filtered[features + ['avaliacao_positiva']].corr()['avaliacao_positiva'].drop('avaliacao_positiva')
    st.bar_chart(correlation)
    
else:
    st.warning("Selecione pelo menos 2 features para construir a árvore de decisão.")
    
    
# ======================================================================
# COMPONENTE DE MACHINE LEARNING (K-Means Clustering) - Versão Simplificada
# ======================================================================
st.header("🔍 Agrupamento de Jogos com K-Means")

st.markdown("""
Agrupe jogos similares baseados em duas características. 
Jogos no mesmo grupo têm valores parecidos para as características selecionadas.
""")

# Selecionar features para clustering
col1, col2 = st.columns(2)
with col1:
    feature_x = st.selectbox(
        "Selecione a primeira característica:",
        options=['preco_dolar_desconto', 'ano_lancamento', 'porcentagem_final', 
                'avaliacao_total_final', 'memoria_ram'],
        index=0,
        key="feature_x"
    )
with col2:
    feature_y = st.selectbox(
        "Selecione a segunda característica:",
        options=['preco_dolar_desconto', 'ano_lancamento', 'porcentagem_final', 
                'avaliacao_total_final', 'memoria_ram'],
        index=2,
        key="feature_y"
    )

if feature_x != feature_y:
    # Preparar dados
    df_cluster = data_set[[feature_x, feature_y, 'titulo']].dropna()
    
    # Selecionar número de clusters
    n_clusters = st.slider(
        "Número de grupos:",
        min_value=2,
        max_value=5,
        value=3,
        key="n_clusters"
    )
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[[feature_x, feature_y]])
    
    # Executar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_cluster['Grupo'] = 'Grupo ' + (clusters + 1).astype(str)
    
        # Visualização 2D
    st.subheader(f"Agrupamento de Jogos ({n_clusters} grupos)")
    fig = px.scatter(
        df_cluster,
        x=feature_x,
        y=feature_y,
        color='Grupo',
        hover_name='titulo',
        labels={
            feature_x: feature_x.replace('_', ' ').title(),
            feature_y: feature_y.replace('_', ' ').title()
        },
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    fig.update_xaxes(range=[0, 100])  # ADICIONE ESTA LINHA PARA LIMITAR O EIXO X
    st.plotly_chart(fig, use_container_width=True)
    
    # Explicação dos grupos
    st.subheader("📌 Explicação dos Grupos")
    
    # Calcular médias por grupo
    group_stats = df_cluster.groupby('Grupo')[[feature_x, feature_y]].mean()
    
    for group in sorted(df_cluster['Grupo'].unique()):
        group_data = df_cluster[df_cluster['Grupo'] == group]
        sample_games = group_data.sample(min(3, len(group_data)))['titulo'].tolist()
        
        with st.expander(f"{group} - {len(group_data)} jogos"):
            st.write(f"**Características médias:**")
            st.write(f"- {feature_x.replace('_', ' ')}: {group_stats.loc[group, feature_x]:.1f}")
            st.write(f"- {feature_y.replace('_', ' ')}: {group_stats.loc[group, feature_y]:.1f}")
            
            st.write("**Exemplos de jogos:**")
            st.write(", ".join(sample_games))
    
    st.info("💡 Passe o mouse sobre os pontos para ver os nomes dos jogos")
    
else:
    st.warning("Selecione duas características diferentes para o agrupamento.")