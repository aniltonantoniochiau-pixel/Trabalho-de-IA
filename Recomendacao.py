import os
import sqlite3
import random
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.sparse import hstack


DB_PATH = "loja.db"
MODEL_PATH = "modelo_recomendacao.joblib"
VEC_PATH = "vectorizer_tfidf.joblib"
CSV_RECOM_PATH = "produtos_recomendados.csv"
ALPHA_DEFAULT = 0.9

def gerar_dados_sinteticos(num_products=500, num_users=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    categorias = {
        "Eletrónica": ["Telemóvel XSmart", "Auriculares BeatSound", "TV UltraHD", "Carregador Turbo", "Portátil NovaBook", "Tablet SkyTab", "Relógio Digital", "Coluna Bluetooth", "Câmara Vision", "Smartwatch Pro"],
        "Alimentação": ["Arroz Premium", "Feijão Preto", "Leite Natural", "Café Moído Fino", "Açúcar Branco", "Biscoitos Caseiros", "Sumo Tropical", "Óleo Vegetal", "Farinha de Trigo", "Sal Refinado"],
        "Vestuário": ["Camisola Classic", "Calças Jeans", "Ténis Urban", "Casaco Winter", "Boné Street", "Sapatos Confort", "Camisa Formal", "Saia Floral", "Chinelos Relax", "Meias Algodão"],
        "Higiene": ["Sabonete Natural", "Champô Suave", "Pasta de Dentes Fresh", "Creme Hidratante", "Desodorizante Spray", "Escova Dental", "Papel Higiénico", "Gel de Banho", "Toalhitas Húmidas", "Perfume Elegance"],
        "Bebidas": ["Água Mineral", "Refrigerante Cola", "Cerveja Light", "Sumo Natural", "Vinho Tinto", "Chá Verde", "Cappuccino Gelado", "Energético Power", "Limonada Fresh", "Leite com Chocolate"]
    }


    produtos = []
    all_nomes = []
    for cat, nomes in categorias.items():
        all_nomes.extend([f"{nome}" for nome in nomes])

    while len(all_nomes) < num_products:
        nome = random.choice(all_nomes)
        var = random.choice(["Plus", "Max", "Light", "Premium", "Pro", "Nova"])
        all_nomes.append(f"{nome} {var}")

    for i in range(num_products):
        nome = all_nomes[i]
        preco = round(np.random.exponential(scale=20.0) + 2.0, 2)
        base_rating = float(np.clip(2.5 + np.random.randn()*0.6, 1.0, 5.0))
        produtos.append((i+1, nome, preco, base_rating))

    produtos_df = pd.DataFrame(produtos, columns=["id_produto", "nome", "preco", "rating_base"])

    compras = []
    for user in range(1, num_users+1):
        n = np.random.poisson(1.5)
        choices = np.random.choice(produtos_df["id_produto"], size=max(1,n), replace=False)
        for pid in choices:
            quantidade = np.random.randint(1,4)
            compras.append((None, pid, user, quantidade))
    compras_df = pd.DataFrame(compras, columns=["id_compra", "id_produto", "id_user", "quantidade"])

    palavras_positivas = ["bom", "excelente", "recomendo", "ótimo", "fantástico", "maravilhoso", "satisfeito", "qualidade"]
    palavras_negativas = ["ruim", "péssimo", "não_recomendo", "decepcionou", "defeito", "insuportável", "fraco", "barato_demais"]
    palavras_neutras = ["chegou", "embalagem", "entrega", "produto", "uso", "funciona", "garantia"]

    comentarios = []
    for pid, row in produtos_df.set_index("id_produto").iterrows():
        n_com = np.random.randint(3, 25)
        for j in range(n_com):
            rating = float(np.clip(row["rating_base"] + np.random.randn()*0.8, 1.0, 5.0))
            prob_pos = (rating - 1.0) / 4.0
            words = []
            length = np.random.randint(5, 18)
            for _ in range(length):
                r = random.random()
                if r < prob_pos * 0.6:
                    words.append(random.choice(palavras_positivas))
                elif r < 0.5:
                    words.append(random.choice(palavras_negativas))
                else:
                    words.append(random.choice(palavras_neutras))
            texto = " ".join(words)
            comentarios.append((None, pid, round(rating,2), texto))
    comentarios_df = pd.DataFrame(comentarios, columns=["id_comentario", "id_produto", "avaliacao", "texto"])

    return produtos_df, compras_df, comentarios_df

def criar_db(produtos_df, compras_df, comentarios_df):
    conn = sqlite3.connect(DB_PATH)
    produtos_df.to_sql("Produtos", conn, index=False, if_exists="replace")
    compras_df.to_sql("Compras", conn, index=False, if_exists="replace")
    comentarios_df.to_sql("Comentarios", conn, index=False, if_exists="replace")
    conn.close()

def carregar_features():
    conn = sqlite3.connect(DB_PATH)
    produtos_df = pd.read_sql("SELECT * FROM Produtos", conn)
    compras_df = pd.read_sql("SELECT * FROM Compras", conn)
    comentarios_df = pd.read_sql("SELECT * FROM Comentarios", conn)
    conn.close()

    vendas = compras_df.groupby("id_produto")["quantidade"].sum().reset_index().rename(columns={"quantidade":"total_vendido"})
    media_avaliacoes = comentarios_df.groupby("id_produto")["avaliacao"].mean().reset_index().rename(columns={"avaliacao":"avg_rating"})
    docs = comentarios_df.groupby("id_produto")["texto"].apply(lambda x: " ".join(x)).reset_index().rename(columns={"texto":"all_comments"})

    features = produtos_df.merge(vendas, on="id_produto", how="left")
    features = features.merge(media_avaliacoes, on="id_produto", how="left")
    features = features.merge(docs, on="id_produto", how="left")
    features.fillna({"total_vendido":0, "avg_rating":features["rating_base"], "all_comments":""}, inplace=True)
    features["sales_score"] = np.log1p(features["total_vendido"])
    return features

def treinar_e_comparar(features):
    vectorizer = TfidfVectorizer(max_features=800)
    X_text = vectorizer.fit_transform(features["all_comments"])

    num_feats = features[["preco", "avg_rating", "total_vendido"]].copy()
    num_feats["preco_norm"] = (num_feats["preco"] - num_feats["preco"].min()) / (num_feats["preco"].max() - num_feats["preco"].min() + 1e-9)
    num_feats["rating_norm"] = (num_feats["avg_rating"] - 1.0) / 4.0
    num_feats["vendido_norm"] = (num_feats["total_vendido"] - num_feats["total_vendido"].min()) / (num_feats["total_vendido"].max() - num_feats["total_vendido"].min() + 1e-9)
    X_num = num_feats[["preco_norm", "rating_norm", "vendido_norm"]].values

    X = hstack([X_text, X_num])
    y = features["sales_score"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    lr = LinearRegression()
    lr.fit(X_train.toarray(), y_train)
    lr_pred = lr.predict(X_test.toarray())

    rf_r2, rf_mse = r2_score(y_test, rf_pred), mean_squared_error(y_test, rf_pred)
    lr_r2, lr_mse = r2_score(y_test, lr_pred), mean_squared_error(y_test, lr_pred)

    print("\n[RESULTADOS DA COMPARAÇÃO]")
    print(f"RandomForest → R²={rf_r2:.4f}, MSE={rf_mse:.6f}")
    print(f"LinearRegression → R²={lr_r2:.4f}, MSE={lr_mse:.6f}")

    melhor_modelo = rf if rf_r2 > lr_r2 else lr
    joblib.dump(melhor_modelo, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"[INFO] Melhor modelo salvo: {'RandomForest' if melhor_modelo==rf else 'LinearRegression'}")

    # Gráfico de comparação
    plt.figure(figsize=(6,4))
    plt.bar(['Random Forest', 'Linear Regression'], [rf_r2, lr_r2], color=['green','blue'])
    plt.title('Comparação de Desempenho dos Modelos (R²)')
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig('grafico_comparacao_modelos.png')
    plt.close()

    return melhor_modelo, vectorizer

def gerar_recomendacoes(features, model, vectorizer, alpha=ALPHA_DEFAULT, k=10):
    num_feats = features[["preco", "avg_rating", "total_vendido"]].copy()
    num_feats["preco_norm"] = (num_feats["preco"] - num_feats["preco"].min()) / (num_feats["preco"].max() - num_feats["preco"].min() + 1e-9)
    num_feats["rating_norm"] = (num_feats["avg_rating"] - 1.0) / 4.0
    num_feats["vendido_norm"] = (num_feats["total_vendido"] - num_feats["total_vendido"].min()) / (num_feats["total_vendido"].max() - num_feats["total_vendido"].min() + 1e-9)
    X_text = vectorizer.transform(features["all_comments"])
    X_all = hstack([X_text, num_feats[["preco_norm", "rating_norm", "vendido_norm"]].values])
    preds = model.predict(X_all if hasattr(model, "predict") else X_all.toarray())
    features["predicted_score"] = preds
    features["final_score"] = preds * (1 + alpha * (1 - num_feats["preco_norm"]))
    df = features.sort_values("final_score", ascending=False)
    recos = df[["nome", "preco", "avg_rating", "total_vendido", "final_score"]].head(k)

    plt.figure(figsize=(10,5))
    plt.barh(recos["nome"], recos["final_score"], color='teal')
    plt.xlabel('Pontuação Final')
    plt.ylabel('Produto')
    plt.title('Top 10 Produtos Recomendados')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('grafico_recomendacoes.png')
    plt.show()

    return recos

def main():
    print("[STEP 1] Gerando dados sintéticos com nomes reais...")
    produtos, compras, comentarios = gerar_dados_sinteticos()
    criar_db(produtos, compras, comentarios)
    print("[STEP 2] Carregando features e treinando modelos...")
    features = carregar_features()
    model, vectorizer = treinar_e_comparar(features)
    print("[STEP 3] Gerando recomendações e gráficos...")
    recos = gerar_recomendacoes(features, model, vectorizer)
    print("\n=== 🛒 TOP 10 PRODUTOS RECOMENDADOS ===")
    print(recos.to_string(index=False))
    recos.to_csv(CSV_RECOM_PATH, index=False)
    print(f"\n[INFO] CSV salvo em: {CSV_RECOM_PATH}")
    print("[INFO] Gráficos salvos: grafico_comparacao_modelos.png e grafico_recomendacoes.png")

if __name__ == "__main__":
    main()
