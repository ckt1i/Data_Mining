import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans , SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取文件
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    texts = []
    text = []
    for line in lines:
        if len(line) == 0:
            continue
        if line == '---':
            if len(text) > 0:
                texts.append(''.join(text))
            text = []
        else:
            text.append(line)
    return texts

# 调整情感词权重
def emphasize_sentiment_words(text, positive_words, negative_words):
    words = text.split()
    emphasized_text = []
    for word in words:
        if word in positive_words:
            emphasized_text.extend([word] * 3)  # 正面词重复 3 次
        elif word in negative_words:
            emphasized_text.extend([word] * 3)  # 负面词重复 3 次
        else:
            emphasized_text.append(word)
    return " ".join(emphasized_text)

# 调整情感分析结果
def adjust_sentiment(sentiments, positive_words, negative_words, texts):
    adjusted_sentiments = []
    for sentiment, text in zip(sentiments, texts):
        if sentiment['label'] == 'neutral' and sentiment['score'] < 0.9997:
            if any(word in text for word in positive_words):
                sentiment['label'] = 'POSITIVE'
            elif any(word in text for word in negative_words):
                sentiment['label'] = 'NEGATIVE'
        adjusted_sentiments.append(sentiment)
    return adjusted_sentiments

# 可视化聚类结果
def visualize_clustering(X_pca, model , sentiments, fig_path):
    colors = [sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'] for sentiment in sentiments]
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='coolwarm', s=50, alpha=0.7)

    # 添加聚类中心
    for cluster in np.unique(model.labels_):
        center = np.mean(X_pca[model.labels_ == cluster], axis=0)
        plt.scatter(center[0], center[1], c='black', s=200, marker='x')

    # 添加颜色条
    plt.colorbar(scatter, label='Sentiment Score，Red: Positive, Blue: Negative')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-means Clustering with Sentiment Scores")
    plt.savefig(fig_path)

# 评估聚类效果
def esitimate(X, model_Kmeans, model_EM, model_SC):
    from sklearn.metrics import silhouette_score
    Silscore = []
    Silscore.append(silhouette_score(X, model_Kmeans.labels_))
    Silscore.append(silhouette_score(X, model_EM.predict(X)))
    Silscore.append(silhouette_score(X, model_SC.labels_))
    model = [model_Kmeans, model_EM, model_SC][np.argmax(Silscore)]
    return model , Silscore

def cluster(filename , is_visualization = False):
    # 配置路径
    model_path = r"D:\Data_Mining\proj3\models\bert"

    # 读取数据
    texts = read_file(filename)
    print("Number of texts: ", len(texts))

    # 加载德语分词模型
    nlp = spacy.load("de_core_news_sm")
    texts_cleaned = [" ".join([token.text for token in nlp(text)]) for text in texts]

    # 定义情感词
    positive_words = {"großartig", "freude", "liebe", "erfolg"}
    negative_words = {"hasse", "schlecht", "traurig", "verlierer"}
    texts_emphasized = [emphasize_sentiment_words(text, positive_words, negative_words) for text in texts_cleaned]

    # 加载情感分析模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # 情感分析
    max_length = 512
    texts_truncated = [text[:max_length] for text in texts_cleaned]
    sentiments = [sentiment_analyzer(text)[0] for text in texts_truncated]
    sentiments_adjusted = adjust_sentiment(sentiments, positive_words, negative_words, texts_cleaned)

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts_emphasized).toarray()

    # 添加情感分数作为附加特征
    sentiment_scores = np.array([sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'] for sentiment in sentiments_adjusted]).reshape(-1, 1)
    X_with_sentiments = np.hstack((X, sentiment_scores))

    # 选取合适的聚类方式进行聚类
    model_kmeans = KMeans(n_clusters=2, random_state=42)
    model_kmeans.fit(X_with_sentiments)
    model_EM = GaussianMixture(n_components=2, random_state=42)
    model_EM.fit(X_with_sentiments)
    model_SC = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
    model_SC.fit(X_with_sentiments)

    model , Silscore = esitimate(X_with_sentiments, model_kmeans, model_EM, model_SC)

    # 输出积极和消极情感的聚类
    Negative_indices = np.where(model.labels_ == 0)
    Negative_sentiments = [sentiments_adjusted[i] for i in Negative_indices[0]]
    Negative_scores = [sentiment['score']*(-1) for sentiment in Negative_sentiments]

    Positive_indices = np.where(model.labels_ == 1)
    Positive_sentiments = [sentiments_adjusted[i] for i in Positive_indices[0]]
    Positive_scores = [sentiment['score'] for sentiment in Positive_sentiments]

    if is_visualization == False:  
        return Positive_scores, Negative_scores, Silscore

    # PCA 降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_with_sentiments)

    # 可视化
    fig_path = r"D:\Data_Mining\proj3\results\D3_Kmeans_optimized.png"
    visualize_clustering(X_pca, model, sentiments_adjusted, fig_path)
    print("Visualization saved to: ", fig_path)   


def main():
    main_path = r"D:\Data_Mining\proj3\Datasets\报刊合集"
    scores = []
    estimation = []
    # 读取目录下的所有文件并进行分析
    papers = os.listdir(main_path)
    for paper in papers:
        filename = os.path.join(main_path, paper)
        print(f"Processing file: {filename}\n")  # Debugging: Print the file path
        Positive_scores, Negative_scores, Sil_scores = cluster(filename, is_visualization = False)
        scores.append([Positive_scores, Negative_scores])
        estimation.append(Sil_scores)

    with open(r"D:\Data_Mining\proj3\results\D3_scores.txt", 'w', encoding='utf-8') as f:
        for i in range(len(papers)):
            f.write("Paper: " + papers[i] + "\n")
            f.write("Positive Scores: " + str(scores[i][0]) + "\n")
            f.write("Negative Scores: " + str(scores[i][1]) + "\n")
            f.write("\n")

    with open(r"D:\Data_Mining\proj3\results\D3_estimation.txt", 'w', encoding='utf-8') as f:
        f.write(",".join(papers) + "\n")
        for i in range(3):
            f.write(",".join([str(estimation[j][i]) for j in range(len(papers))]) + "\n")
    

if __name__ == '__main__':
    main()