import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def linear_transform(scores):
    score_avr = np.mean(scores)
    scores_transform = []
    for score in scores:
        score_transform =  np.abs(score_avr) + (np.abs(score) - np.abs(score_avr)) / (1 - np.abs(score_avr))
        if score < 0:
            score_transform = -score_transform
        scores_transform.append(score_transform)
    return scores_transform

def read_scores(file_path):
    papers = []
    positive_scores = []
    negative_scores = []
    positive_counts = []
    negative_counts = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Paper:"):
                papers.append(line.split(":")[1].strip())
            elif line.startswith("Positive Scores:"):
                scores = list(map(float, line.split(":")[1].strip().strip('[]').split(',')))
                scores_log = linear_transform(scores)
                positive_scores.append(scores_log)  # Use append to keep scores separate for each paper
                positive_counts.append(len(scores))
            elif line.startswith("Negative Scores:"):
                scores = list(map(float, line.split(":")[1].strip().strip('[]').split(',')))
                scores_log = linear_transform(scores)
                negative_scores.append(scores_log)  # Use append to keep scores separate for each paper
                negative_counts.append(len(scores))

    return papers, positive_scores, negative_scores , positive_counts, negative_counts


def draw_boxplot(data):
    # Set plot style
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Plot box plot
    sns.boxplot(data=data, x="Newspaper", y="Score", hue="Sentiment", palette="Set2", showfliers=False)
    plt.title("Sentiment Score Distribution by Newspaper", fontsize=16)
    plt.ylabel("Sentiment Score(Linear transformed)", fontsize=12)
    plt.xlabel("Newspaper", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment", loc="lower left")
    plt.tight_layout()
    plt.show()

# draw 比例堆积柱状图
def draw_barplot(data):

    papers = data["Newspaper"].unique()
    positive_counts = data.groupby("Newspaper")["Positive Counts"].sum().tolist()
    negative_counts = data.groupby("Newspaper")["Negative Counts"].sum().tolist()

    total_counts = [p + n for p, n in zip(positive_counts, negative_counts)]
    positive_ratios = [p / t for p, t in zip(positive_counts, total_counts)]
    negative_ratios = [n / t for n, t in zip(negative_counts, total_counts)]

    plt.figure(figsize=(14, 7))
    plt.bar(papers, positive_ratios, label="Positive", color='green', alpha=0.7)
    plt.bar(papers, negative_ratios, bottom=positive_ratios, label="Negative", color='red', alpha=0.7)
    plt.ylabel("Proportion")
    plt.title("Proportion of Positive and Negative Sentiments by Newspaper")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Read data
    file_path = '/Users/mark/大学/数据挖掘与安全治理/Data_Mining/proj3/results/D3_scores.txt'
    papers, positive_scores, negative_scores, positive_counts, negative_counts = read_scores(file_path)


    # Prepare data for DataFrame
    data = {
        "Newspaper": [],
        "Sentiment": [],
        "Score": [],
        "Positive Counts": [],
        "Negative Counts": [],
    }

    for i, paper in enumerate(papers):
        data["Newspaper"].extend([paper] * len(positive_scores[i]))
        data["Sentiment"].extend(["Positive"] * len(positive_scores[i]))
        data["Score"].extend(positive_scores[i])
        data["Positive Counts"].extend([positive_counts[i]] * len(positive_scores[i]))
        data["Negative Counts"].extend([0] * len(positive_scores[i]))  # Ensure same length

        data["Newspaper"].extend([paper] * len(negative_scores[i]))
        data["Sentiment"].extend(["Negative"] * len(negative_scores[i]))
        data["Score"].extend(negative_scores[i])
        data["Negative Counts"].extend([negative_counts[i]] * len(negative_scores[i]))
        data["Positive Counts"].extend([0] * len(negative_scores[i]))  # Ensure same length


    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Draw boxplot
#    draw_boxplot(df)

    # Draw barplot
    draw_barplot(df)




if __name__ == "__main__":
    main()

