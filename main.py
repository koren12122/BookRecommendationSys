"""
Machine learning project - Book Recommendation System

primary functions:

    getClosestBooks:   Given a number N and a book of a name in the database,
                       return list of N most similar books (with respect to the given book) on the database.
                       [ML algorithms used: K-Nearest Neighbors]


"""
import math
from timeit import timeit

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.vq import kmeans, vq
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import time as tm
from IPython.display import display
from tabulate import tabulate

pd.options.mode.chained_assignment = None  # default='warn'

current_year = 2022
df = pd.read_csv('books.csv', on_bad_lines='skip')  # df = data frame
df.drop(['isbn', 'isbn13', 'language_code', 'text_reviews_count', 'publisher'], inplace=True, axis=1)
df2 = df.copy()



def plotTopTenBooks(df):
    top_ten = df[df['ratings_count'] > 2000]  # we want book with at least 2000 reviews
    top_ten.sort_values(by='average_rating', ascending=False)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 10))
    data = top_ten.sort_values(by='average_rating', ascending=False).head(10)
    sns.barplot(x="average_rating", y="title", data=data, palette='inferno')
    plt.show()


def plotTopTenAuthors(df):
    most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(
        10).set_index('authors')
    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x=most_books['title'], y=most_books.index, palette='inferno')
    ax.set_title("Top 10 authors with most books")
    ax.set_xlabel("Total number of books")
    totals = []
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_width() + .2, i.get_y() + .2, str(round(i.get_width())), fontsize=15, color='black')
    plt.show()


def plotTopTenMostReviewed(df):
    most_rated = df.sort_values('ratings_count', ascending=False).head(10).set_index('title')
    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x=most_rated['ratings_count'], y=most_rated.index, palette='inferno')
    totals = []
    for i in ax.patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_width() + .2, i.get_y() + .2, str(round(i.get_width())), fontsize=15, color='black')
    plt.show()


def plotRatingDistribution(df):
    df.average_rating = df.average_rating.astype(float)
    fig, ax = plt.subplots(figsize=[15, 10])
    sns.displot(df['average_rating'])
    # sns.distplot(df['average_rating'], ax=ax)
    ax.set_title('Average rating distribution for all books', fontsize=20)
    ax.set_xlabel('Average rating', fontsize=13)
    plt.show()


def countAndAverageRationRelation(df):
    ax = sns.relplot(data=df, x="average_rating", y="ratings_count", color='red', sizes=(100, 200), height=7,
                     marker='o')
    plt.title("Relation between Rating counts and Average Ratings", fontsize=15)
    ax.set_axis_labels("Average Rating", "Ratings Count")
    plt.show()


def pagesAndRatingRelation():
    """
    Relation between Number Of Pages and Average Ratings
    """
    plt.figure(figsize=(15, 10))
    ax = sns.relplot(x="average_rating", y="  num_pages", data=df, color='red', sizes=(100, 200), height=7, marker='o')
    plt.title("Relation between Number Of Pages and Average Ratings", fontsize=15)
    ax.set_axis_labels("Average Rating", "Number of Pages")
    plt.show()


def BookRecommender(book_name, idlist, df2):
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df2.loc[newid].title)
    return book_list_name


def getClosestBooks(number_of_books, book_name):
    df2 = df.copy()
    df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
    df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
    df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
    df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
    df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"
    rating_df = pd.get_dummies(df2['rating_between'])
    features = pd.concat([rating_df,
                          df2['average_rating'],
                          df2['ratings_count']], axis=1)
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    model = neighbors.NearestNeighbors(n_neighbors=number_of_books + 1, algorithm='ball_tree')
    model.fit(features)
    distances, idlist = model.kneighbors(features)
    print(distances)
    print(idlist.shape)
    BookNames = BookRecommender(book_name, idlist, df2)
    print(BookNames[1:])


def allDistances():
    df2['score'] = df2.apply(score_calc, axis=1)
    top20 = df2.sort_values('score', ascending=False).head(20)





def plotElbow(remove_outliers=True):
    trial = df[['average_rating', 'ratings_count']]

    if remove_outliers:
        # print(trial.idxmax())
        trial.drop(10336, inplace=True)
        # print(trial.idxmax())
        trial.drop(1697, inplace=True)  # ignoring from the 2 outliers (2 data points that skewed the clustering)

    data = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count'])]).T
    X = data
    distortions = []
    for k in range(2, 15):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        distortions.append(k_means.inertia_)

    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(2, 15), distortions, 'bx-')
    plt.title("Elbow Curve")
    # plt.show()
    return data, 5


def plotClusters(idx, data, centroids):
    sns.set_context('paper')
    plt.figure(figsize=(15, 10))
    plt.plot(data[idx == 0, 0], data[idx == 0, 1], 'or',  # red circles
             data[idx == 1, 0], data[idx == 1, 1], 'ob',  # blue circles
             data[idx == 2, 0], data[idx == 2, 1], 'oy',  # yellow circles
             data[idx == 3, 0], data[idx == 3, 1], 'om',  # magenta circles
             data[idx == 4, 0], data[idx == 4, 1], 'ok',  # black circles

             )
    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8, )

    circle1 = Line2D(range(1), range(1), color='red', linewidth=0, marker='o', markerfacecolor='red')
    circle2 = Line2D(range(1), range(1), color='blue', linewidth=0, marker='o', markerfacecolor='blue')
    circle3 = Line2D(range(1), range(1), color='yellow', linewidth=0, marker='o', markerfacecolor='yellow')
    circle4 = Line2D(range(1), range(1), color='magenta', linewidth=0, marker='o', markerfacecolor='magenta')
    circle5 = Line2D(range(1), range(1), color='black', linewidth=0, marker='o', markerfacecolor='black')

    plt.legend((circle1, circle2, circle3, circle4, circle5)
               , ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'), numpoints=1, loc=0, )
    plt.show()


def plotElbowTwo(pca):
    X = pca
    distortions = []
    for k in range(2, 15):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        distortions.append(k_means.inertia_)
    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(2, 15), distortions, 'bx-')
    plt.title("Elbow Curve")
    # plt.show()
    return pca, 5


def averageRateAndRateCount():
    data, clusters_number = plotElbow()
    centroids, _ = kmeans(data, clusters_number)
    idx, _ = vq(data, centroids)
    plotClusters(idx, data, centroids)


def booksDistributionCluster():
    df2 = df.copy()
    df2.drop(['bookID', 'publication_date', 'title', 'authors'], inplace=True, axis=1)
    pca = PCA(n_components=2).fit_transform(df2)
    pca, clusters_number = plotElbowTwo(pca)
    kmeans = KMeans(init='k-means++', n_clusters=clusters_number, n_init=1)
    y_kmeans = kmeans.fit_predict(pca)
    plt.scatter(pca[y_kmeans == 0, 0], pca[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(pca[y_kmeans == 1, 0], pca[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(pca[y_kmeans == 2, 0], pca[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(pca[y_kmeans == 3, 0], pca[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(pca[y_kmeans == 4, 0], pca[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('book distribution clustering')
    plt.legend()
    plt.show()


relu = lambda a: max(a, 0)


def score_calc(data):
    r = data['average_rating']
    c = data['ratings_count']
    return math.log(math.pow(r, 5) * math.log(relu(c-250)+2) + 1)


def plotRealTopTenBooks(total_vote):
    total_vote = total_vote.sort_values(by='score', ascending=False)
    plt.style.use('seaborn-ticks') #seaborn-whitegrid , , 'seaborn-muted', , , 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white'
    plt.figure(figsize=(10, 10))
    data = total_vote.sort_values(by='score', ascending=False).head(10)
    sns.barplot(x='score', y="title", data=data, palette='inferno')
    plt.show()


def calculateTopScoreBooks():
    total_vote = df2.drop_duplicates(subset=['authors', 'title'], keep='first')
    total_vote.reset_index(inplace=True)
    total_vote = total_vote[['authors', 'title', 'average_rating', 'ratings_count']]
    total_vote['score'] = total_vote.apply(score_calc, axis=1)
    total_vote = total_vote.sort_values(by='score', ascending=False).head(30)
    plotRealTopTenBooks(total_vote)


def addAge(): # add new column named "book age"
    df2['book age'] = df2['publication_date'].apply(lambda x: current_year - int(x.split('/')[2]))


def getRandomBooks(num):
    print(df2.sample(n=num)['title'])


if __name__ == '__main__':
    number_of_books = 2
    # print(df.isnull().sum())  # make sure there is no NULL or NaN values
    # print(df.describe())
    # plotTopTen(df)
    plotTopTenAuthors(df)
    # plotTopTenMostReviewed(df)
    # plotRatingDistribution(df)
    # averageRateAndRateCount()
    # pagesAndRatingRelation(df)
    # calculateTopScoreBooks()
    # df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'Book Age'] = 2022 - df2['average_rating'] -
    # addAge()
    # allDistances()
    # getClosestBooks(3, "Pandora's Star")
    # calculateTopScoreBooks()
    # booksDistributionCluster()

