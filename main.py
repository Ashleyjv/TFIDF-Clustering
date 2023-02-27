# [1] Getting the descriptions
# [2]Splitting it on \n
# [3] Extracting the first N lines [change with N]
# [4] remove stop words, stemming (common NLP methods)
# [5] add that to the csv
from nltk.stem import PorterStemmer
import csv
import requests
from io import StringIO
import sys
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
nltk.download('punkt')
# import the dataset from sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

def datas():
    categories = [
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'alt.atheism',
     'soc.religion.christian',
    ]
    dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, remove=('headers', 'footers', 'quotes'))
    # print (dataset)
    df = pd.DataFrame(dataset.data, columns=["corpus"])
    # print (df)



class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

def extract(data, module_name, fileptr_write, libDescMap):
    # print ('in function')
    lines = data.split('\n')
    ps = PorterStemmer()
    for line in lines:
        # print (line)
        words = line.split()
        for i in range(len(words)):
            words[i] = ps.stem(words[i])
        stemmed_line = " ".join(words)
        # stemmed_line = preprocess_text(line, True)
        # print(stemmed_line)
        # print (line + "-" + stemmed_line)
        if len(stemmed_line) != 0:
            print ('inside')
            # print(libDescMap)
            fileptr_write.write(module_name + '\n')
            fileptr_write.write(stemmed_line + '\n')
            # libDescMap = {}
            libDescMap['library'].append(module_name)
            libDescMap['description'].append(stemmed_line)
            # print (libDescMap)
            # if len(libDescMap['library']) == 5:
                # print (libDescMap)
            # else:
                # print (len(libDescMap['library']))

        # else:
            # print ('reached here')

        # writerobj.writerow(words)
def get_help(module_name, fileptr_write, libDescMap1):
    count = 0
    # print ('in get_help', module_name)
    try:
        modules_copy = sys.modules.copy()
        module = __import__(module_name)
        with Capturing() as output:
            help(module)
        # print (output)
        Start = output.index('DESCRIPTION')
        extract(output[Start:], module_name, fileptr_write, libDescMap1)
        # print (output[Start:])
    except Exception as e:
        # print (e)
        pass




    # h = help(module)
    # fileptr_write.write(h)
    # print (h)


def main():
    # datas()
    # return

    libDescMap = {'library':[], 'description':[]}
    fileptr = open('/Users/ashleyjvarghese/PycharmProjects/RepoAnalysis/venv/imports.txt', 'r')
    lines = fileptr.readlines()
    fileptr_write = open('stemmed_data.txt', 'w')
    for line in lines:
        lib, count = line.split(',')
        lib = lib.split('.')[0]
        path = 'https://pypi.org/pypi/' + lib + "/json"
        data = requests.get(path).json()
        # print (data)
        try:
            data = data['info']['summary']
            extract(data, lib, fileptr_write, libDescMap)

        except:
            # print (lib)
            get_help(lib, fileptr_write, libDescMap)
            continue


    fileptr_write.close()
    print (len(libDescMap['library']), len(libDescMap['description']))

    df = pd.DataFrame(libDescMap)
    print (df)
    # initialize the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    X = vectorizer.fit_transform(df['description'])
    print (X.toarray())
    #converted df to tfidf

    from sklearn.cluster import KMeans

    # initialize kmeans with 3 centroids
    kmeans = KMeans(n_clusters=3, random_state=42)
    # fit the model
    kmeans.fit(X)
    # store cluster labels in a variable
    clusters = kmeans.labels_

    from sklearn.decomposition import PCA

    # initialize PCA with 2 components
    pca = PCA(n_components=2, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]

    # assign clusters and pca vectors to our dataframe
    df['cluster'] = clusters
    df['x0'] = x0
    df['x1'] = x1

    def get_top_keywords(n_terms):
        """This function returns the keywords for each centroid of the KMeans"""
        df = pd.DataFrame(X.todense()).groupby(clusters).mean()  # groups the TF-IDF vector by cluster
        terms = vectorizer.get_feature_names_out()  # access tf-idf terms
        for i, r in df.iterrows():
            print('\nCluster {}'.format(i))
            print(','.join([terms[t] for t in np.argsort(r)[
                                              -n_terms:]]))  # for each row of the dataframe, find the n terms that have the highest tf idf score

    get_top_keywords(10)

    # map clusters to appropriate labels
    cluster_map = {0: "c1", 1: "c2", 2: "c3"}
    # apply mapping
    df['cluster'] = df['cluster'].map(cluster_map)

    # set image size
    plt.figure(figsize=(12, 7))
    # set a title
    plt.title("TF-IDF + KMeans", fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="viridis")
    plt.show()




    # df['cleaned'] = df['description'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
    # print (df)

if __name__ == "__main__":
    main()




