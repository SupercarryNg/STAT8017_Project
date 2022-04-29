from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
import gensim

import pandas as pd
import numpy as np

import umap  # Please pip install umap-learn NOT UMAP

import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, vec):
        self.vec = vec

    def __getitem__(self, idx):
        res = torch.from_numpy(self.vec[idx]).float()
        return res

    def __len__(self):
        return self.vec.shape[0]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded


class Topic_Model:
    #
    def __init__(self, texts, common_texts, class_name, method='TFIDF', k=3):  ###k改成了3
        """
        k: Number of Topics
        texts:
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = corpora.Dictionary(common_texts)
        self.common_corpus = [self.dictionary.doc2bow(text) for text in common_texts]
        self.cluster_model = None
        self.ldamodel = None
        self.vec_dict = {'TFIDF': None, 'LDA': None, 'BERT': None, 'LDA_BERT': None}
        self.lab_dict = {'TFIDF': None, 'LDA': None, 'BERT': None, 'LDA_BERT': None}
        self.gamma = 15  # gamma是一个权重参数，后面需要把LDA和BERT连接在一起，所以需要设定一个权重，谁重要一点
        self.method = method
        self.texts = texts
        self.common_texts = common_texts
        self.class_name = class_name

    # 词向量化
    def vectorize(self, method=None):
        # 没有实例化所以不能在类函数里用self.做形参
        if method == None:
            method = self.method
        # turn tokenized documents into a id <-> term dictionary
        # self.dictionary = corpora.Dictionary(common_texts)
        # convert tokenized documents into a document-term matrix
        # self.corpus = [self.dictionary.doc2bow(text) for text in common_texts]

        if method == 'LDA':
            lda_model = gensim.models.ldamodel.LdaModel(self.common_corpus, num_topics=self.k, alpha='auto',
                                                        id2word=self.dictionary, passes=20)
            n_comments = len(self.common_corpus)
            vec_lda = np.zeros((n_comments, self.k))

            for i in range(n_comments):
                # get_document_topics是一个用于推断文档主题归属的函数/方法，
                # 在这里，假设一个文档可能同时包含若干个主题，但在每个主题上的概率不一样，
                # 概率大的代表该文档越有可能从属于该主题。
                for topic, prob in lda_model.get_document_topics(self.common_corpus[i]):
                    vec_lda[i, topic] = prob
            vec = vec_lda

        elif method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            tfidf_vec = tfidf.fit_transform(self.texts)  ###我把texts改成了self.texts
            print('Getting vector representations for TF-IDF. Done!')
            vec = tfidf_vec

        elif method == 'BERT':
            # Embedding
            # https://www.sbert.net/docs/pretrained_models.html
            # 选用了速度相对较快的小模型
            # 拟合效果最好的预训练模型：all-mpnet-base-v2
            model = SentenceTransformer('all-MiniLM-L12-v2')
            vec_bert = np.array(model.encode(self.texts, show_progress_bar=True))  ####我把texts改成了self.texts
            vec = vec_bert

        elif method == 'LDA_BERT':
            vec_lda = self.vec_dict['LDA']
            vec_bert = self.vec_dict['BERT']
            if type(vec_lda) != np.ndarray or type(vec_bert) != np.ndarray:
                raise Exception('please vectorize and fit LDA/BERT first!')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]

            latent_dim = 32
            epochs = 200
            lr = 0.008
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            X = vec_ldabert
            input_dim = X.shape[1]
            trainset = MyData(X)
            train = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)

            model = Autoencoder(input_dim, latent_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_function = nn.MSELoss()

            for epoch in range(epochs):
                for inputs in train:
                    # Forward
                    inputs = inputs.to(device)
                    codes, decoded = model(inputs)

                    # Backward
                    optimizer.zero_grad()
                    loss = loss_function(decoded, inputs)
                    loss.backward()
                    optimizer.step()

            vec, _ = model(torch.from_numpy(X).float().to(device))

            # input_vec = Input(shape=(input_dim,))
            # encoded = Dense(latent_dim, activation='relu')(input_vec)
            # decoded = Dense(input_dim, activation='relu')(encoded)
            # autoencoder = Model(input_vec, decoded)
            # encoder = Model(input_vec, encoded)
            # encoded_input = Input(shape=(latent_dim,))
            # decoder_layer = autoencoder.layers[-1]
            # decoder = Model(encoded_input, autoencoder.layers[-1](encoded_input))
            # autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
            #
            # X = vec_ldabert
            # X_train, X_test = train_test_split(X)
            # history = autoencoder.fit(X_train, X_train, epochs=200, batch_size=128, shuffle=True,
            #                           validation_data=(X_test, X_test), verbose=0)

            # vec = encoder.predict(vec_ldabert)
            vec = vec.data.cpu().numpy()
        self.vec_dict[method] = vec  ##########################################会不会有问题
        return vec

    def fit(self, method=None, m_clustering=KMeans):
        if method == None:
            method = self.method
        # 默认聚类使用K均值聚类
        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.common_corpus, num_topics=self.k, alpha='auto',
                                                                id2word=self.dictionary, passes=20)
                print('Fitting LDA Done!')

        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            #             # 不同方法向量化后的值用一个字典存起来，前面初始化过了vec={}
            #             self.vec[method] = self.vectorize(texts, common_corpus, method)
            testmodel = self.cluster_model.fit(self.vec_dict[method])
            print('Clustering embeddings. Done!')
            lbs = testmodel.labels_
            self.lab_dict[method] = lbs

    def reduce_(self, vec):
        reducer = umap.UMAP(min_dist=0.9, random_state=42, n_components=2, n_neighbors=200)  ### 我修改了UMAP得参数
        embedding = reducer.fit_transform(vec)
        return embedding

        #################################################

    def visualize(self, method=None):  # 一定先vectorize，fit
        if method == None:
            method = self.method
        if method == 'LDA':
            return
        k = self.k
        vec = self.vec_dict[method]
        lbs = self.lab_dict[method]
        #         if vec.all() == None or lbs == None:
        #             raise Exception('please vectorize and fit first!')
        embedding = self.reduce_(vec)
        if method == 'TFIDF':
            plt.figure(figsize=(12, 8))
        else:
            plt.figure(figsize=(16, 9))

        path = './Results/' + self.class_name + '/%d/cluster/' % k
        if not os.path.exists(path):
            os.makedirs(path)
        # 统计每一类有多少数据
        counter = Counter(lbs)
        n = len(embedding)
        for i in range(len(np.unique(lbs))):
            plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', ms=1.5, alpha=0.5,
                     label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
        plt.legend()
        plt.savefig(path + 'clusterplot')

    def wordcloudplot(self, topic, lbs):  # 不建议直接调用
        common_texts = self.common_texts
        tokens = ' '.join([' '.join(_) for _ in np.array(common_texts)[lbs == topic]])
        k = self.k
        path = './Results/' + self.class_name + '/%d/wordcloudplot/' % k
        if not os.path.exists(path):
            os.makedirs(path)

        wordcloud = WordCloud(width=800, height=560, background_color='white', collocations=False, min_font_size=10,
                              font_path=r'C:\Windows\Fonts\SIMLI.ttf').generate(tokens)

        # plot the WordCloud image
        plt.figure(figsize=(8, 5.6), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(path + str(topic))

    def get_wordcloud(self, method=None):

        #    Get word cloud of each topic from fitted model
        #    :param model: Topic_Model object
        #    :param sentences: preprocessed sentences from docs
        if method == None:
            method = self.method
        if self.method == 'LDA':
            return

        lbs = self.lab_dict[method]
        for i in range(self.k):
            print('Getting wordcloud for topic {} ...'.format(i))
            self.wordcloudplot(i, lbs)

    def get_topics(self, method=None):
        if method == None:
            method = self.method
        lbs = self.lab_dict[method]
        labels = lbs
        k = len(np.unique(lbs))
        topics = ['' for _ in range(k)]
        for i, c in enumerate(self.common_texts):
            topics[labels[i]] += (' ' + ' '.join(c))
            # 把按照label来对commontexts遍历，找到对应label所对应的句子，然后把他们连接到一起，topics的长度为8

        # 不同的label分别统计词频，即为主题
        word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
        # get sorted word counts
        word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
        # 只取前十的词
        topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

        return topics

    def evaluate_model(self, method=None):
        if method == None:
            method = self.method
        topics = self.get_topics(method)
        lbs = self.lab_dict[method]
        vec = self.vec_dict[method]
        k = self.k

        cm = CoherenceModel(topics=topics, texts=self.common_texts, corpus=self.common_corpus,
                            dictionary=self.dictionary, coherence='c_v')
        cv = cm.get_coherence()
        cm = CoherenceModel(topics=topics, texts=self.common_texts, corpus=self.common_corpus,
                            dictionary=self.dictionary, coherence='u_mass')
        umass = cm.get_coherence()
        s_score = silhouette_score(vec, lbs)

        path = './Results/' + self.class_name + '/%d/' % k

        evaluation = pd.DataFrame({'C_V': cv, 'U_mass': umass, 'silhouette_score': s_score}, index=[0])
        evaluation.to_csv(path + 'Evaluation.csv')


if __name__ == '__main__':
    class_name = 'data_total'
    df = pd.read_csv(class_name + '.csv')
    texts = df['stem']
    texts = texts.dropna()
    texts.isna().sum()
    common_texts = []
    for text in texts:
        tmp = text.split(' ')
        common_texts.append(list(filter(None, tmp)))

    texts = texts.values.tolist()
    # print(common_texts)
    ldabert_Model=Topic_Model(texts, common_texts, class_name, method='LDA_BERT', k=8)
    ldabert_Model.vectorize(method='LDA')
    ldabert_Model.vectorize(method='BERT')
    ldabert_Model.vectorize(method='LDA_BERT')
    ldabert_Model.fit()
    ldabert_Model.visualize()
    ldabert_Model.get_wordcloud()