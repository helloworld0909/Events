import numpy as np
import lda
import lda.datasets
import util

token2idx, X = util.loadCPCReports('data/cpc_reports.json')
idx2token = {v: k for k, v in token2idx.items()}
print(X.shape)
model = lda.LDA(n_topics=5, n_iter=1500, random_state=1)
model.fit(X)
topic_word = model.topic_word_
n_top_words = 20
for i, topic_dist in enumerate(topic_word):
    topic_words = list(map(idx2token.get, np.argsort(topic_dist)[:-(n_top_words+1):-1]))
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))