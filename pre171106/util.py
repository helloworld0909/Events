import re
import json
import jieba
import nltk
import numpy as np

def segmentSent(paragraph):
    sents = paragraph.split('\n')
    res = []
    for sent in sents:
        res.extend(re.findall(r"[^；。！？]+[；。！？]", sent))
    return [sent for sent in res if sent]

def loadWords(filename):
    words = set()
    with open(filename, 'r', encoding='utf-8') as inputFile:
        for line in inputFile:
            words.add(line.strip())
    return words

def dumpFD(fd, filename):
    with open(filename, 'w', encoding='utf-8') as outputFile:
        for tf in fd.most_common(10000):
            outputFile.write('{}\t{}'.format(*tf) + '\n')

def loadCPCReports(filename):
    with open(filename, 'r', encoding='utf-8') as inputFile:
        jsonObj = json.load(inputFile)
        tokenLists = {str(idx): [] for idx in range(14, 20)}
        sentLists = {str(idx): [] for idx in range(14, 20)}
        for idx, report in jsonObj.items():
            text = report['text']
            sents = segmentSent(text)
            sentLists[idx] = sents
            for sent in sents:
                tokenLists[idx].extend(jieba.cut(sent))
    fds = {}
    stopwords = loadWords('data/chineseStopWords.txt')
    for idx, tokenList in tokenLists.items():
        fds[idx] = nltk.FreqDist(tuple(filter(lambda t: t not in stopwords and t.strip() != '', tokenList)))


    vocab = set()
    for idx, fd in fds.items():
        vocab = vocab.union(fd.keys())
    token2idx = {}
    for token in vocab:
        token2idx[token] = len(token2idx)

    docid2idx = {}
    for docid in fds.keys():
        docid2idx[docid] = len(docid2idx)

    dtm = np.zeros((len(fds), len(token2idx)), dtype=np.intc)
    for docid, fd in fds.items():
        for token, freq in fd.items():
            dtm[docid2idx[docid]][token2idx[token]] = freq
    return token2idx, dtm


if __name__ == '__main__':
    token2idx, dtm = loadCPCReports('data/cpc_reports.json')

