import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from gensim.models import Word2Vec

model = Word2Vec.load('./model/word2vec_baike')

'''
从txt文件中读取词表
'''
def buildVocabFromFile(filename):
    word_list = []
    with open(filename, encoding='utf-8') as f:
        for word in f.readlines():
            word = word.strip()
            word_list.append(word)
    return word_list


'''
词表去重
'''
def uniquify(word_list):
    return set(word_list)


'''
合并两个词表，并排序
'''
def mergeWordlist(word_list1, word_list2):
    word_list = word_list1 + word_list2
    return set(word_list)


'''
通过词表读入词向量
'''
def buildWordVec(word_list):
    wv = []
    unmap_list = []
    for word in word_list:
        if word not in model.wv.vocab:
            unmap_list.append(word)
        else:
            wv.append(model[word])
    print(f'{len(unmap_list)} words are not in the model.')
    return np.array(wv), unmap_list

'''
以UTF-8编码保存txt词表
'''

def saveVocab(word_list, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    for word in word_list:
        f.write(word+'\n')
    f.close()
    return