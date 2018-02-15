# -*- coding: utf-8 -*-

from gensim.models import word2vec

model= word2vec.Word2Vec.load_word2vec_format('FinalModel')

model=model.most_similar(positive=[u'امریکا',u'روحانی'],negative=[u'ایران'])

for item in model:

    print (item[0])