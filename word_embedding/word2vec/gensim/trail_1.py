from gensim.models import Word2Vec


sentences = "The sentences could be text loaded into memory, or an iterator that progressively loads text, required for very large text corpora.".split(' ')
model = Word2Vec(sentences)

words = list(model.wv.vocab)
print(words)

print(model['e'])

model.wv.save_word2vec_format('model.bin')

model.wv.save_word2vec_format('model.txt', binary=False)

m2 = Word2Vec.load('model.bin')

print(m2['e'] == model['e'])
