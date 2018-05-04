from keras.preprocessing.text import Tokenizer

a = ['word1 word2 word3','word2 word3 word1']

tk = Tokenizer()
tk.fit_on_texts(a)
word_index = tk.word_index
for word, i in word_index.items():
    print(word)