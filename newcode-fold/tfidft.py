import textblob

word = textblob.Word('fuxked')
word = word.singularize().lemmatize("v")
print(word)