
data = open('DataA.txt').read()

corpus = data.lower().split("\n")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1


input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0] # chuyen cau thanh chuoi so
    for i in range (1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        print(n_gram_sequence)
        input_sequences.append(n_gram_sequence)