import os
import tokenization

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import jieba.posseg as pseg
from bs4 import BeautifulSoup

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)

train=pd.read_csv("train.csv").reset_index(drop=True)
test=pd.read_csv("test.csv").reset_index(drop=True)

train['Review'] = (train['headline'].map(str) +' '+ train['short_description']).apply(lambda row: row.strip())
test['Review'] = (test['headline'].map(str) +' '+ test['short_description']).apply(lambda row: row.strip())

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    stopword_list=nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([word for word, flag in words if flag != 'x'])

def preprocess(text):
    text.apply(remove_between_square_brackets, remove_special_characters)
    text.apply(simple_stemmer, remove_stopwords)

    return text

train['Review'] = preprocess(train['Review'])
test['Review'] = preprocess(test['Review'])



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

max_len = 27

def bert_encode(texts, tokenizer, max_len = max_len):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=max_len):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(512, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(256, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(128, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(64, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(30, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=0.5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


train_input = bert_encode(train.Review.values, tokenizer, max_len=max_len)
test_input = bert_encode(test.Review.values, tokenizer, max_len=max_len)

label_to_index = {
    'BLACK VOICES':0, 
    'BUSINESS':1, 
    'WOMEN':2, 
    'WELLNESS':3, 
    'WEIRD NEWS':4, 
    'WEDDINGS':5, 
    'TRAVEL':6, 
    'THE WORLDPOST':7, 
    'TECH':8, 
    'TASTE':9,
    'STYLE & BEAUTY':10, 
    'STYLE':11, 
    'SPORTS':12, 
    'SCIENCE':13, 
    'RELIGION':14, 
    'QUEER VOICES':15, 
    'POLITICS':16, 
    'PARENTS':17, 
    'PARENTING':18, 
    'MEDIA':19,
    'IMPACT':20, 
    'HOME & LIVING':21, 
    'HEALTHY LIVING':22, 
    'GREEN':23, 
    'FOOD & DRINK':24, 
    'ENTERTAINMENT':25, 
    'DIVORCE':26, 
    'CRIME':27, 
    'COMEDY':28, 
    'WORLD NEWS':29
}

y_label = train.category.apply(lambda x: label_to_index[x])
y_label = np.asarray(y_label).astype('float32')
train_labels = tf.keras.utils.to_categorical(y_label, num_classes=30)

model = build_model(bert_layer, max_len=max_len)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3, verbose=1)
 
train_history = model.fit(train_input, train_labels, 
                          # validation_split=0.1,
                          epochs=15,
                          callbacks=[checkpoint, earlystopping],
                          batch_size=32,
                          verbose=1)

# Submission
model.load_weights('model.h5')
test_pred = model.predict(test_input)

SUBMISSION = pd.read_csv('sample_submission.csv', index_col=0)
index_to_label = {v: k for k, v in label_to_index.items()}
test_input = [index_to_label[idx] for idx in np.argmax(test_pred, axis=1)]
submission = pd.DataFrame(test_input, columns=['category'])

submission = submission.loc[:, ['category']].reset_index()
submission.rename(columns = {'index':'id'}, inplace = True)

submission.to_csv('submission.csv', index=False)

import zipfile
with zipfile.ZipFile('submission.zip', mode='w') as zf:
    zf.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)