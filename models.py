import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add, Flatten 
from tensorflow.keras.layers import Embedding, LSTM, GRU, concatenate, Dense, TimeDistributed
from tensorflow.keras.models import Model

def CNN_1D(MAX_NUM_WORDS, NUM_EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUM_LSTM_UNITS, NUM_CLASSES):

    embedding_layer = layers.Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)

    top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    bm_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

    embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    top_embedded = embedding_layer(top_input)
    bm_embedded = embedding_layer(bm_input)

    shared_Conv1D = Conv1D(128, 3, activation='relu', padding='same')
    top_Conv1D = shared_Conv1D(top_embedded)
    bm_Conv1D = shared_Conv1D(bm_embedded)

    shared_Conv1D = MaxPooling1D(pool_size=2)
    top_flatten = shared_Conv1D(top_Conv1D)
    bm_flatten = shared_Conv1D(bm_Conv1D)

    shared_Flatten = Flatten()
    top_output = shared_Flatten(top_flatten)
    bm_output = shared_Flatten(bm_flatten)


    merged = concatenate([top_output, bm_output], axis=-1)

    dense =  Dense(units=NUM_CLASSES, activation='softmax')
    predictions = dense(merged)

    # 我們的模型就是將數字序列的輸入，轉換
    # 成 3 個分類的機率的所有步驟 / 層的總和
    model = Model(inputs=[top_input, bm_input], 
                  outputs=predictions)

    return model

def GRU_model(MAX_NUM_WORDS, NUM_EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUM_LSTM_UNITS, NUM_CLASSES):

    embedding_layer = layers.Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    # 分別定義 新聞標題與內文為 A & B 做為模型輸入
    # A & B 都是一個長度為 27 的數字序列
    input = Input(shape=(54, ), dtype='int32')

    # 詞嵌入層
    # 經過詞嵌入層的轉換，兩個新聞標題都變成
    # 一個詞向量的序列，而每個詞向量的維度
    embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    embedded = embedding_layer(input)

    # LSTM 層
    shared_128 = GRU(NUM_LSTM_UNITS, return_sequences=True)
    top_1 = shared_128(embedded)

    shared_64 = GRU(64, return_sequences=False)
    top_2 = shared_64(top_1)


    # 串接層將兩個新聞標題的結果串接單一向量
    # 方便跟全連結層相連
    merged = concatenate([top_2], axis=-1)

    dense =  Dense(units=NUM_CLASSES, activation='softmax')
    predictions = dense(merged)

    # 我們的模型就是將數字序列的輸入，轉換
    # 成 3 個分類的機率的所有步驟 / 層的總和
    model = Model(inputs=[input], 
                  outputs=predictions)
    
    return model

def Simple_LSTM(MAX_NUM_WORDS, NUM_EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, NUM_LSTM_UNITS, NUM_CLASSES):

    embedding_layer = layers.Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    # 分別定義 新聞標題與內文為 A & B 做為模型輸入
    # A & B 都是一個長度為 27 的數字序列
    top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    bm_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

    # 詞嵌入層
    # 經過詞嵌入層的轉換，兩個新聞標題都變成
    # 一個詞向量的序列，而每個詞向量的維度
    embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    top_embedded = embedding_layer(top_input)
    bm_embedded = embedding_layer(bm_input)

    # LSTM 層
    shared_conv1D_128 = Conv1D(filters=128, kernel_size=1, strides=1, padding='same')
    top_conv_1 = shared_conv1D_128(top_embedded)
    bm_conv_1 = shared_conv1D_128(bm_embedded)
    
    # LSTM 層
    shared_lstm_128 = LSTM(NUM_LSTM_UNITS, return_sequences=True)
    top_1 = shared_lstm_128(top_conv_1)
    bm_1 = shared_lstm_128(bm_conv_1)

    shared_lstm_64 = LSTM(64, return_sequences=True)
    top_2 = shared_lstm_64(top_1)
    bm_2 = shared_lstm_64(bm_1)

    # 串接層將兩個新聞標題的結果串接單一向量
    # 方便跟全連結層相連
    merged = concatenate([top_2, bm_2], axis=-1)

    dense =  TimeDistributed(Dense(units=NUM_CLASSES, activation='softmax'))
    predictions = dense(merged)

    # 我們的模型就是將數字序列的輸入，轉換
    # 成 3 個分類的機率的所有步驟 / 層的總和
    model = Model(inputs=[top_input, bm_input], 
                  outputs=predictions)
    
    return model