import numpy as np
import pickle
from keras.models import load_model

class NlpService:

    def __init__(self, answer):
        self.answer = answer
        # インデックスと文字で辞書を作成
        self.char_indices = {}
        self.indices_char = {}
        self.max_length_x = 128


        with open('models/kana_chars.pickle', mode='rb') as f:
            chars_list = pickle.load(f)

        for i, char in enumerate(chars_list):
            self.char_indices[char] = i
        for i, char in enumerate(chars_list):
            self.indices_char[i] = char

        self.n_char = len(chars_list)

# 文章をone-hot表現に変換する関数
    def sentence_to_vector(self, sentence):
        vector = np.zeros((1, self.max_length_x, self.n_char), dtype=np.bool)
        for j, char in enumerate(sentence):
            vector[0][j][self.char_indices[char]] = 1
        return vector

    def create_response_answer(self, message, beta=5):
        encoder_model = load_model('models/encoder_model.h5')
        decoder_model = load_model('models/decoder_model.h5')
        vec = self.sentence_to_vector(message)  # 文字列をone-hot表現に変換
        state_value = encoder_model.predict(vec)
        y_decoder = np.zeros((1, 1, self.n_char))  # decoderの出力を格納する配列
        y_decoder[0][0][self.char_indices['\t']] = 1  # decoderの最初の入力はタブ。one-hot表現にする。

        respond_sentence = ""  # 返答の文字列
        while True:
            y, h = decoder_model.predict([y_decoder, state_value])
            p_power = y[0][0] ** beta  # 確率分布の調整
            next_index = np.random.choice(len(p_power), p=p_power/np.sum(p_power))
            next_char = self.indices_char[next_index]  # 次の文字
            if (next_char == "\n" or len(respond_sentence) >= self.max_length_x):
                break  # 次の文字が改行のとき、もしくは最大文字数を超えたときは終了
            respond_sentence += next_char
            y_decoder = np.zeros((1, 1, self.n_char))  # 次の時刻の入力
            y_decoder[0][0][next_index] = 1

            state_value = h  # 次の時刻の状態

        return respond_sentence
        # return "こんにちは"


