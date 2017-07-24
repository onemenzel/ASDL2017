import tflearn
import numpy
import datetime

# TODO reset load restriction
truncate_token_lists = 70   # int or None
truncate_before_counting = True

class Code_Completion_Baseline:

    def __init__(self):
        self.string_to_number = None
        self.number_to_string = None
        self.net = None
        self.model = None

    @staticmethod
    def token_to_string(token):
        return token["type"] + "-@@-" + token["value"]

    @staticmethod
    def string_to_token(string):
        split = string.split("-@@-")
        return {"type": split[0], "value": split[1]}

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def prepare_data(self, token_lists):

        if truncate_before_counting:
            del token_lists[truncate_token_lists:]

        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        # prepare x,y pairs
        xs = []
        ys = []

        # truncate the token lists for faster debugging
        if truncate_token_lists and not truncate_before_counting:
            del token_lists[truncate_token_lists:]

        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(previous_token_string))
                    ys.append(self.one_hot(token_string))

        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    def create_network(self):
        # TODO try convolutional RNN and LSTM shapes
        net = tflearn.input_data(shape=[None, len(self.string_to_number)])
        net = tflearn.reshape(net, (-1, len(self.string_to_number), 1))
        # self.net = tflearn.fully_connected(self.net, 32)
        # self.net = tflearn.fully_connected(self.net, 64)

        # get the "good" parts
        net = tflearn.conv_1d(net, len(self.string_to_number), 3, activation='relu', regularizer="L2")
        net = tflearn.max_pool_1d(net, 2)
        net = tflearn.batch_normalization(net)
        net = tflearn.conv_1d(net, len(self.string_to_number)*2, 3, activation='relu', regularizer="L2")
        net = tflearn.max_pool_1d(net, 2)
        net = tflearn.batch_normalization(net)

        # remember the meanings
        net = tflearn.lstm(net, 64)
        net = tflearn.dropout(net, 0.5)

        # map to next value
        net = tflearn.fully_connected(net, len(self.string_to_number)*2, activation='tanh')
        net = tflearn.fully_connected(net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(net)
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        fmt = '%Y-%m-%d_%H-%M-%S'
        now = datetime.datetime.now().strftime(fmt)
        self.model.fit(xs, ys, n_epoch=1, batch_size=1024, show_metric=True, run_id=now)
        self.model.save(model_file)

    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
