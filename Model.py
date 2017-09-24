from itertools import product

import pandas as pd
from StringIO import StringIO

from collections import defaultdict

from columns import *
import numpy as np


# TODO: Multiple Sentences are repeated !
# TODO: ('.', ':') should not be tag transition

class POSModel:
    @staticmethod
    def viterbi_algo(string_tokens, states, emission_prob, tag_transition_prob, word_vocab, start_of_string_state='<s>',
                     end_of_string_state='.'):

        T = len(string_tokens)
        N = len(states)

        viterbi = np.zeros((N + 3, T + 1))
        backpointer = np.zeros((N + 3, T + 1))

        for index, word in enumerate(string_tokens):
            if word not in word_vocab:
                string_tokens[index] = unknown_word

        safe_mul = lambda x, y: np.exp(np.log(x) + np.log(y))

        for index, state in enumerate(states, 1):
            tag_trans = tag_transition_prob[(start_of_string_state, state)]
            emission = emission_prob[(state, string_tokens[0])]
            viterbi[index, 1] = safe_mul(tag_trans, emission)
            backpointer[index, 1] = 0

        for o_i, o in enumerate(string_tokens[1:], 2):
            for s_i, s in enumerate(states, 1):

                for _s_i, _s in enumerate(states, 1):
                    prev = viterbi[_s_i, o_i - 1]
                    tag_trans = tag_transition_prob[(_s, s)]
                    temp = safe_mul(prev, tag_trans)
                    if temp > viterbi[s_i, o_i]:
                        viterbi[s_i, o_i] = temp
                        backpointer[s_i, o_i] = _s_i
                emission = emission_prob[(s, o)]
                viterbi[s_i, o_i] = safe_mul(viterbi[s_i, o_i], emission)

        for s_i, s in enumerate(states, 1):

            tag_trans = tag_transition_prob[(s, end_of_string_state)]
            temp = safe_mul(viterbi[s_i, T], tag_trans)

            if temp > viterbi[N + 1, T]:
                viterbi[N + 1, T] = temp
                backpointer[N + 1, T] = s_i

        ans = list(np.zeros((len(string_tokens) + 1,)))

        z_i = int(backpointer[N + 1, T])
        ans[T] = states[z_i - 1]

        for index in xrange(T, 1, -1):
            z_i = int(backpointer[z_i, index])
            ans[index - 1] = states[z_i - 1]

        ans = ans[1:]
        ans.append(end_of_string_state)
        return ans

    def _reset(self):

        self.train_df_col = None
        self.train_df_comb = None
        self.train_df_raw_sentence = None
        self.dev_df_raw_sentence = None
        self.dev_df_col = None

        self.baseline_model = None

        self.train_word_vocab = None
        self.train_state_vocab = None
        self.train_uni_word_vocab = None

        self.uni_tag_count_dict = None
        self.bi_tag_count_dict = None
        self.tag_and_word_count_dict = None

        self.train_uni_tag_count_dict = None
        self.train_bi_tag_count_dict = None
        self.train_tag_and_word_count_dict = None

        self.prob_tag_transition_dict = None
        self.prob_emission_dict = None

        self.baseline_accuracy = None
        self.predict_accuracy = None

    def __init__(self, **kwargs):
        self.train_dev_split = kwargs.pop('train_dev_split', 0.8)
        self.smoothing = kwargs.pop('smoothing', None)
        self._lambda = kwargs.pop('_lambda', None)
        self.d_kn = kwargs.pop('d_kn', 0.75)

        self.train_df_col = None
        self.train_df_comb = None
        self.train_df_raw_sentence = None
        self.dev_df_raw_sentence = None
        self.dev_df_col = None

        self.baseline_model = None

        self.train_word_vocab = None
        self.train_state_vocab = None
        self.train_uni_word_vocab = None

        self.uni_tag_count_dict = None
        self.bi_tag_count_dict = None
        self.tag_and_word_count_dict = None

        self.train_uni_tag_count_dict = None
        self.train_bi_tag_count_dict = None
        self.train_tag_and_word_count_dict = None

        self.prob_tag_transition_dict = None
        self.prob_emission_dict = None

        self.baseline_accuracy = None
        self.predict_accuracy = None

    def read(self):

        self.train_df_col = self._process_train_raw_sentences_to_col()
        self.dev_df_col = self._process_dev_raw_sentences_to_col()

        df1 = pd.DataFrame(self.train_df_col)
        df2 = df1.drop(df1.index[0]).reset_index(drop=True)

        new = pd.concat([df1, df2], axis=1, join_axes=[df1.index]).dropna()
        new.columns = [word_pos_col + '1',
                       word_col + '1',
                       word_tag_col + '1',
                       word_pos_col + '2',
                       word_col + '2',
                       word_tag_col + '2']

        self.train_df_comb = dict()
        self.train_df_comb[bi_tag_gram_col] = pd.Series(list(zip(new[word_tag_col + '1'], new[word_tag_col + '2'])))

        # Removing (., <s>) case
        mask = self.train_df_comb[bi_tag_gram_col] == (end_of_string_state, start_of_string_state)
        self.train_df_comb[bi_tag_gram_col] = self.train_df_comb[bi_tag_gram_col][~mask]
        self.train_df_comb[bi_tag_gram_col].reset_index(inplace=True, drop=True)

        # Removing (<s>, <.>) case because of empty strings
        mask = self.train_df_comb[bi_tag_gram_col] == (start_of_string_state, end_of_string_state)
        self.train_df_comb[bi_tag_gram_col] = self.train_df_comb[bi_tag_gram_col][~mask]
        self.train_df_comb[bi_tag_gram_col].reset_index(inplace=True, drop=True)

        self.train_df_comb[tag_and_word_gram_col] = pd.Series(
            list(zip(new[word_tag_col + '1'], new[word_col + '1'])))

        # # Removing (<s>, <s> and (., .) case)
        mask = (self.train_df_comb[tag_and_word_gram_col] == (start_of_string_state, start_of_string_state)) | (
            self.train_df_comb[tag_and_word_gram_col] == (end_of_string_state, end_of_string_state))
        self.train_df_comb[tag_and_word_gram_col] = self.train_df_comb[tag_and_word_gram_col][~mask]
        self.train_df_comb[tag_and_word_gram_col].reset_index(inplace=True, drop=True)

        # Choosing vocabulary as tokens with count greater than 1
        self.train_word_vocab = self.train_df_col \
            .groupby(word_col).filter(lambda x: x[word_col].count() > unknown_word_count)[word_col].unique()

        # Removing '.', '<s>' from the train_word_vocab
        mask = (self.train_word_vocab == end_of_string_state) | (self.train_word_vocab == start_of_string_state)
        self.train_word_vocab = self.train_word_vocab[~mask]
        self.train_word_vocab = np.append(self.train_word_vocab, unknown_word)
        self.train_word_vocab.sort()

        # Changing words to unknown_word with count less than 1
        self.train_df_col[word_col] = self.train_df_col[word_col].apply(
            lambda x: unknown_word if x not in self.train_word_vocab and x not in (
                start_of_string_state, end_of_string_state) else x)

        self.train_df_comb[tag_and_word_gram_col] = self.train_df_comb[tag_and_word_gram_col].apply(
            lambda x: x if x[1] in self.train_word_vocab else (x[0], unknown_word))

        self.train_state_vocab = self.train_df_col[word_tag_col].unique()

        # Removing '.', <s> from the train_state_vocab
        mask = (self.train_state_vocab == end_of_string_state) | (self.train_state_vocab == start_of_string_state)
        self.train_state_vocab = self.train_state_vocab[~mask]
        self.train_state_vocab.sort()

        assert len(self.train_state_vocab) == total_number_of_tags

    @staticmethod
    def process_raw_to_col(df):
        return pd.read_csv(
            StringIO(df[sentence_col].str.cat(sep='\n')),
            sep='\t',
            header=None,
            names=[word_pos_col, word_col, word_tag_col]
        ).reset_index(drop=True)

    def _process_train_raw_sentences_to_col(self):
        df = self.train_df_raw_sentence
        return POSModel.process_raw_to_col(df)

    def _process_dev_raw_sentences_to_col(self):
        df = self.dev_df_raw_sentence
        return POSModel.process_raw_to_col(df)

    def get_word_freq_model(self):

        if not self.baseline_model:
            df = self.train_df_col
            self.baseline_model = df.groupby(word_col)[word_tag_col]. \
                agg(lambda x: x.value_counts().index[0])

            # Removing start of the string state and end of the string state from baseline_model
            mask = (self.baseline_model == start_of_string_state) | (self.baseline_model == end_of_string_state)

            self.baseline_model = self.baseline_model[~mask]

            self.baseline_model = self.baseline_model.to_dict()

            # Vocabulary in baseline_model and training dataframe should be same
            assert len(self.baseline_model.keys()) == len(self.train_word_vocab)
        return self.baseline_model

    def _get_uni_tag_count_dict(self):

        if not self.uni_tag_count_dict:
            df = self.train_df_col
            self.uni_tag_count_dict = df[word_tag_col].agg(lambda x: x.value_counts()).to_dict()

            assert len(self.uni_tag_count_dict.keys()) == total_number_of_tags + 2

        return self.uni_tag_count_dict

    def get_uni_tag_count(self, tag):
        return self._get_uni_tag_count_dict()[tag]

    def _get_bi_tag_count_dict(self):

        if not self.bi_tag_count_dict:
            self.bi_tag_count_dict = self.train_df_comb[bi_tag_gram_col].value_counts().to_dict()
        return self.bi_tag_count_dict

    def _get_train_bi_tag_count_dict(self):

        if not self.bi_tag_count_dict:
            self._get_bi_tag_count_dict()

        if not self.train_bi_tag_count_dict:

            self.train_bi_tag_count_dict = self.bi_tag_count_dict.copy()

            for key in product(self.train_state_vocab, self.train_state_vocab):
                if key not in self.train_bi_tag_count_dict:
                    self.train_bi_tag_count_dict[key] = 0.0

            # Adding probability of tag at start of the string
            for key in self.train_state_vocab:
                key = (start_of_string_state, key)
                if key not in self.train_bi_tag_count_dict:
                    self.train_bi_tag_count_dict[key] = 0.0

            # Adding probability of tag at the end of the string
            for key in self.train_state_vocab:
                key = (key, end_of_string_state)
                if key not in self.train_bi_tag_count_dict:
                    self.train_bi_tag_count_dict[key] = 0.0

        return self.train_bi_tag_count_dict

    def get_train_bi_tag_count(self, tag1, tag2):
        return self._get_train_bi_tag_count_dict()[(tag1, tag2)]

    def _get_tag_and_word_count_dict(self):
        if not self.tag_and_word_count_dict:
            self.tag_and_word_count_dict = self.train_df_comb[tag_and_word_gram_col].value_counts().to_dict()

        return self.tag_and_word_count_dict

    def _get_train_tag_and_word_count_dict(self):

        if not self.tag_and_word_count_dict:
            self._get_tag_and_word_count_dict()

        if not self.train_tag_and_word_count_dict:

            self.train_tag_and_word_count_dict = self.tag_and_word_count_dict.copy()

            for tag in self.train_state_vocab:
                for word in self.train_word_vocab:
                    key = (tag, word)
                    if key not in self.train_tag_and_word_count_dict:
                        self.train_tag_and_word_count_dict[key] = 0

        return self.train_tag_and_word_count_dict

    def get_train_tag_and_word_count(self, tag, word):
        return self._get_train_tag_and_word_count_dict()[(tag, word)]

    def _get_kn_smoothing_tag_trans(self):
        if not self.train_bi_tag_count_dict:
            self._get_train_bi_tag_count_dict()

        train_bi_tag_count_dict = self._get_train_bi_tag_count_dict()

        train_bi_tag_count_df = pd.DataFrame.from_dict(train_bi_tag_count_dict.items())
        train_bi_tag_count_df.columns = ['tag1_tag2', 'count']

        train_bi_tag_count_df['tag1'] = train_bi_tag_count_df['tag1_tag2'].apply(lambda x: x[0])
        train_bi_tag_count_df['tag2'] = train_bi_tag_count_df['tag1_tag2'].apply(lambda x: x[1])

        train_state_vocab = list(self.train_state_vocab[:])
        train_state_vocab.append(end_of_string_state)

        prob_cont_df = pd.DataFrame(train_state_vocab, columns=['tag', ])

        # Count of all the bigrams where prob_cont_df.tag has non-zero count as second tag
        prob_cont_df['value'] = prob_cont_df['tag'].apply(lambda x: train_bi_tag_count_df[
                                                                        (train_bi_tag_count_df['tag2'] == x) & (
                                                                            train_bi_tag_count_df['count'] > 0)]
                                                          .shape[0] /
                                                                    (1.0 *
                                                                     len(self.bi_tag_count_dict))
                                                          )
        prob_cont_df = prob_cont_df.set_index(prob_cont_df['tag'], drop=True)
        train_state_vocab = list(self.train_state_vocab[:])
        train_state_vocab.append(start_of_string_state)
        d = self.d_kn
        lambda_df = pd.DataFrame(train_state_vocab, columns=['tag'])
        lambda_df['value'] = lambda_df['tag'].apply(
            lambda x: d * train_bi_tag_count_df[
                (train_bi_tag_count_df['tag1'] == x) & (train_bi_tag_count_df['count'] > 0)].shape[0] /
                      train_bi_tag_count_df[train_bi_tag_count_df['tag1'] == x]['count'].sum()
        )
        lambda_df = lambda_df.set_index(lambda_df['tag'], drop=True)

        prob_tag_transition_df = train_bi_tag_count_df[['tag1', 'tag2']]

        prob_tag_transition_df['value'] = prob_tag_transition_df[['tag1', 'tag2']].apply(
            lambda x: max(1.0 * (train_bi_tag_count_dict[(x[0], x[1])] - d), 0) / self.get_uni_tag_count(x[0]) +
                      lambda_df.ix[
                          x[0]].value *
                      prob_cont_df.ix[
                          x[1]].value,
            axis=1)

        prob_tag_transition_df['tag1_tag2'] = prob_tag_transition_df[['tag1', 'tag2']].apply(lambda x: (x[0], x[1]),
                                                                                             axis=1)
        prob_tag_transition_df.set_index('tag1_tag2', inplace=True)

        return prob_tag_transition_df['value'].to_dict()

    def _get_prob_tag_transition_dict(self):

        if not self.train_bi_tag_count_dict:
            self._get_train_bi_tag_count_dict()

        if not self.prob_tag_transition_dict:
            self.prob_tag_transition_dict = {}
            if self.smoothing == 'kn':
                self.prob_tag_transition_dict = self._get_kn_smoothing_tag_trans()
            else:
                for key, val in self._get_train_bi_tag_count_dict().items():
                    if self.smoothing == 'laplace':
                        self.prob_tag_transition_dict[key] = \
                            (val + 1.0) / (self.get_uni_tag_count(key[0]) + total_number_of_tags + 1)
                    elif self.smoothing == 'lambda':
                        temp = val * 1.0 / self.get_uni_tag_count(key[0])
                        self.prob_tag_transition_dict[key] = self._lambda * temp + (1.0 - self._lambda) * 1.0 / len(
                            self.train_state_vocab)
                    else:
                        self.prob_tag_transition_dict[key] = val * 1.0 / self.get_uni_tag_count(key[0])

        return self.prob_tag_transition_dict

    def get_prob_tag_transition(self, tag1, tag2):
        return self._get_prob_tag_transition_dict()[(tag1, tag2)]

    def _get_kn_smoothing_emission(self):
        pass

    def _get_prob_emission_dict(self):

        if not self.train_tag_and_word_count_dict:
            self._get_train_tag_and_word_count_dict()

        if not self.prob_emission_dict:
            self.prob_emission_dict = {}
            for key, val in self._get_train_tag_and_word_count_dict().items():
                # if self.smoothing:
                #     self.prob_emission_dict[key] = \
                #         (val + 1.0) / (self.get_uni_tag_count(key[0]) + len(self.train_word_vocab))
                # else:
                self.prob_emission_dict[key] = \
                    val * 1.0 / self.get_uni_tag_count(key[0])

        return self.prob_emission_dict

    def get_prob_emission(self, tag, word):
        if word not in self.train_word_vocab:
            word = unknown_word
        return self._get_prob_emission_dict()[(tag, word)]

    def _get_accuracy(self, count1, count2):
        return count1 * 1.0 / count2

    def k_fold_testing(self, k, file_name):
        text = open(file_name).read()
        lines = text.split('\n\n')
        df = pd.DataFrame(lines, columns=[sentence_col, ])
        df[sentence_col] = df[sentence_col]. \
            apply(lambda x: '0\t' + start_of_string_state + '\t' + start_of_string_state + '\n' + x)

        np.random.seed(12345)
        df.apply(np.random.shuffle, axis=0)

        count = df.shape[0]
        test_count = int(count * 1.0 / k) + 1
        accuracy_list = []
        for i in xrange(k):
            print "\nFor Fold %s as dev " % i
            mask = np.zeros(count, dtype=bool)
            if i == k - 1:
                mask[i * test_count:] = True
            else:
                mask[i * test_count: (i + 1) * test_count] = True

            self.train_df_raw_sentence = df[~mask]
            self.dev_df_raw_sentence = df[mask]

            print "Train : ", self.train_df_raw_sentence.shape[0]
            print "Dev : ", self.dev_df_raw_sentence.shape[0]

            self.read()
            self.train()
            accuracy_list.append(self.predict_accuracy)
            self._reset()

        return accuracy_list

    def train(self):
        self._get_prob_tag_transition_dict()
        self._get_prob_emission_dict()

        df = self.dev_df_raw_sentence

        df[dev_string_token_col] = df[sentence_col] \
            .apply(lambda x: [y.split('\t')[1] for y in x.strip().split('\n')][1:-1])

        df[dev_string_tag_col] = df[sentence_col] \
            .apply(lambda x: [y.split('\t')[2] for y in x.strip().split('\n')][1:])

        df[dev_string_predict_col] = df[dev_string_token_col].apply(
            lambda x: self.predict(x) if len(x) > 0 else [end_of_string_state, ])

        df[dev_string_freq_model_col] = df[dev_string_token_col].apply(
            lambda x: self.freq_model_predict(x) if len(x) > 0 else [end_of_string_state, ])

        df[count_freq_match_col] = df.apply(
            lambda x: np.sum(np.array(x[dev_string_tag_col]) == np.array(x[dev_string_freq_model_col])), axis=1)

        df[count_predict_match_col] = df.apply(
            lambda x: np.sum(np.array(x[dev_string_tag_col]) == np.array(x[dev_string_predict_col])), axis=1)

        df[count_dev_tag_col] = df.apply(lambda x: len(x[dev_string_tag_col]), axis=1)

        self.baseline_accuracy = self._get_accuracy(df[count_freq_match_col].sum(), df[count_dev_tag_col].sum())
        self.predict_accuracy = self._get_accuracy(df[count_predict_match_col].sum(), df[count_dev_tag_col].sum())

        print "Baseline accuracy: ", self.baseline_accuracy
        print "HMM accuracy: ", self.predict_accuracy

    def freq_model_predict(self, string_tokens):

        ans = []
        freq_model = self.get_word_freq_model()

        # Removing unknown words
        for index, string in enumerate(string_tokens):
            if string not in self.train_word_vocab:
                string_tokens[index] = unknown_word

        for string in string_tokens:
            tag = freq_model[string]
            ans.append(tag)

        ans.append(end_of_string_state)

        return ans

    def predict(self, string_tokens):
        return POSModel.viterbi_algo(string_tokens,
                                     list(self.train_state_vocab),
                                     self._get_prob_emission_dict(),
                                     self._get_prob_tag_transition_dict(),
                                     list(self.train_word_vocab))
