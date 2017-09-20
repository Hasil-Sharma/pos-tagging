from itertools import product

import pandas as pd
from StringIO import StringIO
from columns import *
import numpy as np


# TODO: Investigate reason from ('<s>', '.'): 1 (bi_tag)
# TODO: Multiple Sentences are repeated !
# TODO: ('.', ':') should not be tag transition

class Sentence:
    def __init__(self, string):
        self.string = pd.read_csv(StringIO(string),
                                  sep='\t',
                                  header=None,
                                  names=[word_pos_col, word_col, word_tag_col])

        self.uni_tag_list = self.string[word_tag_col].tolist()
        self.uni_word_list = self.string[word_col].tolist()
        assert len(self.uni_word_list) == len(self.uni_tag_list)

    def get_uni_tag(self):
        return self.uni_tag_list

    def get_word(self):
        return self.uni_word_list

    def get_bi_tag(self):
        temp_list = self.uni_tag_list
        return [value for value in zip(temp_list, temp_list[1:])]

    def get_tag_word(self):
        return zip(self.uni_tag_list[1:-1], self.uni_word_list[1:-1])


class POSModel:
    @staticmethod
    def viterbi_algo(string_tokens, states, emission_prob, tag_transition_prob, word_vocab, start_of_string_state='<s>',
                     end_of_string_state='.'):

        T = len(string_tokens)
        N = len(states)

        viterbi = np.zeros((N + 2, T))
        backpointer = np.zeros((N + 2, T))

        for index, word in enumerate(string_tokens):
            if word not in word_vocab:
                string_tokens[index] = unknown_word

        safe_mul = lambda x, y: np.exp(np.log(x) + np.log(y))

        for index, state in enumerate(states, 1):
            viterbi[index, 0] = safe_mul(tag_transition_prob[(start_of_string_state, state)],
                                         emission_prob[(state, string_tokens[0])])
            backpointer[index, 0] = index

        for o_i, o in enumerate(string_tokens[1:], 1):
            for s_i, s in enumerate(states, 1):

                for _s_i, _s in enumerate(states, 1):

                    temp = safe_mul(viterbi[_s_i, o_i - 1], tag_transition_prob[(_s, s)])

                    if temp > viterbi[s_i, o_i]:
                        viterbi[s_i, o_i] = temp
                        backpointer[s_i, o_i] = _s_i

                viterbi[s_i, o_i] = safe_mul(viterbi[s_i, o_i], emission_prob[(s, o)])

        for s_i, s in enumerate(states, 1):

            temp = safe_mul(viterbi[s_i, T - 1], tag_transition_prob[(s, '.')])

            if temp > viterbi[N + 1, T - 1]:
                viterbi[N + 1, T - 1] = temp
                backpointer[N + 1, T - 1] = s_i

        ans = [end_of_string_state, ]

        s_i = N + 1
        t_i = T - 1

        while True:
            b_i = int(backpointer[s_i, t_i])
            if t_i == -1:
                break

            ans.append(states[b_i - 1])
            s_i = b_i
            t_i = t_i - 1

        return ans[::-1]

    # @staticmethod
    # def _explode_column(df, column_to_explode, result_cols):
    #     return pd.DataFrame(
    #         df[column_to_explode]
    #             .apply(pd.Series).stack()
    #             .rename(column_to_explode)
    #             .to_frame()
    #             .reset_index(drop=True)[column_to_explode].tolist(),
    #         columns=result_cols
    #     )

    def __init__(self, **kwargs):
        self.train_test_split = kwargs.pop('train_test_split', 0.8)

        self.train_df_col = None
        self.train_df_comb = None
        self.train_df_raw_sentence = None
        self.test_df_raw_sentence = None
        self.test_df_col = None

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

    def read(self, file_name):

        text = open(file_name).read()
        lines = text.split('\n\n')
        df = pd.DataFrame(lines, columns=[sentence_col, ])
        df[sentence_col] = df[sentence_col]. \
            apply(lambda x: '0\t' + start_of_string_state + '\t' + start_of_string_state + '\n' + x)

        np.random.seed(2343)
        msk = np.random.rand(len(df)) < self.train_test_split

        self.train_df_raw_sentence = df[msk]
        self.test_df_raw_sentence = df[~msk]

        self.train_df_col = self._process_train_raw_sentences_to_col()
        self.test_df_col = self._process_test_raw_sentences_to_col()

        df1 = pd.DataFrame(self.train_df_col)
        df2 = df1.drop(df1.index[0]).reset_index(drop=True)

        new = pd.concat([df1, df2], axis=1, join_axes=[df1.index]).dropna()
        new.columns = [word_pos_col + '1',
                       word_col + '1',
                       word_tag_col + '1',
                       word_pos_col + '2',
                       word_col + '2',
                       word_tag_col + '2', ]

        self.train_df_comb = dict()
        self.train_df_comb[bi_tag_gram_col] = pd.Series(list(zip(new[word_tag_col + '1'], new[word_tag_col + '2'])))

        # Removing (., <s>) case
        mask = self.train_df_comb[bi_tag_gram_col] == (end_of_string_state, start_of_string_state)
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
            .groupby(word_col).filter(lambda x: x[word_col].count() > 1)[word_col].unique()

        # Removing '.' and '<s>' from the train_word_vocab
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

    # def _process_train_raw_sentences_to_comb(self):
    #
    #     df = self.train_df_raw_sentence.copy()
    #     df['sentence_object'] = df.loc[:, sentence_col]. \
    #         apply(lambda x: Sentence(x))
    #
    #     df = df.drop(sentence_col, axis=1)
    #     df = pd.concat([df, df.sentence_object.apply(
    #         lambda x: pd.Series(
    #             {
    #                 uni_tag_gram_col: x.get_uni_tag(),
    #                 uni_word_gram_col: x.get_word(),
    #                 bi_tag_gram_col: x.get_bi_tag(),
    #                 tag_and_word_gram_col: x.get_tag_word()
    #             })
    #     )], axis=1) \
    #         .drop('sentence_object', axis=1)
    #
    #     return df.reset_index(drop=True)

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

    def _process_test_raw_sentences_to_col(self):
        df = self.test_df_raw_sentence
        return POSModel.process_raw_to_col(df)

    def get_word_freq_model(self):

        # TODO: Add handling of UNKOWN tag for unkown words in test data

        if not self.baseline_model:
            df = self.train_df_col
            self.baseline_model = df.groupby(word_col)[word_tag_col]. \
                agg(lambda x: x.value_counts().index[0]).to_dict()

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
            # df = self.train_df_comb
            # t1_t2_df = POSModel._explode_column(df, bi_tag_gram_col, ['t1', 't2'])
            # self.bi_tag_count_dict = t1_t2_df.groupby(['t1', 't2']).size().to_dict()
            self.bi_tag_count_dict = self.train_df_comb[bi_tag_gram_col].value_counts().to_dict()
        return self.bi_tag_count_dict

    def _get_train_bi_tag_count_dict(self, smoothing=None):

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

            if smoothing is None:
                pass
        return self.train_bi_tag_count_dict

    def get_train_bi_tag_count(self, tag1, tag2):
        return self._get_train_bi_tag_count_dict()[(tag1, tag2)]

    def _get_tag_and_word_count_dict(self):
        if not self.tag_and_word_count_dict:
            # df = self.train_df_comb
            # t1_t2_df = POSModel._explode_column(df, tag_and_word_gram_col, ['t1', 't2'])
            # self.tag_and_word_count_dict = t1_t2_df.groupby(['t1', 't2']).size().to_dict()
            self.tag_and_word_count_dict = self.train_df_comb[tag_and_word_gram_col].value_counts().to_dict()

        return self.tag_and_word_count_dict

    def _get_train_tag_and_word_count_dict(self, smoothing=None):

        if not self.tag_and_word_count_dict:
            self._get_tag_and_word_count_dict()

        if not self.train_tag_and_word_count_dict:

            self.train_tag_and_word_count_dict = self.tag_and_word_count_dict.copy()

            for tag in self.train_state_vocab:
                for word in self.train_word_vocab:
                    key = (tag, word)
                    if key not in self.train_tag_and_word_count_dict:
                        self.train_tag_and_word_count_dict[key] = 0

            if smoothing is None:
                pass
        return self.train_tag_and_word_count_dict

    def get_train_tag_and_word_count(self, tag, word):
        return self._get_train_tag_and_word_count_dict()[(tag, word)]

    def _get_prob_tag_transition_dict(self):

        if not self.train_bi_tag_count_dict:
            self._get_train_bi_tag_count_dict()

        if not self.prob_tag_transition_dict:
            self.prob_tag_transition_dict = {}
            for key, val in self._get_train_bi_tag_count_dict().items():
                self.prob_tag_transition_dict[key] = \
                    val * 1.0 / self.get_uni_tag_count(key[0])

        return self.prob_tag_transition_dict

    def get_prob_tag_transition(self, tag1, tag2):
        return self._get_prob_tag_transition_dict()[(tag1, tag2)]

    def _get_prob_emission_dict(self):

        if not self.train_tag_and_word_count_dict:
            self._get_train_tag_and_word_count_dict()

        if not self.prob_emission_dict:
            self.prob_emission_dict = {}
            for key, val in self._get_train_tag_and_word_count_dict().items():
                self.prob_emission_dict[key] = \
                    val * 1.0 / self.get_uni_tag_count(key[0])

        return self.prob_emission_dict

    def get_prob_emission(self, tag, word):
        if word not in self.train_word_vocab:
            word = unknown_word
        return self._get_prob_emission_dict()[(tag, word)]

    def train(self):

        self._get_prob_tag_transition_dict()
        self._get_prob_emission_dict()

        # import ipdb
        # ipdb.set_trace(context=5)

        df = self.test_df_raw_sentence.head(20)

        df[test_string_tag_col] = df[sentence_col] \
            .apply(lambda x: [y.split('\t')[1] for y in x.split('\n')][1:-1])

        df[test_string_token_col] = df[sentence_col] \
            .apply(lambda x: [y.split('\t')[2] for y in x.split('\n')][1:])

        df[test_string_predict_col] = df[test_string_tag_col].apply(lambda x: self.predict(x))

        return df

    def predict(self, string_tokens):
        return POSModel.viterbi_algo(string_tokens,
                                     list(self.train_state_vocab),
                                     self._get_prob_emission_dict(),
                                     self._get_prob_tag_transition_dict(),
                                     list(self.train_word_vocab))
