from Model import POSModel
model = POSModel(use_saved_data_frame = True)
model.read(file_name='./berp-POS-training.txt')
model.train()
# import pandas as pd
# import numpy as np
# from StringIO import StringIO
# import sys
# from itertools import permutations
# file_name = './berp-POS-training.txt'
#
#
# most_freq_tag = 'NNP'
#
# train_test_split = float(sys.argv[1]) if len(sys.argv) > 2 else 0.8
#
# start_of_string = '<s>'
#
#
#
# def get_uni_tag_count_dict(df):
#     uni_tag_count_dict = df[word_tag_col].agg(lambda x : x.value_counts()).to_dict()
#     return uni_tag_count_dict
#
# def get_bi_tag_count_dict(df):
#     t1_t2_df = explode_column(df, bi_tag_gram_col, ['t1', 't2'])
#     return t1_t2_df.groupby(['t1', 't2']).size().to_dict()
#
# def get_tag_word_count_dict(df):
#     t1_t2_df = explode_column(df, word_tag_gram_col, ['t1', 't2'])
#     return t1_t2_df.groupby(['t1', 't2']).size().to_dict()
#
# def get_word_freq_model(df):
#     return df \
#             .groupby(word_col)[word_tag_col]\
#             .agg(lambda x : x.value_counts().index[0]).to_dict()
#
#
# def process_raw_sentences_to_comb(df):
#     df['sentence_object'] = df.loc[: , sentence_col].\
#                                 apply(lambda x : Sentence(x))
#
#     df = df.drop(sentence_col, axis = 1)
#     df = pd.concat([df,
#                       df.sentence_object.apply(
#                           lambda x : pd.Series(
#                               {
#                                   uni_tag_gram_col : x.get_uni_tag(),
#                                   uni_word_gram_col : x.get_word(),
#                                   bi_tag_gram_col : x.get_bi_tag(),
#                                   word_tag_gram_col : x.get_tag_word()
#                               })
#                       )], axis = 1)\
#                       .drop('sentence_object', axis = 1)
#     return df.reset_index(drop = True)
#
# def process_raw_sentences_to_col(df):
#     return pd.read_csv(
#                 StringIO(df[sentence_col].str.cat(sep = '\n')),
#                 sep = '\t',
#                 header = None,
#                 names = [word_pos_col, word_col, word_tag_col]
#             ).reset_index(drop = True)
#
# def get_df_from_file():
#     text = open(file_name).read()
#     lines = text.split("\n\n")
#     df = pd.DataFrame(lines, columns = [sentence_col, ])
#     msk = np.random.rand(len(df)) < train_test_split
#
#     train_df = df[msk]
#     test_df = df[~msk]
#
#     train_df_comb = process_raw_sentences_to_comb(train_df)
#     train_df_col = process_raw_sentences_to_col(train_df)
#
#     return train_df_comb, train_df_col, test_df
#
# def predict(test_df, f):
#     test_df = process_raw_sentences_to_col(test_df)
#     test_df[predicted_tag] = test_df[word_col].apply(f)
#     return 1.0 - test_df[test_df[predicted_tag] != test_df[word_tag_col]].count()[0]*1.0/test_df.count()[0]
#
# def explode_column(df, column_to_explode, result_cols):
#     return pd.DataFrame(
#         df[column_to_explode].apply(pd.Series) \
#         .stack()\
#         .rename(column_to_explode) \
#         .to_frame() \
#         .reset_index(drop = True)[column_to_explode].tolist(),
#         columns = result_cols
#     )
#
# def fill_bi_tag_count_dict(sparse_dict, uniq_tags):
#
#     # Adding all the bigrams that didn't exist in the corpuse
#     for key in permutations(uniq_tags, 2):
#         sparse_dict[key] = 0 if key not in sparse_dict else sparse_dict[key]
#
#     for key in uniq_tags:
#         key = (start_of_string, key)
#         sparse_dict[key] = 0 if key not in sparse_dict else sparse_dict[key]
#
#
# def fill_tag_word_count_dict(sparse_dict, uniq_tags, uniq_words):
#     for tag in uniq_tags:
#         for word in uniq_words:
#             key = (tag, word)
#             sparse_dict[key] = sparse_dict[key] if key in sparse_dict else 0
# # def main():
#
# def get_prob_transition_dict(bi_tag_count, uni_tag_count):
#     prob = {}
#     for key, val in bi_tag_count.items():
#         prob[key] = val*1.0/uni_tag_count[key[0]]
#     return prob
#
# def get_prob_emission_dict(tag_word_count, uni_tag_count):
#     prob = {}
#     for key,val in tag_word_count.items():
#         prob[key] = val * 1.0 / uni_tag_count[key[0]]
#     return prob
#
# train_df_comb, train_df_col, test_df = get_df_from_file()
#
# word_freq_tag_model = get_word_freq_model(train_df_col)
#
# f = lambda x : word_freq_tag_model[x] if x in word_freq_tag_model else most_freq_tag
#
# print "Base Line Accuracy", predict(test_df, f)
#
# uni_tag_count_dict = get_uni_tag_count_dict(train_df_col)
# bi_tag_count_dict = get_bi_tag_count_dict(train_df_comb)
#
# uniq_tags = uni_tag_count_dict.keys()
#
# # Dictionary is passed by reference
# fill_bi_tag_count_dict(bi_tag_count_dict, uniq_tags)
#
# tag_word_count_dict = get_tag_word_count_dict(train_df_comb)
# uniq_words = train_df_col[word_col].unique().tolist()
# fill_tag_word_count_dict(tag_word_count_dict, uniq_tags, uniq_words)
#
# #Number of training
# uni_tag_count_dict[start_of_string] = 15865 - pd.Series(test_df).count()
#
# prob_tag_transition_dict = get_prob_transition_dict(bi_tag_count_dict, uni_tag_count_dict);
# prob_emission_dict = get_prob_emission_dict(tag_word_count_dict, uni_tag_count_dict)
#
# model = POSModel(prob_tag_transition_dict, prob_emission_dict, uniq_tags)
