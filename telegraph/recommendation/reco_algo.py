import pandas as pd
import numpy as np

from gensim.models import Word2Vec


from scipy import spatial
import ast

import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None

#############################
#   Comme la partie d'avant n'est pas fonctionnel, je n'ai pas encore eu le temps de travailler cette partie de code
############################


def normalise_and_recommandation():
    wine_variety_vectors = pd.read_csv('./data/wine_aromas_nonaromas.csv', index_col='Unnamed: 0')

    wine_variety_vectors['weight'] = wine_variety_vectors['weight'].apply(lambda x: 1 - x)
    wine_variety_vectors['acid'] = wine_variety_vectors['acid'].apply(lambda x: 1 - x)
    wine_variety_vectors['salt'] = wine_variety_vectors['salt'].apply(lambda x: 1 - x)
    wine_variety_vectors['bitter'] = wine_variety_vectors['bitter'].apply(lambda x: 1 - x)

    descriptor_frequencies = pd.read_csv('./data/wine_variety_descriptors.csv', index_col='index')
    wine_word2vec_model = Word2Vec.load("./data/food_word2vec_model.bin")
    word_vectors = wine_word2vec_model.wv
    food_nonaroma_infos = pd.read_csv('./data/average_nonaroma_vectors.csv', index_col='Unnamed: 0')

    def minmax_scaler(val, minval, maxval):
        val = max(min(val, maxval), minval)
        normalized_val = (val - minval) / (maxval - minval)
        return normalized_val

    def check_in_range(label_range_dict, value):
        for label, value_range_tuple in label_range_dict.items():
            lower_end = value_range_tuple[0]
            upper_end = value_range_tuple[1]
            if value >= lower_end and value <= upper_end:
                return label
            else:
                continue

    def calculate_avg_food_vec(sample_foods):
        sample_food_vecs = []
        for s in sample_foods:
            sample_food_vec = word_vectors[s]
            sample_food_vecs.append(sample_food_vec)
        sample_food_vecs_avg = np.average(sample_food_vecs, axis=0)
        return sample_food_vecs_avg

    def nonaroma_values(nonaroma, average_food_embedding):
        average_taste_vec = food_nonaroma_infos.at[nonaroma, 'average_vec']
        average_taste_vec = re.sub('\s+', ',', average_taste_vec)
        average_taste_vec = average_taste_vec.replace('[,', '[')
        average_taste_vec = np.array(ast.literal_eval(average_taste_vec))

        similarity = 1 - spatial.distance.cosine(average_taste_vec, average_food_embedding)
        # scale the similarity using our minmax scaler
        scaled_similarity = minmax_scaler(similarity, food_nonaroma_infos.at[nonaroma, 'farthest'],
                                          food_nonaroma_infos.at[nonaroma, 'closest'])
        standardized_similarity = check_in_range(food_weights[nonaroma], scaled_similarity)
        similarity_and_scalar = (scaled_similarity, standardized_similarity)
        return similarity_and_scalar

    def return_all_food_values(sample_foods):
        food_nonaromas = dict()
        average_food_embedding = calculate_avg_food_vec(sample_foods)
        for nonaroma in ['sweet', 'acid', 'salt', 'piquant', 'fat', 'bitter']:
            food_nonaromas[nonaroma] = nonaroma_values(nonaroma, average_food_embedding)
        food_weight = nonaroma_values('weight', average_food_embedding)
        return food_nonaromas, food_weight, average_food_embedding

    food_weights = {
        'weight': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
        'sweet': {1: (0, 0.45), 2: (0.45, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
        'acid': {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
        'salt': {1: (0, 0.3), 2: (0.3, 0.55), 3: (0.55, 0.8), 4: (0.8, 1)},
        'piquant': {1: (0, 0.4), 2: (0.4, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
        'fat': {1: (0, 0.4), 2: (0.4, 0.5), 3: (0.5, 0.6), 4: (0.6, 1)},
        'bitter': {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.65), 4: (0.65, 1)}
    }

    wine_weights = {
        'weight': {1: (0, 0.25), 2: (0.25, 0.45), 3: (0.45, 0.75), 4: (0.75, 1)},
        'sweet': {1: (0, 0.25), 2: (0.25, 0.6), 3: (0.6, 0.75), 4: (0.75, 1)},
        'acid': {1: (0, 0.05), 2: (0.05, 0.25), 3: (0.25, 0.5), 4: (0.5, 1)},
        'salt': {1: (0, 0.15), 2: (0.15, 0.25), 3: (0.25, 0.7), 4: (0.7, 1)},
        'piquant': {1: (0, 0.15), 2: (0.15, 0.3), 3: (0.3, 0.6), 4: (0.6, 1)},
        'fat': {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
        'bitter': {1: (0, 0.2), 2: (0.2, 0.37), 3: (0.37, 0.6), 4: (0.6, 1)}
    }

    def weight_rule(df, food_weight):
        df = df.loc[(df['weight'] >= food_weight[1] - 1) & (df['weight'] <= food_weight[1])]
        return df

    def acidity_rule(df, food_nonaromas):
        df = df.loc[df['acid'] >= food_nonaromas['acid'][1]]
        return df

    def sweetness_rule(df, food_nonaromas):
        df = df.loc[df['sweet'] >= food_nonaromas['sweet'][1]]
        return df

    def bitterness_rule(df, food_nonaromas):
        if food_nonaromas['bitter'][1] == 4:
            df = df.loc[df['bitter'] <= 2]
        return df

    def bitter_salt_rule(df, food_nonaromas):
        if food_nonaromas['bitter'][1] == 4:
            df = df.loc[(df['salt'] <= 2)]
        if food_nonaromas['salt'] == 4:
            df = df.loc[(df['bitter'][1] <= 2)]
        return df

    def acid_bitter_rule(df, food_nonaromas):
        if food_nonaromas['acid'][1] == 4:
            df = df.loc[(df['bitter'] <= 2)]
        if food_nonaromas['bitter'][1] == 4:
            df = df.loc[(df['acid'] <= 2)]
        return df

    def acid_piquant_rule(df, food_nonaromas):
        if food_nonaromas['acid'][1] == 4:
            df = df.loc[(df['piquant'] <= 2)]
        if food_nonaromas['piquant'][1] == 4:
            df = df.loc[(df['acid'] <= 2)]
        return df

    def nonaroma_rules(wine_df, food_nonaromas, food_weight):
        df = weight_rule(wine_df, food_weight)
        list_of_tests = [acidity_rule, sweetness_rule, bitterness_rule, bitter_salt_rule, acid_bitter_rule,
                         acid_piquant_rule]
        for t in list_of_tests:
            df_test = t(df, food_nonaromas)
            if df_test.shape[0] > 5:
                df = t(df, food_nonaromas)
        return df

    def sweet_pairing(df, food_nonaromas):
        if food_nonaromas['sweet'][1] == 4:
            df['pairing_type'] = np.where(
                ((df.bitter == 4) | (df.fat == 4) | (df.piquant == 4) | (df.salt == 4) | (df.acid == 4)), 'contrasting',
                df.pairing_type)
        return df

    def acid_pairing(df, food_nonaromas):
        if food_nonaromas['acid'][1] == 4:
            df['pairing_type'] = np.where(((df.sweet == 4) | (df.fat == 4) | (df.salt == 4)), 'contrasting',
                                          df.pairing_type)
        return df

    def salt_pairing(df, food_nonaromas):
        if food_nonaromas['salt'][1] == 4:
            df['pairing_type'] = np.where(
                ((df.bitter == 4) | (df.sweet == 4) | (df.piquant == 4) | (df.fat == 4) | (df.acid == 4)),
                'contrasting', df.pairing_type)
        return df

    def piquant_pairing(df, food_nonaromas):
        if food_nonaromas['piquant'][1] == 4:
            df['pairing_type'] = np.where(((df.sweet == 4) | (df.fat == 4) | (df.salt == 4)), 'contrasting',
                                          df.pairing_type)
        return df

    def fat_pairing(df, food_nonaromas):
        if food_nonaromas['fat'][1] == 4:
            df['pairing_type'] = np.where(
                ((df.bitter == 4) | (df.sweet == 4) | (df.piquant == 4) | (df.salt == 4) | (df.acid == 4)),
                'contrasting', df.pairing_type)
        return df

    def bitter_pairing(df, food_nonaromas):
        if food_nonaromas['bitter'][1] == 4:
            df['pairing_type'] = np.where(((df.sweet == 4) | (df.fat == 4) | (df.salt == 4)), 'contrasting',
                                          df.pairing_type)
        return df

    def congruent_pairing(pairing_type, max_food_nonaroma_val, wine_nonaroma_val):
        if pairing_type == 'congruent':
            return 'congruent'
        elif wine_nonaroma_val >= max_food_nonaroma_val:
            return 'congruent'
        else:
            return ''

    def congruent_or_contrasting(df, food_nonaromas):

        max_nonaroma_val = max([i[1] for i in list(food_nonaromas.values())])
        most_defining_tastes = [key for key, val in food_nonaromas.items() if val[1] == max_nonaroma_val]
        df['pairing_type'] = ''
        for m in most_defining_tastes:
            df['pairing_type'] = df.apply(lambda x: congruent_pairing(x['pairing_type'], food_nonaromas[m][1], x[m]),
                                          axis=1)

        list_of_tests = [sweet_pairing, acid_pairing, salt_pairing, piquant_pairing, fat_pairing, bitter_pairing]
        for t in list_of_tests:
            df = t(df, food_nonaromas)
        return df

    def sort_by_aroma_similarity(df, food_aroma):

        def nparray_str_to_list(array_string):
            average_taste_vec = re.sub('\s+', ',', array_string)
            average_taste_vec = average_taste_vec.replace('[,', '[')
            average_taste_vec = np.array(ast.literal_eval(average_taste_vec))
            return average_taste_vec

        df['aroma'] = df['aroma'].apply(nparray_str_to_list)
        df['aroma_distance'] = df['aroma'].apply(lambda x: spatial.distance.cosine(x, food_aroma))
        df.sort_values(by=['aroma_distance'], ascending=True, inplace=True)
        return df

    def find_descriptor_distance(word, foodvec):
        descriptor_wordvec = word_vectors[word]
        similarity = 1 - spatial.distance.cosine(descriptor_wordvec, foodvec)
        return similarity

    def most_impactful_descriptors(recommendation):
        recommendation_frequencies = descriptor_frequencies.filter(like=recommendation, axis=0)
        recommendation_frequencies['similarity'] = recommendation_frequencies['descriptors'].apply(
            lambda x: find_descriptor_distance(x, aroma_embedding))
        recommendation_frequencies.sort_values(['similarity', 'relative_frequency'], ascending=False, inplace=True)
        recommendation_frequencies = recommendation_frequencies.head(5)
        most_impactful_descriptors = list(recommendation_frequencies['descriptors'])
        return most_impactful_descriptors


def retrieve_pairing_type_info(wine_recommendations, full_nonaroma_table, pairing_type):
    pairings = wine_recommendations.loc[wine_recommendations['pairing_type'] == pairing_type].head(4)
    wine_names = list(pairings.index)
    return wine_names
