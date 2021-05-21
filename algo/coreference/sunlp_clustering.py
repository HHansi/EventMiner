
# modified from https://github.com/SU-NLP/Event-Clustering-within-News-Articles/blob/master/model.py

import pandas as pd


def get_scores(post_val_data, reward, penalty):
    """
    Scoring Algorithm.

    :param post_val_data: dataframe with columns ['id', 'text_a_id', 'text_b_id', 'text_a', 'text_b', 'predictions']
    :param reward:
    :param penalty:
    :return:
    """

    news_scores = {}
    for news_index in post_val_data["id"].unique():

        # Create a dict to store the pairwise predictions.
        news_relationships = {}

        for _, sent_pair in post_val_data[post_val_data["id"] == news_index].iterrows():

            # Get pair IDs and corresponding pairwise prediction.
            sent1_index = sent_pair["text_a_id"]
            sent2_index = sent_pair["text_b_id"]
            prediction = sent_pair["predictions"]

            # Store the predictions in the format: {sent_id1: {another_sent_id1: pairwise_prediction1, another_sent_id2: pairwise_prediction2}}
            if sent1_index in news_relationships:
                news_relationships[sent1_index][sent2_index] = prediction
            else:
                news_relationships[sent1_index] = {sent2_index: prediction}

            # Store the relationships symmetrically
            # Symmetric case would be: {sent_id: {another_sent_id: pairwise_prediction}, {another_sent_id: {sent_id: pairwise_prediction}}}
            if sent1_index < sent2_index:

                if sent2_index in news_relationships:
                    news_relationships[sent2_index][sent1_index] = prediction
                else:
                    news_relationships[sent2_index] = {sent1_index: prediction}

        # Create a dict to store the relationship scores.
        # Neighbor terminology used in this code means that the sentence pairs 'main_key' and 'neigh_key' appear to be in the same cluster.
        final_neighbors = {}

        for main_key, main_neighs in news_relationships.items():

            # The first sentence, say 'main_key'
            final_neighbors[main_key] = {}

            for main_neigh in main_neighs.items():

                # 'main_neigh_key': The second sentence that forms the pair together with the first sentence 'main_key'
                # 'main_pred': Model's prediction for the pair (main_key, main_neigh_key)
                main_neigh_key, main_pred = main_neigh

                # Set initial scores based the pairwise predictions.
                # 1 if they are predicted to be in the same cluster.
                # -1 Otherwise (penalize).
                if main_pred == 1:
                    neighbor_score = 1
                else:
                    neighbor_score = -1

                # Consider common relationships that main_key and main_neigh_key have.
                # Reward their pairwise score if they have common neighbors.
                # Penalize their pairwise if they have neighbors that are not common.
                if main_neigh_key in news_relationships:

                    # Iterate over the neighbors of main_neigh_key (the second sentence)
                    for helper_neighs in news_relationships[main_neigh_key].items():

                        # 'helper_neigh_key': The sentence that forms the pair together with the second sentence 'main_neigh_key'
                        # 'main_pred': Model's prediction for the pair (main_neigh_key, helper_neigh_key)
                        helper_neigh_key, helper_neigh_pred = helper_neighs

                        # Iterate over the neighbors of main_key to see whether it also appears to be in the same cluster with helper_neigh_key
                        for x_neigh_key, x_pred in main_neighs.items():

                            if x_neigh_key == helper_neigh_key:

                                # If main_key (the first sentence) and main_neigh_key (the second sentence) have a common neighbor, reward their pairwise score.
                                # If helper_neigh_key is the neighbor of only one of them (the first or second sentence), penalize the pairwise score of main_key and main_neigh_key.
                                # Otherwise, do nothing since we might not know.
                                if x_pred == 1 and helper_neigh_pred == 1:
                                    neighbor_score += reward
                                elif x_pred == 1 and helper_neigh_pred == 0:
                                    neighbor_score -= penalty
                                elif x_pred == 0 and helper_neigh_pred == 1:
                                    neighbor_score -= penalty

                                break

                # Scores for one news.
                final_neighbors[main_key][main_neigh_key] = neighbor_score

        # Store the scores together with the corresponding news.
        news_scores[news_index] = final_neighbors
    return news_scores


def get_clusters(news_scores):
    '''
    Clustering Algorithm

    '''

    # Example input
    '''
    scores = {2: {4: 1, 27: 0, 36: 2, 37: 0, 40: -6, 43: -4}, 
              4: {2: 1, 27: 0, 36: -1, 37: -1, 40: -3, 43: -3}, 
              27: {2: 0, 4: 0, 36: 0, 37: -2, 40: -4, 43: -2}, 
              36: {2: 2, 4: -1, 27: 0, 37: 1, 40: -5, 43: -3}, 
              37: {2: 0, 4: -1, 27: -2, 36: 1, 40: -4, 43: -5}, 
              40: {2: -6, 4: -3, 27: -4, 36: -5, 37: -4, 43: 0}, 
              43: {2: -4, 4: -3, 27: -2, 36: -3, 37: -5, 40: 0}}
    '''

    news_clusters = {}

    for news_index, scores in news_scores.items():

        column_names = ["Sen_1", "Sen_2", "Score"]
        df = pd.DataFrame(columns=column_names)

        # Create a dataframe of pairwise sentence scores
        for sentence, scores in scores.items():
            # print(scores)
            for key in scores:
                df = df.append(pd.DataFrame({"Sen_1": [sentence], "Sen_2": [key], "Score": [scores[key]]}),
                               ignore_index=True)

        # Sort the dataframe by descending order of score, and the ascending order of sentence 1 and 2
        df.sort_values(by=['Score', 'Sen_1', 'Sen_2'], ascending=[0, 1, 1], inplace=True)

        # Create a sentence list with all currently assigned to group 0
        sentence_ids = df['Sen_1'].tolist()
        sentence_ids.extend(df['Sen_2'].tolist())
        sentences = pd.DataFrame(set(sentence_ids), columns=['Sentences'])

        # sentences = pd.DataFrame(set(df['Sen_1'].tolist()), columns=['Sentences'])
        sentences['Group'] = 0

        # Eliminate all sentence pairs with score <= 0
        df = df[df['Score'] > 0]

        group_count = 0

        if not df.empty:

            # Eliminate duplicate rows
            df['Sen_min'] = df.apply(lambda row: min(row.Sen_1, row.Sen_2), axis=1)
            df['Sen_max'] = df.apply(lambda row: max(row.Sen_1, row.Sen_2), axis=1)
            df.drop(['Sen_1', 'Sen_2'], axis=1, inplace=True)
            df.drop_duplicates(inplace=True)

            # Iterate over the dataframe and assign sentence pairs to groups based on the below conditions:
            # - If the current sentence pair have both Group = 0 (means they've not yet assigned to any group), then create a new group and assign both sentence to this new group
            # - Else if only one of the sentence has Group = 0 in the pair, then that sentence is assigned to the group of the other sentence
            # - Else sentences are already assigned to other groups, then no need to do anything

            for index, row in df.iterrows():
                if sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'].iloc[0] == 0 and \
                        sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'].iloc[0] == 0:
                    group_count = group_count + 1
                    sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'] = group_count
                    sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'] = group_count
                elif sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'].iloc[0] == 0:
                    sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'] = \
                    sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'].iloc[0]
                elif sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'].iloc[0] == 0:
                    sentences.loc[sentences['Sentences'] == row['Sen_max'], 'Group'] = \
                    sentences.loc[sentences['Sentences'] == row['Sen_min'], 'Group'].iloc[0]
                else:
                    pass

        # At the end if there are still sentences that have not been assigned to any group, then assign them to seperate groups individually
        for index, row in sentences.iterrows():
            if row['Group'] == 0:
                group_count = group_count + 1
                sentences.loc[sentences['Sentences'] == row['Sentences'], 'Group'] = group_count

        news_clusters[news_index] = []
        for gr in sentences["Group"].unique():
            news_clusters[news_index].append(sentences[sentences["Group"] == gr]["Sentences"].values.tolist())

    return news_clusters
