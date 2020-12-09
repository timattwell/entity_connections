'''
relavance.py
Author: Tim Attwell
Date: 15/10/2020

Contains all the code involving the relevance ranking. 

Classes and functions of note are expanded on below.
'''
# Takes in a list of a lists containing the named 
# entities from each article, and applies the bm25
# entity ranking model.
# Returns a ranked list of relevant articles
import numpy as np
from rank_bm25 import BM25Plus as BM25
import nltk
from tqdm import tqdm

class ArticleRanking():
    def __init__(self, inputs):
        # inputs = [[entiti es, from, art, one],
        #           [ent, from, ar ticle, two],
        #           [entities, in, docu ment, three]]   
        # Each list of entities is considered a "condensed article" 
        self.inputs = inputs
        self.bm25 = BM25([[x.lower() for x in y] for y in self.inputs])

    # Returns relavance scores of submited entities.
    def get_entity_scores(self, query_set):
        # query = [tokenised, query]
        self.ent_scores = []
        # Loops through the mentioned entities to get its BM25 relevance score for
        # each "condensed" article. 
        query_set = [x.lower() for x in query_set]
        
        for query in tqdm(query_set):
            q = nltk.word_tokenize(query)
            #print(q)
            query_scores = self.bm25.get_scores(q)
            #print(query_scores)
            # The mean of article relevance scores is taken to give a clean final value.
            self.ent_scores.append(np.sum(query_scores)/query_scores.shape[0])
        return self.ent_scores

    # Returns ranked list of entities according to BM25 ranking
    def get_ranked_entities(self, query_set, query_labs):
        self.get_entity_scores(query_set)
        self.ranked_ent_scores = sorted(list(zip(query_set, list(self.ent_scores), query_labs)), key = lambda x: x[1], reverse=True)
        return self.ranked_ent_scores



