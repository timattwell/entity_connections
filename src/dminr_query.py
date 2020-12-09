'''
dminr_query.py
Author: Tim Attwell
Date: 15/10/2020

Contains all the code involving the aquisition and processing of data from INJECT.
The main tasks are this:
1) To query INJECT and recieve back a JSON of data, including article body text.
2) Split up the articles into sentences using nltk and send these sentences to nerbert.
3) Recieve the raw entities and tags, and process them so multi-word entities are reattached.

Classes and functions of note are expanded on below.
'''
import requests
import json
import nltk
from tqdm import tqdm 
from ner_bert import EntityClassifier
import time

'''
class SearchTask - everything invoving the INJECT search and extracted entity processing.
    
    fn __init__() - the nltk tokeniser and nerbert are loaded at class initialisation 
    so individual search_funct() calls do not have to load it every time.

    fn search_funct() - inject is queried and the articles in the returned JOSN, req, are 
    looped through. The article body is tokenised into sentences and the entities extracted 
    using nerbert. Multi-word entities can be attached using the 'B'(Start) and 'I'(Continued) 
    pre-tags. The entities are then saved into a list of lists corresponding to the article 
    they were found in. 
'''
class SearchTask():
    def __init__(self, args, model, embeddings, tokenizer):
        nltk.download('punkt')
        self.ent_clas = EntityClassifier(args, model, embeddings, tokenizer)

    def recurrant_search(self):
        #search_ents = {"token": [], "label": []}
        while True:
            srch_trm = input("Enter search term(with +'s): ")   
            if srch_trm == "q":
                break
            search_ents = self.search_funct(srch_trm)
            print(search_ents)

    def query_inject_url(self, query, size, start_date, end_date):
        page_size = int(size)
        page = 1
        start = (page - 1) * page_size
        url = 'http://138.201.139.21/articles?q={}&published_before={}&published_after={}&lang={}&size={}&offset={}'.format(
            query,
            end_date,
            start_date,
            'en',
            page_size,
            start,
        )
        #print(url)
        return url

    def search_funct(self, query, size, start_date, end_date, per_art=True):
        search_tic = time.time()
        
        req = requests.get(self.query_inject_url(query, size, start_date, end_date))
        
        print(req)
        print("Articles returned in {} seconds.".format(time.time()-search_tic))
        #print(req.json())
        ent_tic = time.time()
        if per_art == True:
            self.ent_per_art = []
        self.entities = {"token": [], "ID": [], "label": []}
        for article in tqdm(req.json()['hits']):
            #print("Title: " + article["title"])
            #print("  URL: " + article["url"])
            #print("   ID: " + article["id"])
            #res = requests.get(url="http://localhost:9200/_search?q=" +
            #                article["id"])
            try:
                sent = article['body']
                art_list = nltk.tokenize.sent_tokenize(sent)
                art_ents = {"token": [], "ID": [], "label": []}
                for sentence in tqdm(art_list):
                    if len(sentence) > 512:
                        sentence = sentence[:512]
                    new_ents = self.ent_clas.infer_entities(sentence)
                    #print(new_ents)
                    #print("hmm")
                    art_ents["token"].extend(new_ents["token"])
                    art_ents["ID"].extend(new_ents["ID"])
                    art_ents["label"].extend(new_ents["label"])
                    #print(art_ents)
                self.entities["token"].extend(art_ents["token"])
                self.entities["ID"].extend(art_ents["ID"])
                self.entities["label"].extend(art_ents["label"])
                #print(self.entities)
                if per_art == True:
                    self.ent_per_art.append(self.entities)
                    #print(self.ent_per_art)
            except:
                print(req)

        if per_art == True:
            self.entities = self.ent_per_art 
        #rint(self.entities)  

        print("Entities extracted in {} seconds.".format(time.time()-ent_tic))

        return self.entities

