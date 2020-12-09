"""
This file is an amalgamation of main.py and api.py
Any changes made to those files needs to be reflected here.
In order for the process to be run by gunicorn it needs access to the Flask app.
"""

from argparse import ArgumentParser
from datetime import date
from dminr_query import SearchTask
from flask import Flask, request, jsonify
from pprint import pprint
import dminr_query
import nyt_query
import relevance
import string
import time
import torch
import training


# make the flask app available to gunicorn
app = Flask(__name__)


# create object to mimic command line arguments
class Args(object):
    pass


def serve():

    # set up required command line arguments
    args = Args()
    args.model_size = 'large'
    args.max_len = 128
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, embeddings, tokenizer = training.load_model(args)

    task = SearchTask(args, model, embeddings, tokenizer)

    @app.route("/connections/predict", methods=['GET'])
    def predict():
        main_tic = time.time()
        query = request.args.get('q', default=None)
        size = request.args.get('size', default='10')
        today = date.today()
        start_date = request.args.get('start_date', default="1900-01-01")
        end_date = request.args.get('end_date', default=today.strftime("%Y-%m-%d"))
        length = request.args.get('length', default='10')

        try:
            out = task.search_funct(
                query,
                size,
                start_date,
                end_date,
                )
            ner_tic = time.time()
        except Exception as e:
            print(e)
            print("result - Import Failed")


        try:
            query_ents = set()
            entity=""
            label = str()
            for i in out:
                currentID = 0

                for j in range(len(i["token"])):
                    if currentID == i["ID"][j]:
                        entity = entity + " " + i["token"][j]
                        label = i["label"][j][2:]
                    else:
                        if entity.strip(string.punctuation) != "":
                            query_ents.add((entity, label))
                        entity = i["token"][j]
                    currentID = i["ID"][j]
                if entity.strip(string.punctuation) != "" :
                    query_ents.add((entity, label))
            query_ents = list(query_ents)
            query_labels = [x[1] for x in query_ents]
            query_entities = [x[0] for x in query_ents]


            corpus = []
            for i in out:
                innercorp = []
                for j in i["token"]:
                    innercorp.append(j)
                corpus.append(innercorp)

            art_rank = relevance.ArticleRanking(corpus)
            rankings = art_rank.get_ranked_entities(query_entities, query_labels)
            now_tic = time.time()
            print("Entities ranked in {} seconds.".format(now_tic-ner_tic))
            print("Total return time: {} seconds.".format(now_tic-main_tic))
            return_len = int(length)
            return_ents = dict()
            for i, entry in enumerate(rankings):
                print(entry)
                return_ents[entry[0]] = [i+1, entry[1], entry[2]]
                if i >= return_len - 1:
                    break

            return jsonify({"result": return_ents})
        except Exception as e:
            print(e)
            return jsonify({"result":"Model Failed"})

serve()


if __name__ == "__main__":
    app.run('0.0.0.0')
