from flask import Flask, request, jsonify
import relevance
import string
import time
from dminr_query import SearchTask
from pprint import pprint
from datetime import date

def run(args, model, embeddings, tokenizer, port_num):
    
    task = SearchTask(args, model, embeddings, tokenizer)

    app = Flask(__name__)
    @app.route("/predict",methods=['GET'])
    def predict():
        main_tic = time.time()
        query = request.args.get('q', default=None)
        #page = request.args.get('page', default='1')
        size = request.args.get('size', default='10')
        today = date.today()
        start_date = request.args.get('start_date', default="1900-01-01")
        end_date = request.args.get('end_date', default=today.strftime("%Y-%m-%d"))
        length = request.args.get('length', default='10')
        

        print("{} - {}".format(query, type(query)))
        print("{} - {}".format(length, type(length)))
        print("{} - {}".format(end_date, type(end_date)))
        print("{} - {}".format(start_date, type(start_date)))
        
        #print(type(text))
        try:
            out = task.search_funct(
                query,
                size,
                start_date,
                end_date,
                )
            #print(out)
            #print("^out")
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
                            #print(entity)
                            query_ents.add((entity, label))#.strip(string.punctuation))
                        entity = i["token"][j]
                    currentID = i["ID"][j]
                if entity.strip(string.punctuation) != "" :
                    query_ents.add((entity, label))#.strip(string.punctuation))
            #print(query_ents)
            query_ents = list(query_ents)
            query_labels = [x[1] for x in query_ents]
            query_entities = [x[0] for x in query_ents]


            corpus = []
            for i in out:
                innercorp = []
                for j in i["token"]:    
                    innercorp.append(j)
                corpus.append(innercorp)
            #print(corpus)
            #print("corpus")

            art_rank = relevance.ArticleRanking(corpus)
            rankings = art_rank.get_ranked_entities(query_entities, query_labels)
            now_tic = time.time()
            print("Entities ranked in {} seconds.".format(now_tic-ner_tic))
            print("Total return time: {} seconds.".format(now_tic-main_tic))
            #pprint(rankings)
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

    
    app.run('0.0.0.0',port=port_num)