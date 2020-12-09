'''
tmain.py
Author: Tim Attwell
Date: 15/10/2020

Main funtion of entity_connections. 
'''
from argparse import ArgumentParser
import training
import nyt_query
import dminr_query
import torch
from flask import Flask, request, jsonify
import relevance
import string
import time
import api


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help="True = Train, *False = Don't Train")
    parser.add_argument("--classify", type=bool, default=True, help="True = do classification task, *False = Don't do classification task")
    parser.add_argument("--model_size", type=str, default='large', help="""*large = BERT Large
                                                                          base = BERT base""")
    parser.add_argument("--epochs", type=int, default=4, help="Define number of training epochs (*4)")
    parser.add_argument("--bs", type=int, default=32, help="Define batch size (*32)")
    #parser.add_argument("--training_data", type=str, default="./data/ner_dataset.csv", help="""Path to training data.
    #                                                                                         default=./.data/ner_dataset.csv""")
    parser.add_argument("--save_model", type=bool, default=True, help="*True = Save model, False = Do not save model")
    parser.add_argument("--nyt", type=bool, default=False, help="Use the New York Times API")
    parser.add_argument("--local", type=bool, default=False, help="Run search in command line rather than setting up localhost API.")
    
    args = parser.parse_args()

    # Include max sentence length and device in the "args" object 
    # for use in other functions
    args.max_len = 128
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # If --train True flag is called, train the model from scratch.
    # If not, then load the model from pretrained
    if args.train == True:
        model, embeddings, tokenizer = training.build_model(args)
    else:
        model, embeddings, tokenizer = training.load_model(args)

    # If --classify flag is False, then end the program after model training 
    # or loading is complete.
    if args.classify == True:
        # If --local is True, then run the search straight from the command line,
        # but if False, then set up the network accessible API.
        if args.local == True:
            task = dminr_query.SearchTask(args, model, embeddings, tokenizer)
            task.recurrant_search()
        else:
            api.run(args, model, embeddings, tokenizer, port_num=1414)

























