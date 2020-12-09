# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW

from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score

from tools import save_pkl, load_pkl

from pprint import pprint

class EntityClassifier():
    def __init__(self, args, model, embeddings, tokenizer):
        self.args = args
        self.tag_values = embeddings[0]
        self.tag2idx = embeddings[1]
        self.model = BertForTokenClassification.from_pretrained('../datalarge/')
        self.model.to(args.device)
        self.tokenizer = tokenizer
        self.saveID = 0

    
    def infer_entities(self, test_sentence):
        tokenized_sentence = self.tokenizer.encode(test_sentence)
        input_ids = torch.tensor([tokenized_sentence]).to(self.args.device)

        with torch.no_grad():
            output = self.model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        # join bpe split tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.tag_values[label_idx])
                new_tokens.append(token)
        
    
        mydict = {"token": [], "ID": [], "label": []}

        self.saveID = self.saveID + 1
        prevlab = "O"
        prev2lab = prevlab
        for token, label in zip(new_tokens, new_labels):
            #print("{}\t{}".format(label, token))
            if label[0] == 'B': 
                self.saveID = self.saveID + 1
                mydict["token"].append(token)
                mydict["ID"].append(self.saveID)
                mydict["label"].append(label)
            
            elif (label[0] == 'I'):
                if (prev2lab == "O") & (prevlab == "O"):
                    self.saveID = self.saveID + 1
                mydict["token"].append(token)
                mydict["ID"].append(self.saveID)
                mydict["label"].append(label)

            prev2lab = prevlab
            prevlab = label[0]
            
        if self.saveID > 2**32:
            self.saveID = 0

        return mydict            