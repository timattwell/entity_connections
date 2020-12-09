# entity_extraction

Named Entity Recognition with Bidirectional Encoder Representations from Transformers, or **NERBERT** uses the bert model to extract entities from articles gathered by a search-engine-like or other document agrigation and search service such as INJECT or Elasticsearch.

Once the entities have been extracted by the model, their relevance to the initial search term is analysised using the BM25 algorithm, and then are ranked according to this score.

## Usage
A search query goes in, a json comes out with a ranked top ten greatest hits in a list of lists containing entity name and BM25 score in the `["Results"]` field.

## Installation and running.
1. Clone and then cd to this repo.
2. Use `pip -r requirements.txt` to install requirements
3. cd to src and run `python main.py`.
4. ???
5. Send your requests.

## Training
If no pretrained models are provided at install time, then the models will need to be trained. This can be achieved by running `python main.py --train True`. This will train BERT using data from the CONLL2003 entity extraction dataset in ./data/training/ . The default number of training epochs is 4, however this can be changed with the `--epochs` tag. This should do the trick, and after running once should not need to be used again, except in the event of deleting the pretrained model file or using an updated training dataset.

