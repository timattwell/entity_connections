# entity_extraction

Named Entity Recognition with Bidirectional Encoder Representations from Transformers, or **NERBERT** uses the bert model to extract entities from articles gathered by a search-engine-like or other document agrigation and search service such as INJECT or Elasticsearch.

Once the entities have been extracted by the model, their relevance to the initial search term is analysised using the BM25+ algorithm, and then are ranked according to this score.

## Usage
A search query goes in, a json comes out with a ranked top ten list of lists containing entity name, entity type and BM25+ score in the `["Results"]` field.

## Installation and running.
1. Clone and then cd to this repo.
    - `git clone https://github.com/timattwell/entity_connections.git`
    - `cd entitity_connections`
2. Install requirements with:
    - `pip -r requirements.txt`
3. Move to `src` directory and run `main.py`.
    - `cd src`
    - `python3 main.py`
4. Send your requests.

## Training
If no pretrained models are provided at install time, then the models will need to be trained. This can be achieved by running `python main.py --train True`. This will train BERT using data from the CONLL2003 entity extraction dataset in ./data/training/ . The default number of training epochs is 4, however this can be changed with the `--epochs` tag. This should do the trick, and after running once should not need to be used again, except in the event of deleting the pretrained model file or using an updated training dataset.

## API calling
The api can be called with several arguments. These are:
- query - Search query (required)
- size - Number of articles included in search (optional - default = 10)
- start_date - Articles published after this date (optional)
- end_date - Articles published before this date (optional - default = current)
- length - top 'x' entities returned (optional - default = 10)

The final url with then be of the form:
`http://localhost:1414/predict?q={}&size={}&start_date={}&end_date={}&length={}`
