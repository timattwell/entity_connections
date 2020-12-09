# entity_extraction

Named Entity Recognition with Bidirectional Encoder Representations from Transformers, or **NERBERT** uses the bert model to extract entities from articles gathered by a search-engine-like or other document agrigation and search service such as INJECT or Elasticsearch.

Once the entities have been extracted by the model, their relevance to the initial search term is analysised using the BM25+ algorithm, and then are ranked according to this score.

## Usage
A search query goes in, a json comes out with a ranked top ten list of lists containing entity name, entity type and BM25+ score in the `["Results"]` field.

## Installation and running.
1. Clone and then cd to this repo.
    - `git clone https://github.com/DMINR-City/entity_connections.git`
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



---

## Deployment

It will help to review the [DMINR-backend](https://github.com/DMINR-City/DMINR-backend) and [DMINR-frontend](https://github.com/DMINR-City/DMINR-frontend) deployment guides, as this repo uses a similar same process, and assumes you have already set up the backend and frontend repos on the server already.

See also [this guide](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-20-04) for deploying a Flask application to an Ubuntu server.

- Create an ssh key on the server:

  ```
  sudo ssh-keygen
  ```

  Follow the steps. It is worth changing the default key path so that you can create a unique ssh key for each repo, e.g. change the path to `/root/.ssh/id_rsa_connections`

  Don't enter a passphrase.

  Once this is complete, update the config file: `/root/.ssh/config`:

  ```
  Host          dminr_connections
  HostName      github.com
  User          git
  IdentityFile  /root/.ssh/id_rsa_connections
  ```

  Next, copy the public key from `/root/.ssh/id_rsa_connections.pub` and [add this to the GitHub repository as a deploy key](https://github.com/DMINR-City/entity_connections/settings/keys)

  This will allow you to clone this private repo on the server, using the config name.

  (This is the same process as the backend and frontend repos)

- Navigate to the existing directory for the website files:

  ```
  cd /var/www/dminr.city.ac.uk
  ```

- Clone this repo onto the server:

  ```
  git clone dminr_connections:DMINR-City/entity_connections.git connections
  ```

- Create and activate a virtualenv for the project:

  ```
  cd connections
  virtualenv env
  source env/bin/activate
  ```

- Install the project dependencies:

  ```
  pip install -r requirements.txt
  pip install gunicorn
  ```

- If the model isn't already trained, do so (see notes above deployment guide)

- Set up a new gunicorn service based on [these instructions](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-20-04).

   For example you may create this file:

  ```
  sudo nano /etc/systemd/system/gunicorn_dminr_connections.service
  ```

   Its contents may be:

  ```
  [Unit]
  Description=gunicorn dminr connections daemon
  After=network.target

  [Service]
  User=your_username
  Group=www-data
  WorkingDirectory=/var/www/dminr.city.ac.uk/connections/src
  ExecStart=/var/www/dminr.city.ac.uk/connections/env/bin/gunicorn --workers 3 --bind unix:/path/to/gunicorn_dminr_connections.sock -m 007 wsgi:app

  [Install]
  WantedBy=multi-user.target
  ```

  Follow the instructions to enable this service.

- Set up a path to this API in the existing nginx config for the site, for example:

  ```
  location /connections {
    include proxy_params;
    proxy_pass http://unix:/path/to/gunicorn_dminr_connections.sock>
  }
  ```

The API should now be available publicly at `http://[URL]/connections/predict?q=`


### Updating the app

If updates are made to this repo, to update the server you need to run the following commands:

```
cd /var/www/dminr.city.ac.uk/connections
source env/bin/activate
git pull -p
sudo systemctl restart gunicorn_dminr_connections
```