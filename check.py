import requests
import json


def inject_url(query):
    #start = (page - 1) * page_size
    url = 'http://138.201.139.21/articles?q={}'.format(
        query,
    )
    return url

req = requests.get(inject_url("Trump"))
for i in req.json()["hits"]:
    print(i["body"])



                

