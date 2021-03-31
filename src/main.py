from typing import List, Dict

from fastapi import FastAPI

from .config import CLASS_DICT
from .sentiment import evaluate_sentiment

app = FastAPI()


@app.post("/sentiment/scores")
def get_score(request: Dict[str, List[str]]):
    return {"results": evaluate_sentiment(request["texts"])}


@app.post("/sentiment/scores/best")
def get_best_class(request: Dict[str, List[str]]):
    results = evaluate_sentiment(request["texts"])
    resp = dict()
    for i in range(len(results)):
        classes = results[i]
        nsubj = classes['nsubj']
        del(classes['nsubj'])
        best_class = max(classes.items(), key=lambda x: x[1])[0]
        print('    results[i]:', results[i])
        print('    results[i]["nsubj"]', results[i]['nsubj'])
        resp[nsubj] = best_class
    return {"results": resp}


# new signal
# to evaluate scores for the texts
# related to given keywords
@app.post("/sentiment/rel/scores")
def get_relative_score(texts: List[str], keywords: List[str]):
    return {"results": evaluate_sentiment(texts, keywords), "keywords": keywords}


# new signal
# to return the most related classes
# for the texts related to given keywords
@app.post("/sentiment/rel/scores/best")
def get_best_relative_class(texts: List[str], keywords: List[str]):
    results = evaluate_sentiment(texts, keywords)
    return {"results": [max(text_res.items(), key=lambda x: x[1])[0] for text_res in results], "keywords": keywords}


@app.get("/sentiment/classes")
def get_classes():
    return {"classes": list(CLASS_DICT.values())}