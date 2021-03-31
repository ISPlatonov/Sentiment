# sentiment

### Quick start
1. Clone project
2. Build Dockerfile
```
cd sentiment/
docker build -t sentiment .
docker run -p 8000:8000 --name sentiment sentiment
```

### Usage

```
import requests

requests.get('http://localhost:8000/sentiment/classes').json()
# {'classes': ['negative', 'neutral', 'positive', 'skip', 'speech', 'humor']}

post(url='http://localhost:8000/sentiment/scores/best', json={'texts': ['Маша ела кашу. Вася пил сок.']}).json()
# {'results': {'маша': 'neutral', 'вася': 'neutral'}}

post(url='http://localhost:8000/sentiment/scores', json={'texts': ['Маша ела кашу. Вася пил сок.']}).json()
# {'results': [{'negative': 0.07681494206190109, 'neutral': 0.5154229402542114, 'positive': 0.15101037919521332, 'skip': 0.12285768985748291, 'speech': 0.06735800951719284, 'humor': 0.0665360689163208, 'nsubj': ['маша']}, 
#              {'negative': 0.06391550600528717, 'neutral': 0.6007885932922363, 'positive': 0.04485698789358139, 'skip': 0.18244265019893646, 'speech': 0.057975299656391144, 'humor': 0.05002095177769661, 'nsubj': ['вася']}]}

# requests with keywords
requests.post('http://localhost:8000/sentiment/rel/scores/best', json={"texts": ["Очень плохо", "Прекрасно", "Благодарю", "Баба Капа покормила Лунтика", "Волга стала очень грязной"], "keywords": ["Баба Капа", "Волга"]}).json()
# {'results': ['skip', 'skip', 'skip', 'neutral', 'negative'], 
# 'keywords': ['Баба Капа', 'Волга']}

requests.post('http://localhost:8000/sentiment/rel/scores', json={"texts": ["Очень плохо", "Прекрасно", "Благодарю", "Баба Капа покормила Лунтика", "Волга стала очень грязной"], "keywords": ["Баба Капа", "Волга"]}).json()
# {'results': [{'negative': 0.082, 'neutral': 0.338, 'positive': 0.074, 'skip': 0.398, 'speech': 0.058, 'humor': 0.046}, 
#              {'negative': 0.082, 'neutral': 0.338, 'positive': 0.074, 'skip': 0.398, 'speech': 0.058, 'humor': 0.046},  
#              {'negative': 0.082, 'neutral': 0.338, 'positive': 0.074, 'skip': 0.398, 'speech': 0.058, 'humor': 0.046}, 
#              {'negative': 0.055, 'neutral': 0.545, 'positive': 0.123, 'skip': 0.168, 'speech': 0.043, 'humor': 0.064}, 
#              {'negative': 0.474, 'neutral': 0.161, 'positive': 0.057, 'skip': 0.115, 'speech': 0.053, 'humor': 0.137}], 
# 'keywords': ['Баба Капа', 'Волга']}
```