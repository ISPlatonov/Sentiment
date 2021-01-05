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

requests.post('http://localhost:8000/sentiment/scores/best', json={"texts": ["Очень плохо", "Прекрасно", "Благодарю"]})
# {'results': ['negative', 'positive', 'speech']}

requests.post('http://localhost:8000/sentiment/scores', json={"texts": ["Очень плохо", "Прекрасно", "Благодарю"]}).json()
# {'results': [{'negative': 0.90, 'neutral': 0.03, 'positive': 0.02, 'skip': 0.02, 'speech': 0.01, 'humor': 0.02},
#              {'negative': 0.06, 'neutral': 0.05, 'positive': 0.71, 'skip': 0.09, 'speech': 0.04, 'humor': 0.05},
#              {'negative': 0.02, 'neutral': 0.04, 'positive': 0.03, 'skip': 0.06, 'speech': 0.82, 'humor': 0.03}]}
```