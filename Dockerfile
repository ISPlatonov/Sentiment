FROM python:3.7.3

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /code

RUN wget http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz && \
    mkdir model_comments && tar -xvzf elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz -C model_comments && \
    rm elmo_ru-twitter_2013-01_2018-04_600k_steps.tar.gz && \
    wget http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz && \
    mkdir model_posts && tar -xvzf elmo_ru-news_wmt11-16_1.5M_steps.tar.gz -C model_posts && \
    rm elmo_ru-news_wmt11-16_1.5M_steps.tar.gz && \
    wget https://www.dropbox.com/s/rcbcbv4aw2igsvx/comments_cnn_model_v3.h5?dl=1 -O comments_cnn_model.h5 && \
    wget https://www.dropbox.com/s/0b0f922m1ck5djc/news_cnn_model_v3.h5?dl=1 -O news_cnn_model.h5 && \
    wget https://www.dropbox.com/s/sk8evplyc5pj2qq/text_type_model_v2.pkl?dl=1 -O text_type_model.pkl

COPY requirements.txt /code

RUN pip install -r requirements.txt
RUN python -m deeppavlov install rusentiment_elmo_twitter_cnn
RUN pip install scikit-learn==0.22.2.post1 && pip install fastapi==0.63.0
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt stopwords perluniprops nonbreaking_prefixes

ENV TZ=Europe/Moscow
COPY . /code

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0"]