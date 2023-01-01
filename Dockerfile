FROM python:3.8

WORKDIR /bot

COPY requirements.txt ./



RUN pip install -r requirements.txt

COPY . .


CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 use_bot:bot