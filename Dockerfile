FROM python:3.8

WORKDIR /bot

COPY requirements.txt ./

RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "use_bot:bot"]
