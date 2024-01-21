FROM python:3.9.1
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt