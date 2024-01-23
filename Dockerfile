FROM python:3.9.1
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt