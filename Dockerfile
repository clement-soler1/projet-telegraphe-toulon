FROM python:3
ENV PYTHONUNBUFFERED=1
COPY requirements.txt /ceresia_app/requirements.txt
COPY . /ceresia_app
WORKDIR /ceresia_app
RUN pip install -r requirements.txt
EXPOSE 5000