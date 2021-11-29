FROM python:3
ENV PYTHONUNBUFFERED=1
COPY requirements.txt /telegraph_app/requirements.txt
COPY . /telegraph_app
WORKDIR /telegraph_app
RUN pip install -r requirements.txt
EXPOSE 5000