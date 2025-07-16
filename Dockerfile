FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py .

RUN chown -R 1000:1000 /app

CMD ["python", "main.py"]
