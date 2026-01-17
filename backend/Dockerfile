FROM python:3.13-slim

WORKDIR /app

RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD [ "uvicorn", "app.main:app","8000","0.0.0.0"]