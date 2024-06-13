FROM python:latest
WORKDIR /app
COPY ./requirements.txt /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . /app
CMD ["python3","app.py"]