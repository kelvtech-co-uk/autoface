FROM python:latest
WORKDIR /app
COPY ./requirements.txt /app
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 80/tcp
EXPOSE 80/udp
COPY . /app
CMD ["python3","app.py"]