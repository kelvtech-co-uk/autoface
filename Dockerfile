FROM python:slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt
#COPY . /app
#CMD ["python3","app.py"]