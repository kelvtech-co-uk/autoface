FROM python:slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# CMD curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
# CMD ./script.deb.sh
# RUN apt update && apt install -y git-lfs 
# RUN git clone https://github.com/opencv/opencv_zoo.git
# WORKDIR /app/opencv_zoo
# RUN git lfs install
# RUN git lfs pull
# COPY . /app
# CMD ["python3","app.py"]