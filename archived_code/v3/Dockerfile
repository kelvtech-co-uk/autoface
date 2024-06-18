FROM python:slim
WORKDIR /app
COPY ./requirements.txt /app
RUN apt update && apt install -y clinfo intel-opencl-icd
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
# | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
# RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
# RUN apt update && apt install -y intel-basekit
# RUN git clone https://github.com/PyOCL/oclInspector.git
# RUN git clone https://github.com/opencv/opencv_zoo.git
# WORKDIR /app/opencv_zoo
# RUN git lfs install
# RUN git lfs pull
# COPY . /app
# CMD ["python3","app.py"]