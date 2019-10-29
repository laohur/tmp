#filename base.Dockerfile
FROM nvcr.io/nvidia/tensorrtserver:19.09-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y aptitude
RUN apt-get -y install curl libcurl4-openssl-dev
RUN apt-get -y install iputils-ping


#change the time zone
RUN apt-get install tzdata
RUN rm /etc/localtime
RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt-get -y install python3 python3-pip
#TODO update tensorflow2.0

RUN cd /  && mkdir clients && cd clients

RUN wget https://github.com/NVIDIA/tensorrt-inference-server/releases/download/v1.6.0/v1.6.0_ubuntu1804.clients.tar.gz
RUN tar xzf v1.6.0_ubuntu1804.clients.tar.gz

RUN mv -f lib/* /usr/lib
RUN mv -f include/*  /usr/include

RUN pip3 install  --user  --upgrade  python/tensorrtserver-1.6.0-py2.py3-none-linux_x86_64.whl

RUN pip3 install flask requests uwsgi
RUN pip3 install psutil

RUN pip3 install   --user  --upgrade  numpy pillow nvidia-ml-py3

RUN pip3 install tensorflow

RUN pip3 uninstall -y protobuf
RUN pip3 install --no-binary protobuf protobuf
RUN pip3 install protobuf3-to-dict

RUN apt-get -y install redis-server
RUN service redis-server start
# RUN redis-server
RUN pip3 install redis
RUN pip3 install aiohttp


# RUN mkdir /workspace

# COPY . /workspace

WORKDIR /workspace

# RUN chmod +x /workspace/start_tfsavedmodel.sh

# CMD ["/workspace/start_tfsavedmodel.sh"]



ENTRYPOINT ["/bin/bash"]
