FROM ubuntu:18.10

RUN apt-get update -qqq
RUN apt-get install -y python3-pip

RUN mkdir -p /opt/
COPY requirements.txt /opt/

RUN pip3 install --upgrade pip
RUN pip3 install -r /opt/requirements.txt
COPY Weather_Docker.py /opt/
COPY weatherHistory.csv /opt/
ENTRYPOINT ["python3", "/opt/Weather_20190613.py"]
