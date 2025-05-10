FROM python:3.9

#set envionment variables
ENV PYTHONUNBUFFERED 1

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip
COPY requirements.txt /workdir/
COPY data_load.py /workdir/
WORKDIR /workdir

RUN pip install --no-cache-dir -r requirements.txt