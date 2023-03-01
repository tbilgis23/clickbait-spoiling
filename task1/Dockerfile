FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN pip3 install pandas jupyterlab docker datasets transformers

RUN pip3 install sentencepiece

RUN pip3 install protobuf==3.20

RUN apt-get update \
    && apt-get install -y git-lfs \
    && mkdir /model
    
RUN cd /model \
    && git lfs install \
    && git clone 'https://huggingface.co/Tugay/clickbait_spoiling_classification' \
	&& cd clickbait_spoiling_classification \
	&& git lfs install \
	&& git fetch \
	&& rm -Rf .git


COPY task1.py /

RUN chmod 777 /task1.py

ENTRYPOINT [ "/task1.py" ]
