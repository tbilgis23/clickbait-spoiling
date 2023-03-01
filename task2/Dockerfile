FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN pip3 install pandas jupyterlab docker datasets transformers

RUN pip3 install sentencepiece

RUN pip3 install protobuf==3.20

RUN apt-get update \
	&& apt-get install -y git-lfs \
    && mkdir /models
RUN cd /models \
    && git lfs install \
    && git clone 'https://huggingface.co/Tugay/clickbait_spoiling_multi' \
	&& cd clickbait_spoiling_multi \
	&& git lfs install \
	&& git fetch \
	&& rm -Rf .git
RUN cd /models \
    && git clone 'https://huggingface.co/Tugay/clickbait_spoiling_passage' \
	&& cd clickbait_spoiling_passage \
	&& git lfs install \
	&& git fetch \
	&& rm -Rf .git
RUN cd /models \
    && git clone 'https://huggingface.co/Tugay/clickbait_spoiling_phrase' \
	&& cd clickbait_spoiling_phrase \
	&& git lfs install \
	&& git fetch \
	&& rm -Rf .git 

COPY task2.py /
RUN chmod 777 /task2.py

ENTRYPOINT [ "/task2.py" ]