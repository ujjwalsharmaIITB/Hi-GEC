FROM python

RUN apt update && apt upgrade -y && apt install aspell aspell-doc libaspell-dev aspell-hi -y

RUN pip install aspell-python-py3 tqdm numpy 

CMD ["bash"]