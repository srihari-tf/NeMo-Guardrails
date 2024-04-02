FROM python:3.10

# Install git, gcc, and g++ using a single RUN statement to reduce layers
RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    g++

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN python -c "from fastembed.embedding import FlagEmbedding; FlagEmbedding('sentence-transformers/all-MiniLM-L6-v2');"

COPY . /code/app
RUN nemoguardrails --help

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "