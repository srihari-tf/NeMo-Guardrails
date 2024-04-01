FROM python:3.10

# Install git, gcc, and g++ using a single RUN statement to reduce layers
RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    g++


RUN pip install nemoguardrails openai
EXPOSE 8000

COPY ./examples/bots /config


# Download the `all-MiniLM-L6-v2` model using a Python command
RUN python -c "from fastembed.embedding import FlagEmbedding; FlagEmbedding('sentence-transformers/all-MiniLM-L6-v2');"

# Initialize Nemoguardrails to ensure everything is set up correctly
RUN nemoguardrails --help

# Define the default command to run when starting the container
ENTRYPOINT ["nemoguardrails"]
CMD ["server", "--verbose", "--config=/config"]
