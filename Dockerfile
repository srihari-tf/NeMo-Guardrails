# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.10

# Install git, gcc, and g++ using a single RUN statement to reduce layers
RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    g++

# Set the working directory to /nemoguardrails
WORKDIR /nemoguardrails

# Copy the current directory contents into the container at /nemoguardrails
COPY . /nemoguardrails

# Install NeMo Guardrails with all dependencies
RUN pip install -e .[all]

# Remove the PIP cache to reduce image size
RUN rm -rf /root/.cache/pip

# Make port 8000 available to the world outside this container
EXPOSE 8000

# We copy the example bot configurations into /config
WORKDIR /config
COPY ./examples/bots /config

# Change the work directory back to /nemoguardrails
WORKDIR /nemoguardrails

# Download the `all-MiniLM-L6-v2` model using a Python command
RUN python -c "from fastembed.embedding import FlagEmbedding; FlagEmbedding('sentence-transformers/all-MiniLM-L6-v2');"

# Initialize Nemoguardrails to ensure everything is set up correctly
RUN nemoguardrails --help

# Define the default command to run when starting the container
ENTRYPOINT ["/usr/local/bin/nemoguardrails"]
CMD ["server", "--verbose", "--config=/config"]
