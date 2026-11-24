# Nguyễn Tâm Bình - 2324801030003
FROM eclipse-temurin:21-jdk-noble

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip unzip curl git && rm -rf /var/lib/apt/lists/*

ARG JOERN_VERSION=v4.0.681
RUN curl -fsSL -o /tmp/joern-cli.zip "https://github.com/joernio/joern/releases/download/${JOERN_VERSION}/joern-cli-linux-x86_64.zip" && unzip -q /tmp/joern-cli.zip -d /opt && rm /tmp/joern-cli.zip && ls /opt
ENV PATH="/opt/joern-cli:/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
