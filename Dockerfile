# Dockerfile
FROM mambaorg/micromamba:1.5.8 AS base
WORKDIR /srv
COPY conda_environment.yml /tmp/conda_environment.yml
RUN micromamba create -y -n housing -f /tmp/conda_environment.yml && \
    micromamba clean --all --yes
SHELL ["micromamba", "run", "-n", "housing", "/bin/bash", "-lc"]

FROM base AS app
WORKDIR /srv
COPY . /srv

# Ensure the runtime user owns the working dir
USER root
# MAMBA_USER is defined in the base image; fall back to 1000 if not.
RUN chown -R ${MAMBA_USER:-1000}:${MAMBA_USER:-1000} /srv
USER ${MAMBA_USER:-1000}

EXPOSE 8080
CMD ["micromamba","run","-n","housing","uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]

