# Dockerfile
FROM mambaorg/micromamba:1.5.8 AS base
WORKDIR /srv

# Copy env spec and create env
COPY conda_environment.yml /srv/conda_environment.yml
RUN micromamba create -y -n housing -f /srv/conda_environment.yml && \
    micromamba clean --all --yes
SHELL ["micromamba", "run", "-n", "housing", "/bin/bash", "-lc"]

# ---- App image ----
FROM base AS app
WORKDIR /srv
COPY . /srv

EXPOSE 8080
# Default command (CI overrides this with a different command)
CMD ["micromamba","run","-n","housing","uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
