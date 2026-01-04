FROM ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12:main
RUN apt-get update && apt-get install -y python3-dev && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv /.venv --python 3.14
RUN uv pip install maturin
ENV PATH="/.venv/bin:$PATH"