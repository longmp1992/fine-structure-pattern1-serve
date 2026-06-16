FROM python:3.12-slim

ARG HISTOSEG_REF=7e0526013f2d36200e464a070a359dc12a982c19

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    GRADIO_TEMP_DIR=/tmp/gradio \
    APP_DATA_DIR=/home/appuser/app/project-vol \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /home/appuser/app

COPY --chown=1000:1000 requirements.txt ./requirements.txt

RUN python -m pip install -U pip \
    && python -m pip install -r requirements.txt \
    && git clone https://github.com/hutaobo/HistoSeg.git /tmp/HistoSeg \
    && cd /tmp/HistoSeg \
    && git checkout "${HISTOSEG_REF}" \
    && python -m pip install /tmp/HistoSeg \
    && rm -rf /tmp/HistoSeg \
    && mkdir -p "$APP_DATA_DIR" /tmp/.cache /tmp/matplotlib /tmp/gradio \
    && chown -R 1000:1000 /home/appuser /tmp/.cache /tmp/matplotlib /tmp/gradio

COPY --chown=1000:1000 main.py ./main.py

USER 1000:1000

EXPOSE 7860

CMD ["python", "main.py"]
