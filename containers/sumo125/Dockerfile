FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      libfontconfig1 \
      libgl1 \
      libglu1-mesa \
      libgomp1 \
      libnsl2 \
      libx11-6 \
      libxcursor1 \
      libxext6 \
      libxfixes3 \
      libxft2 \
      libxi6 \
      libxrandr2 \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      numpy==1.26.4 \
      scipy==1.11.4 \
      matplotlib==3.8.4 \
      pymoo==0.6.1.1 \
      eclipse-sumo==1.25.0 \
      libsumo==1.25.0 \
      sumolib==1.25.0 \
      traci==1.25.0

ENV SUMO_HOME=/opt/venv/lib/python3.10/site-packages/sumo \
    PYTHONPATH=/workspace/Final_Submission/GPR_KG_Code:/opt/venv/lib/python3.10/site-packages/sumo/tools
ENV PATH=$SUMO_HOME/bin:$PATH

WORKDIR /workspace/Final_Submission/GPR_KG_Code

CMD ["python", "-m", "experiments.ingolstadt21.validate_oos_feasibility", "--dry-run"]
