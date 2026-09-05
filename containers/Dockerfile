FROM python:3.12-slim
LABEL org.opencontainers.image.title="PHAxis CPU demo and report"
WORKDIR /opt/phaxis
COPY . .
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 phaxis
USER phaxis
WORKDIR /home/phaxis
ENTRYPOINT ["phaxis"]
CMD ["--help"]
