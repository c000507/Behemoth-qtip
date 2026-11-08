# Router image (small, Fargate). Build context: this directory.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY router.py .
RUN useradd -r -u 10001 router
USER router
ENV PYTHONUNBUFFERED=1
EXPOSE 1337
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/health'%os.environ.get('LISTEN_PORT','1337'),timeout=4)"
CMD ["python", "router.py"]
