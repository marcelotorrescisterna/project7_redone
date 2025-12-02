#FROM python:3.11-slim
FROM public.ecr.aws/docker/library/python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos solo requirements primero (optimiza cache)
COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ahora copiamos el resto de la carpeta app/
COPY app/ .

# Entrena modelo durante el build
RUN python train_model.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
