# Usar la imagen slim de Python 3.10.8
FROM python:3.10.8-slim

# recordar exponer el puerto en el que se expondrá tu aplicación.
EXPOSE 8080

# instalar la última versión de pip
RUN pip install -U pip

# Copia los archivos necesarios al contenedor
COPY requirements.txt requirements.txt

# Instalar las dependencias necesarias
RUN pip install -r requirements.txt

# copiar en un directorio propio (para que no esté en el directorio de nivel superior)
COPY . /app
WORKDIR /app

# ejecutar!
ENTRYPOINT ["streamlit", "run", "Main.py", "--server.port=8080", "--server.address=0.0.0.0"]