# Use the official Airflow image as base
FROM apache/airflow:2.8.1-python3.11

# Switch to root for installing packages
USER root

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    && apt-get clean

# Switch back to airflow user
USER airflow

# Copy requirements (optional, if you split them)
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAGs and other project code
# COPY dags/ /opt/airflow/dags/
# COPY Files/ /opt/airflow/Files/
# COPY ./app/Files/ /opt/airflow/app/Files/  

# Ensure nltk data is downloaded
RUN python -m nltk.downloader stopwords

RUN airflow db init
RUN airflow users  create --role Admin --username airflow --email airflow --firstname airflow --lastname airflow --password airflow

# Default command (can be overridden in docker-compose)
CMD ["bash"]