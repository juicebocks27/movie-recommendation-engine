import logging
from flask import Flask, jsonify, request, render_template, Response
import pandas as pd
import pickle
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST, Counter
from prometheus_flask_exporter import PrometheusMetrics
import threading
import subprocess
import time
from dotenv import load_dotenv
load_dotenv()

request_count = Counter('request_count', 'App Request Count', ['method', 'endpoint'])

nltk.download('stopwords')

# ---------------------- LOGGER CONFIG ---------------------- #
logging.basicConfig(
    filename='app_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
# ----------------------------------------------------------- #

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Initialize PorterStemmer
ps = PorterStemmer()

# Load your DataFrame
with open('Files/new_df_dict.pkl', 'rb') as f:
    new_df_dict = pickle.load(f)
new_df = pd.DataFrame.from_dict(new_df_dict)
movie_titles = new_df['title'].values

with open('Files/movies_dict.pkl', 'rb') as f:
    movies_dict = pickle.load(f)
movies = pd.DataFrame.from_dict(movies_dict)

with open('Files/movies2_dict.pkl', 'rb') as f:
    movies2_dict = pickle.load(f)
movies2 = pd.DataFrame.from_dict(movies2_dict)

# Poster Cache
poster_cache = {}

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{movie_id}"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w780"
FALLBACK_IMAGE = "https://media.istockphoto.com/vectors/error-icon-vector-illustration-vector-id922024224?k=6&m=922024224&s=612x612&w=0&h=LXl8Ul7bria6auAXKIjlvb6hRHkAodTqyqBeA6K7R54="

def fetch_poster(movie_id):
    if movie_id in poster_cache:
        return poster_cache[movie_id]

    url = TMDB_API_URL.format(movie_id=movie_id)
    params = {"api_key": TMDB_API_KEY}

    try:
        logger.info(f"Fetching poster for movie_id: {movie_id}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            poster_cache[movie_id] = poster_url
            return poster_url

        logger.warning(f"No poster_path found for movie_id: {movie_id}")
    except requests.RequestException as e:
        logger.warning(f"Request failed for movie_id {movie_id}: {e}")
    except ValueError as e:
        logger.warning(f"Invalid JSON for movie_id {movie_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error for movie_id {movie_id}: {e}")

    return FALLBACK_IMAGE

def load_similarity_matrix(filename):
    logger.info(f"Loading similarity matrix: {filename}")
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_recommendations(title, sim_file):
    try:
        logger.info(f"Generating recommendations for '{title}' using {sim_file}")
        similarity = load_similarity_matrix(sim_file)
        idx = new_df[new_df['title'].str.lower() == title.lower()].index[0]
        distances = similarity[idx]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommendations = []
        for i in movie_list:
            movie_title = new_df.iloc[i[0]]['title']
            movie_id = new_df.iloc[i[0]]['movie_id']
            poster = fetch_poster(movie_id)
            recommendations.append({"title": movie_title, "poster": poster})
        return recommendations
    except IndexError:
        logger.error(f"Movie '{title}' not found in dataset.")
        return {"error": "Movie not found"}, 404
    except Exception as e:
        logger.exception(f"Unexpected error for movie '{title}': {e}")
        return {"error": str(e)}, 500

@app.route('/recommendations/tags/<title>', methods=['GET'])
def recommend_by_tags(title):
    logger.info(f"API Call: /recommendations/tags/{title}")
    return jsonify(get_recommendations(title, 'Files/similarity_tags_tags.pkl'))

@app.route('/recommendations/genres/<title>', methods=['GET'])
def recommend_by_genres(title):
    logger.info(f"API Call: /recommendations/genres/{title}")
    return jsonify(get_recommendations(title, 'Files/similarity_tags_genres.pkl'))

@app.route('/recommendations/cast/<title>', methods=['GET'])
def recommend_by_cast(title):
    logger.info(f"API Call: /recommendations/cast/{title}")
    return jsonify(get_recommendations(title, 'Files/similarity_tags_tcast.pkl'))

@app.route('/', methods=['GET'])
def home():
    logger.info("Rendering Home Page")
    return render_template('index.html', movie_titles=movie_titles)

@app.before_request
def before_request():
    request_count.labels(request.method, request.path).inc()

# -------------------- PROMETHEUS METRICS DEFINITIONS --------------------
IO_METRICS = {
    'io_read_rate': Gauge("io_read_rate", "Disk read rate in bytes/sec", ["device"]),
    'io_write_rate': Gauge("io_write_rate", "Disk write rate in bytes/sec", ["device"]),
    'io_tps': Gauge("io_tps", "Transactions per second", ["device"]),
    'io_read_bytes': Gauge("io_read_bytes", "Total bytes read", ["device"]),
    'io_write_bytes': Gauge("io_write_bytes", "Total bytes written", ["device"])
}

cpu_avg_percent = Gauge("cpu_avg_percent", "CPU utilization percentage", ["mode"])
meminfo_gauges = {}

# -------------------- METRIC COLLECTION FUNCTIONS --------------------

def collect_io_stats():
    try:
        result = subprocess.run(["iostat", "-dx"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.splitlines()
        io_metrics = {}

        for line in lines[3:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            device = parts[0]
            io_metrics[device] = {
                "io_read_rate": float(parts[1]),
                "io_write_rate": float(parts[2]),
                "io_tps": float(parts[3]),
                "io_read_bytes": float(parts[4]) * 1024,
                "io_write_bytes": float(parts[5]) * 1024
            }
        return io_metrics
    except Exception as e:
        logger.error(f"Error collecting IO stats: {e}")
        return {}

def collect_cpu_stats():
    try:
        result = subprocess.run(["iostat", "-c"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if "avg-cpu" in line:
                cpu_values = lines[i + 1].split()
                return {
                    "user": float(cpu_values[0]),
                    "nice": float(cpu_values[1]),
                    "system": float(cpu_values[2]),
                    "iowait": float(cpu_values[3]),
                    "idle": float(cpu_values[5])
                }
        return {}
    except Exception as e:
        logger.error(f"Error collecting CPU stats: {e}")
        return {}

def get_mem_metrics():
    metrics = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip().lower().replace('(', '_').replace(')', '_')
                    metrics[key] = float(value.split()[0]) * 1024
        return metrics
    except Exception as e:
        logger.error(f"Memory error: {str(e)}")
        return {}

# -------------------- METRIC UPDATE THREAD --------------------

def update_metrics_forever():
    while True:
        try:
            cpu_data = collect_cpu_stats()
            io_data = collect_io_stats()
            mem_data = get_mem_metrics()

            for device, stats in io_data.items():
                IO_METRICS['io_read_rate'].labels(device=device).set(stats["io_read_rate"])
                IO_METRICS['io_write_rate'].labels(device=device).set(stats["io_write_rate"])
                IO_METRICS['io_tps'].labels(device=device).set(stats["io_tps"])
                IO_METRICS['io_read_bytes'].labels(device=device).set(stats["io_read_bytes"])
                IO_METRICS['io_write_bytes'].labels(device=device).set(stats["io_write_bytes"])

            for mode, value in cpu_data.items():
                cpu_avg_percent.labels(mode=mode).set(value)

            for name, value in mem_data.items():
                if name not in meminfo_gauges:
                    meminfo_gauges[name] = Gauge(f"meminfo_{name}", f"Memory metric: {name}")
                meminfo_gauges[name].set(value)

            logger.info("Metrics updated")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            time.sleep(5)

# -------------------- FLASK ENDPOINTS --------------------

@app.route("/metrics")
def metrics():
    # update_metrics_forever()
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)


# start_http_server(5001)  # Prometheus will scrape this port

if __name__ == '__main__':
    threading.Thread(target=update_metrics_forever, daemon=True).start()
    logger.info("Starting Flask App")
    app.run(debug=True, host="0.0.0.0", port = 5555)