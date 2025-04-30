from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import preprocessing utilities (assumes utils are placed in a separate file in dags/)
from preprocess import read_csv_to_df

# Constants
DATA_DIR = '/opt/airflow/files'  # This should be volume-mounted in docker-compose

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define Python callables
def generate_dataframes():
    movies, new_df, movies2 = read_csv_to_df()

    with open(f'{DATA_DIR}/movies_dict.pkl', 'wb') as f:
        pickle.dump(movies.to_dict(), f)

    with open(f'{DATA_DIR}/movies2_dict.pkl', 'wb') as f:
        pickle.dump(movies2.to_dict(), f)

    with open(f'{DATA_DIR}/new_df_dict.pkl', 'wb') as f:
        pickle.dump(new_df.to_dict(), f)

def vectorize_and_store(col_name):
    with open(f'{DATA_DIR}/new_df_dict.pkl', 'rb') as f:
        new_df = pd.DataFrame.from_dict(pickle.load(f))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vec = cv.fit_transform(new_df[col_name].fillna("")).toarray()
    similarity = cosine_similarity(vec)

    with open(f'{DATA_DIR}/similarity_tags_{col_name}.pkl', 'wb') as f:
        pickle.dump(similarity, f)

# Define DAG
with DAG(
    dag_id="generate_movie_pickle_files",
    default_args=default_args,
    description='Generate pickle files for movie similarity',
    schedule_interval="0 12 * * 0",  # Every Sunday at 12:00 PM
    catchup=False,
    tags=["movie_recommender"]
) as dag:

    task_generate_dfs = PythonOperator(
        task_id='generate_dataframes',
        python_callable=generate_dataframes
    )

    task_tags = PythonOperator(
        task_id='generate_similarity_tags',
        python_callable=vectorize_and_store,
        op_args=['tags']
    )

    task_genres = PythonOperator(
        task_id='generate_similarity_genres',
        python_callable=vectorize_and_store,
        op_args=['genres']
    )

    task_keywords = PythonOperator(
        task_id='generate_similarity_keywords',
        python_callable=vectorize_and_store,
        op_args=['keywords']
    )

    task_tcast = PythonOperator(
        task_id='generate_similarity_tcast',
        python_callable=vectorize_and_store,
        op_args=['tcast']
    )

    task_tprod = PythonOperator(
        task_id='generate_similarity_tprduction_comp',
        python_callable=vectorize_and_store,
        op_args=['tprduction_comp']
    )

    # Set task dependencies
    task_generate_dfs >> [task_tags, task_genres, task_keywords, task_tcast, task_tprod]