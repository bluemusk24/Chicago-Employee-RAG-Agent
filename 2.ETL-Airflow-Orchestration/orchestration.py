from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum


def run_pipeline():
    import requests
    import pandas as pd
    import dlt
    import duckdb
    from dlt.common.pipeline import LoadInfo

    # Pagination function
    def pagination(url):
        while True:
            response = requests.get(url)
            response.raise_for_status()
            yield response.json()

            if "next" not in response.links:
                break
            url = response.links["next"]["url"]

    # DLT Resource
    @dlt.resource(table_name="employee")
    def get_employee():
        employee_url = "https://data.cityofchicago.org/resource/xzkq-xp2w.json"
        yield pagination(employee_url)

    # Create Pipeline
    pipeline = dlt.pipeline(
        pipeline_name="chicago_employee",
        destination="duckdb",
        dataset_name="chicago_employee_data",
        dev_mode=True,
    )

    load_info = pipeline.run([get_employee()])
    assert len(load_info.loads_ids) == 1
    load_info.raise_on_failed_jobs()

    # Connect to DuckDB
    conn = duckdb.connect(f"{pipeline.pipeline_name}.duckdb")
    conn.sql(f"SET search_path = '{pipeline.dataset_name}'")

    employee_data = conn.sql("SELECT * FROM employee").df()

    # TRANSFORMATION LOGIC
    num_dtypes = ["annual_salary", "typical_hours", "hourly_rate"]
    cat_dtypes = [
        "name",
        "job_titles",
        "department",
        "full_or_part_time",
        "salary_or_hourly",
    ]

    # Drop DLT metadata columns safely
    df = employee_data.drop(
        columns=[col for col in ["_dlt_load_id", "_dlt_id"] if col in employee_data.columns]
    )

    # Convert numeric columns
    for num in num_dtypes:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")
            df[num] = df[num].fillna(df[num].mean())

    # Convert categorical columns
    for cat in cat_dtypes:
        if cat in df.columns:
            df[cat] = df[cat].astype("string")

    # Remove duplicates
    df = df.drop_duplicates()

    print("Pipeline completed successfully.")
    print(df.head())

# Airflow DAG Definition
default_args = {
    "owner": "airflow",
    "start_date": pendulum.today("UTC").add(days=-1),
    "retries": 1,
}

with DAG(
    dag_id="employee_pipeline_dag",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=["etl", "chicago"],
) as dag:

    employee_task = PythonOperator(
        task_id="run_employee_pipeline",
        python_callable=run_pipeline,
    )