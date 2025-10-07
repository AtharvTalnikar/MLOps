from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.simple_regression import (
    load_reg_data,
    preprocess_reg,
    train_reg_model,
    predict_reg,
)

default_args = {
    "owner": "your_name",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="Airflow_Lab1_Regression",
    description="Simple Linear Regression pipeline (file-based XCom)",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["lab", "regression"],
    max_active_runs=1,
) as dag:

    t_load = PythonOperator(
        task_id="load_reg_data",
        python_callable=load_reg_data,
    )

    def _preprocess(train_parquet_path: str):
        return preprocess_reg(train_parquet_path) 

    t_pre = PythonOperator(
        task_id="preprocess_reg",
        python_callable=_preprocess,
        op_args=[t_load.output],
    )

    def _train(pre_tuple, filename="reg_model.sav"):
        X_path, y_path, scaler_path = pre_tuple
        return train_reg_model(X_path, y_path, filename)

    t_train = PythonOperator(
        task_id="train_reg_model",
        python_callable=_train,
        op_args=[t_pre.output],
    )

    def _predict(train_tuple, pre_tuple):
        model_path, r2_train = train_tuple
        X_path, y_path, scaler_path = pre_tuple
        pred_first = predict_reg(model_path, scaler_path)
        print(f"[REG] r2_train={r2_train:.4f}, model={model_path}, pred_first_row={pred_first}")
        return pred_first

    t_predict = PythonOperator(
        task_id="predict_reg",
        python_callable=_predict,
        op_args=[t_train.output, t_pre.output],
    )

    t_load >> t_pre >> t_train >> t_predict
