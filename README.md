# Loan Default Streaming Prediction

![Project Pipeline](https://github.com/hien2706/ds200-lab4/blob/main/pictures/project_pipeline.jpg)

## Overview

This project demonstrates a **real-time streaming machine learning pipeline** using Kafka and Spark. We simulate loan application data with a Python **producer**, stream it into **Kafka**, and consume it with **Spark Structured Streaming**. Each micro‑batch is preprocessed, used to **incrementally train** an **SGDClassifier** model, and then saved along with evaluation metrics.

---

## Architecture

1. **Data Producer**  
   A Python script (`producer/producer.py`) generates synthetic loan applications and pushes them to the `loan-applications` Kafka topic.

2. **Kafka**  
   Acts as the messaging layer, buffering incoming records.

3. **Spark Consumer (Streaming)**  
   Spark Structured Streaming reads from Kafka in fixed intervals (e.g. every 30s). Each batch is:
   - Parsed against a predefined schema.
   - Preprocessed via a Spark ML **Pipeline** (indexing, one‑hot encoding, scaling, assembling).
   - Fed to an **SGDClassifier** using `partial_fit()` for **incremental learning**.
   - Evaluated on the same batch to produce accuracy metrics.
   - The updated model and metrics are saved to disk.

```text
Producer → Kafka → Spark Streaming → (train on batch) → Models + Metrics
```

---

## Models & Training

- **Model**: `sklearn.linear_model.SGDClassifier` with `loss="log_loss"`, `penalty="l2"`, `warm_start=True`.
- **Incremental Learning**:  
  Each batch calls `model.partial_fit(X, y, classes=[0,1])`, so the model **remembers** prior data and **updates** continuously.
- **Saving**:
  - **Models**: `spark/data/loan_default_prediction/models/model_{count}_batch_{id}.joblib`
  - **Metrics**: `spark/data/loan_default_prediction/metrics/batch_{id}_metrics.json`

---

## Directory Structure

```
.
├── docker-compose.yml        # Orchestrates Kafka, Zookeeper, Spark, Producer
├── README.md                 # This file
├── LICENSE
├── pictures
│   └── project_pipeline.jpg  # Architecture diagram
├── producer
│   ├── Dockerfile.Producer   # Dockerfile for producer container
│   ├── producer.py           # Generates and sends loan data to Kafka
│   └── requirements.txt      # Python deps for producer
├── spark
│   ├── Dockerfile.PySpark    # Dockerfile for Spark container
│   ├── requirements.txt      # Python deps for Spark job
│   ├── apps                  # (unused / future apps)
│   ├── data
│   │   └── loan_default_prediction
│   │       ├── batch_data    # (optional: raw batch dumps)
│   │       ├── metrics       # JSON metrics per batch
│   │       └── models        # Joblib models per batch
│   ├── jobs
│   │   ├── modules           # Modular code
│   │   │   ├── data_schema.py
│   │   │   ├── feature_transformer.py
│   │   │   ├── model_trainer.py
│   │   │   └── stream_processor.py
│   │   └── spark_consumer.py # Entry‑point for Spark job
│   └── run_logs
│       └── spark_job.log     # Captured stdout/stderr of Spark job
└── hien_note.txt             # Developer notes
```

- **`producer/`**: Python code & Dockerfile to push data into Kafka.  
- **`spark/jobs/modules/`**: Reusable classes (schema, transformer, trainer, processor).  
- **`spark/jobs/spark_consumer.py`**: Main streaming job.  
- **`spark/data/loan_default_prediction/`**: Output directory for models & metrics.

---

## Setup & Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/hien2706/ds200-lab4.git
   cd ds200-lab4
   ```

2. **Start services**  
   ```bash
   docker-compose up -d
   ```

3. **Create the Kafka topic**  
   ```bash
   docker exec -it kafka1 bash
   kafka-topics --bootstrap-server localhost:9092 --create --topic loan-applications --partitions 1 --replication-factor 1
   ```

4. **Start the data producer**  
   ```bash
   docker exec -it kafka-producer bash
   python producer.py
   ```

5. **Run the Spark streaming job**  
   ```bash
   docker exec -it spark-master bash
   nohup spark-submit \
     --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2,io.delta:delta-spark_2.12:3.3.0 \
     --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
     --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
     --conf "spark.sql.streaming.checkpointLocation=/tmp/kafka-syslog-checkpoint" \
     /app/spark_jobs/spark_consumer.py \
     > /app/spark_logs/spark_job.log 2>&1 &
   ```

6. **Monitor logs**  
   ```bash
   tail -f spark/run_logs/spark_job.log
   ```

---

## Conclusion

This lab showcases a **fully-streaming ML pipeline**: generating, ingesting, preprocessing, and **incrementally training** a model in real time. The modular structure and Dockerized approach make it easy to extend—swap in new models, adjust feature pipelines, or scale to larger clusters.
