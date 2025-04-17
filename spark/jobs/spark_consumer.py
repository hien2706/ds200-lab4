import os
from modules.stream_processor import LoanDefaultStreamProcessor

if __name__ == "__main__":
    output_dir        = os.getenv("OUTPUT_DIR", "/opt/spark-data/loan_default_prediction")
    kafka_servers     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka1:29092")
    kafka_topic       = os.getenv("KAFKA_TOPIC", "loan-applications")
    batch_interval    = os.getenv("BATCH_INTERVAL", "30 seconds")

    processor = LoanDefaultStreamProcessor(output_dir)
    processor.start(kafka_servers, kafka_topic, batch_interval)
