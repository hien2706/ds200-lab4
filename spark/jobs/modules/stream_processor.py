from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import TimestampType
import logging
import os

from data_schema import LoanDataSchema
from feature_transformer import LoanFeatureTransformer
from model_trainer import LoanDefaultModelTrainer

logger = logging.getLogger(__name__)

class LoanDefaultStreamProcessor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.trainer = LoanDefaultModelTrainer(output_dir)
        self.transformer = LoanFeatureTransformer()
        self.spark = self._create_spark()
        self.schema = LoanDataSchema.get_schema()

    def _create_spark(self) -> SparkSession:
        return (SparkSession.builder
                .appName("LoanDefaultPrediction")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2")
                .getOrCreate())

    def _process_batch(self, df: DataFrame, batch_id: int):
        count = df.count()
        logger.info(f"Batch {batch_id}: {count} rows")
        if count == 0:
            return
        # build pipeline
        pipeline = self.transformer.build_pipeline()
        # train
        fitted_pipe, model = self.trainer.train(pipeline, df, batch_id)
        # eval
        metrics = self.trainer.evaluate(model, pipeline, df, batch_id)
        self.trainer.save_metrics(metrics, batch_id)
        self.trainer.save_model(batch_id)

    def start(self, kafka_servers: str, topic: str, interval: str="30 seconds"):
        os.makedirs(f"{self.output_dir}/batch_data", exist_ok=True)
        df = (self.spark.readStream.format("kafka")
              .option("kafka.bootstrap.servers", kafka_servers)
              .option("subscribe", topic)
              .option("startingOffsets", "earliest").load())
        parsed = df.select(from_json(col("value").cast("string"), self.schema).alias("data")).select("data.*")
        ts = parsed.withColumn("event_timestamp", col("timestamp").cast(TimestampType()))
        query = (ts.writeStream
                 .foreachBatch(self._process_batch)
                 .trigger(processingTime=interval)
                 .start())
        query.awaitTermination()