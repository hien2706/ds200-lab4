from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging
import os
import json
from typing import List, Dict, Optional
from pyspark.sql.functions import lit
from sklearn.linear_model import SGDClassifier  # Import SGDClassifier
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoanDataSchema:
    """Define the schema for loan application data"""
    
    @staticmethod
    def get_schema():
        return StructType([
            StructField("timestamp", StringType(), True),
            StructField("application_id", StringType(), True),
            StructField("person_income", IntegerType(), True),
            StructField("person_home_ownership", StringType(), True),
            StructField("loan_amnt", IntegerType(), True),
            StructField("loan_intent", StringType(), True),
            StructField("loan_int_rate", DoubleType(), True),
            StructField("loan_percent_income", DoubleType(), True),
            StructField("previous_loan_defaults_on_file", StringType(), True),
            StructField("label", IntegerType(), True)
        ])


class LoanFeatureTransformer:
    """Transform raw loan data into model-ready features"""
    
    def __init__(self):
        # Define categorical features
        self.categorical_cols = ["person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
        
        # Define numerical features
        self.numerical_cols = ["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income"]
        
    def build_pipeline(self):
        """Build a preprocessing pipeline for feature transformation"""
        from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
        stages = []
        
        # Process categorical features
        # 1. String indexing
        string_indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") 
            for col in self.categorical_cols
        ]
        stages.extend(string_indexers)
        
        # 2. One-hot encoding
        one_hot_encoders = [
            OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
            for col in self.categorical_cols
        ]
        stages.extend(one_hot_encoders)
        
        # Standard scaling for numerical features
        for col in self.numerical_cols:
            # Create assembler to put single feature into a vector
            feature_assembler = VectorAssembler(
                inputCols=[col], 
                outputCol=f"{col}_vec"
            )
            stages.append(feature_assembler)
            
            # Scale the feature
            feature_scaler = StandardScaler(
                inputCol=f"{col}_vec",
                outputCol=f"{col}_scaled",
                withStd=True,
                withMean=True
            )
            stages.append(feature_scaler)
        
        # Assemble all transformed features into a single vector
        assembler_inputs = [f"{col}_vec" for col in self.categorical_cols] + [f"{col}_scaled" for col in self.numerical_cols]
        final_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
        stages.append(final_assembler)
        
        return stages


class LoanDefaultModelTrainer:
    """Train a loan default prediction model on batches of data"""
    
    def __init__(self, output_dir: str):
        self.feature_transformer = LoanFeatureTransformer()
        self.output_dir = output_dir
        self.model_count = 0
        self.model = self.load_or_create_model()  # Load existing model or create a new one

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)

    def load_or_create_model(self):
        """Load an existing model or create a new one if not found"""
        model_path = f"{self.output_dir}/models/model.joblib"  # You can name the model based on your criteria
        
        if os.path.exists(model_path):
            logger.info(f"Model found at {model_path}, loading the model.")
            model = joblib.load(model_path)
        else:
            logger.info(f"No model found at {model_path}, creating a new model.")
            model = SGDClassifier(loss="log", penalty="l2", max_iter=1, warm_start=True)
        
        return model

    def train_model(self, df: DataFrame, batch_id: int) -> SGDClassifier:
        """Train a model on a batch of data"""
        logger.info(f"Starting training on batch {batch_id} with {df.count()} records")
        
        # Transform the features
        feature_pipeline = self.feature_transformer.build_pipeline()
        feature_df = df
        for stage in feature_pipeline:
            feature_df = stage.fit(feature_df).transform(feature_df)
        
        # Extract features and labels from the DataFrame
        X = feature_df.select("features").rdd.map(lambda row: row[0].toArray()).collect()
        y = feature_df.select("label").rdd.map(lambda row: row[0]).collect()
        
        # Train the model using SGDClassifier with partial_fit for incremental learning
        if len(X) > 0:
            self.model.partial_fit(X, y, classes=[0, 1])  # Assuming binary classification

        # Increment model count
        self.model_count += 1
        
        return self.model
    
    def evaluate_model(self, model: SGDClassifier, df: DataFrame, batch_id: int) -> Dict:
        """Evaluate the model and return metrics"""
        logger.info(f"Evaluating model on batch {batch_id}")
        
        # Transform the features
        feature_pipeline = self.feature_transformer.build_pipeline()
        feature_df = df
        for stage in feature_pipeline:
            feature_df = stage.fit(feature_df).transform(feature_df)
        
        # Make predictions
        predictions = model.predict(feature_df.select("features").rdd.map(lambda row: row[0].toArray()).collect())
        
        # Calculate metrics
        correct = sum([1 if pred == true else 0 for pred, true in zip(predictions, df.select("label").collect())])
        accuracy = correct / len(predictions) if len(predictions) > 0 else 0

        metrics = {
            "batch_id": batch_id,
            "model_id": self.model_count,
            "accuracy": accuracy
        }
        
        return metrics
    
    def save_metrics(self, metrics: Dict, batch_id: int):
        """Save metrics to a JSON file"""
        metrics_file = f"{self.output_dir}/metrics/batch_{batch_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    
    def save_model(self, model: SGDClassifier, batch_id: int):
        """Save model to disk"""
        model_path = f"{self.output_dir}/models/model_{self.model_count}_batch_{batch_id}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def print_results(self, metrics: Dict):
        """Print training results to console"""
        logger.info(f"===== Model {metrics['model_id']} (Batch {metrics['batch_id']}) =====")
        logger.info(f"Accuracy:       {metrics['accuracy']:.4f}")
        logger.info("=" * 50)



class LoanDefaultStreamProcessor:
    """Process streaming loan application data and train models in batches"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.model_trainer = LoanDefaultModelTrainer(output_dir)
        
        # Create Spark session
        self.spark = self._create_spark_session()
        
        # Set log level
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Define schema
        self.schema = LoanDataSchema.get_schema()
    
    def _create_spark_session(self) -> SparkSession:
        """Create a Spark session with necessary configurations"""
        return (SparkSession.builder
                .appName("LoanDefaultPrediction")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2")
                .config("spark.sql.streaming.checkpointLocation", "/tmp/kafka-syslog-checkpoint")
                .getOrCreate())
    
    def _process_batch(self, df: DataFrame, epoch_id: int):
        """Process each batch of loan application data"""
        count = df.count()
        logger.info(f"Batch {epoch_id}: Received {count} loan applications")
        
        if count == 0:
            logger.info(f"Batch {epoch_id} is empty, skipping training")
            return
        
        # Print batch stats
        logger.info("Default distribution in this batch:")
        df.groupBy("label").count().show()
        
        # Save batch data to Delta table for reference
        delta_path = f"{self.output_dir}/batch_data/batch_{epoch_id}"
        df.write.format("delta").mode("overwrite").save(delta_path)
        logger.info(f"Batch data saved to {delta_path}")
        
        try:
            # Train model on this batch
            model = self.model_trainer.train_model(df, epoch_id)
            
            # Evaluate model
            metrics = self.model_trainer.evaluate_model(model, df, epoch_id)
            
            # Print results
            self.model_trainer.print_results(metrics)
            
            # Save metrics
            self.model_trainer.save_metrics(metrics, epoch_id)
            
            # Save model
            self.model_trainer.save_model(model, epoch_id)
            
        except Exception as e:
            logger.error(f"Error processing batch {epoch_id}: {e}", exc_info=True)
    
    def start_processing(self, kafka_bootstrap_servers: str, topic: str, batch_interval: str = "30 seconds"):
        """Start processing the streaming data"""
        logger.info(f"Starting stream processing from topic '{topic}' with batch interval {batch_interval}")
        
        # Create output directories if they don't exist
        os.makedirs(f"{self.output_dir}/batch_data", exist_ok=True)
        
        # Read from Kafka
        kafka_df = (self.spark
                   .readStream
                   .format("kafka")
                   .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
                   .option("subscribe", topic)
                   .option("startingOffsets", "earliest")
                   .load())
        
        # Parse JSON data
        parsed_df = kafka_df.select(
            from_json(col("value").cast("string"), self.schema).alias("data")
        ).select("data.*")
        
        # Convert timestamp string to timestamp type
        df_with_timestamp = parsed_df.withColumn(
            "event_timestamp", 
            col("timestamp").cast(TimestampType())
        )
        
        # Fix previous_loan_defaults to match expected values
        fixed_df = df_with_timestamp.withColumn(
            "previous_loan_defaults_on_file",
            col("previous_loan_defaults_on_file")
        )
        
        # Process in batches
        query = (fixed_df
                .writeStream
                .foreachBatch(self._process_batch)
                .trigger(processingTime=batch_interval)
                .start())
        
        # Wait for the query to terminate
        logger.info("Stream processing started. Waiting for termination...")
        query.awaitTermination()


def main():
    """Main entry point for the application"""
    # Set output directory
    output_dir = "/opt/spark-data/loan_default_prediction"
    
    # Configure Kafka connection
    kafka_bootstrap_servers = "kafka1:29092"
    kafka_topic = "loan-applications"
    
    # Create and start the stream processor
    processor = LoanDefaultStreamProcessor(output_dir)
    processor.start_processing(kafka_bootstrap_servers, kafka_topic, "30 seconds")


if __name__ == "__main__":
    main()
