from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

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