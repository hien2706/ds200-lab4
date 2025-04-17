from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler

class LoanFeatureTransformer:
    """Transform raw loan data into model-ready features"""
    def __init__(self):
        self.categorical_cols = [
            "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"
        ]
        self.numerical_cols = [
            "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income"
        ]

    def build_pipeline(self):
        stages = []
        # Categorical: index + one-hot
        for col in self.categorical_cols:
            stages.append(StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip"))
            stages.append(OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec"))
        # Numerical: assemble + scale
        for col in self.numerical_cols:
            stages.append(VectorAssembler(inputCols=[col], outputCol=f"{col}_vec"))
            stages.append(StandardScaler(inputCol=f"{col}_vec", outputCol=f"{col}_scaled", withStd=True, withMean=True))
        # Final features
        assembler_inputs = [f"{c}_vec" for c in self.categorical_cols] + [f"{n}_scaled" for n in self.numerical_cols]
        stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="features"))
        return Pipeline(stages=stages)