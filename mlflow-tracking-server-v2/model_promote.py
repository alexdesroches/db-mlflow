# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC ### Load an experiment

# COMMAND ----------

# Let's load the experiment... 
# If this were *really* another notebook, I'd have to obtain the Experiment ID from the MLflow page.  
# But since we are in the original notebook, I can get it as a default value

df_client = spark.read.format("mlflow-experiment").load()
df_client.createOrReplaceTempView("vw_client")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Querying MLflow tables

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's take a peek at the data that MLflow returns
# MAGIC 
# MAGIC SELECT * FROM vw_client

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dynamically & Programatically Select the Best Model

# COMMAND ----------

# Let's query the MLflow data in a way that shows us the most accurate model in the first row
# This is possible because we logged accuracy as a metric using MLflow
# Then we can grab the run_id to get a handle to the model itself
from pyspark.sql.functions import col, lit, concat

# Select only rows where status = 'FINISHED'
# Order the rows so the row with the highest accuracy comes first
#
df_model_selector = (spark.read.table("vw_client")
                    .where(col("status") == "FINISHED")
                    .select(  "experiment_id"
                            , "run_id"
                            , "end_time"
                            , col("metrics.Accuracy").alias("accuracy")
                            , concat(col("artifact_uri") , lit("/spark-model")).alias("artifact_uri"))
                    .sort(col("accuracy").desc())
                    )


display(df_model_selector)

# COMMAND ----------

# Let's put some interesting columns into Python variables

selected_experiment_id = df_model_selector.first()[0]
selected_model_id = df_model_selector.first()[1]
selected_model_accuracy = df_model_selector.first()[3]
selected_model_uri = df_model_selector.first()[4]

print(f"Selected experiment ID: {selected_experiment_id}")
print(f"Selected model ID: {selected_model_id}")
print(f"Selected model accuracy: {selected_model_accuracy}")
print(f"Selected model URI: {selected_model_uri}")


# COMMAND ----------

# Now we can actually instantiate our chosen model with one line of code!

selected_model = mlflow_spark.load_model(selected_model_uri)


# COMMAND ----------

# Optionally register the model via the API
result = mlflow.register_model(
    selected_model_uri,
    team_name
)

# COMMAND ----------

from mlflow.tracking import MlflowClient
# Optionally push to Production
if selected_model_accuracy >= .90:
  # Can build logic here to push to prod if new accuracy > current prod accuracy
  client = MlflowClient()
  client.transition_model_version_stage(
      name=team_name,
      version=1,
      stage="Production"
  )

# COMMAND ----------

# Or, you can instantiate the model using dynamically using metadata!

model_name = team_name

# Load by model version
model_version_uri = "models:/{model_name}/1".format(model_name=model_name)
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow_spark.load_model(model_version_uri)

# Load by model stage
model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model_production = mlflow_spark.load_model(model_production_uri)



# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify that the input data we're using has no label column
# MAGIC 
# MAGIC SELECT * FROM current_readings_unlabeled

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the model...

# COMMAND ----------

# Here we prepare the data so the model can use it
# This is just a subset of the code we saw earlier when we developed the model

# First we read in the raw data
df_client_raw_data = spark.sql("""
  SELECT 
    device_type,
    device_id,
    reading_1,
    reading_2,
    reading_3
  FROM current_readings_unlabeled  
""")
    
# Create a numerical index of device_type values (it's a category, but Decision Trees don't need OneHotEncoding)
device_type_indexer = StringIndexer(inputCol="device_type", outputCol="device_type_index")
df_client_raw_data = device_type_indexer.fit(df_client_raw_data).transform(df_client_raw_data)

# Create a numerical index of device_id values (it's a category, but Decision Trees don't need OneHotEncoding)
device_id_indexer = StringIndexer(inputCol="device_id", outputCol="device_id_index")
df_client_raw_data = device_id_indexer.fit(df_client_raw_data).transform(df_client_raw_data)

# Populated df_raw_data with the all-numeric values
df_client_raw_data.createOrReplaceTempView("vw_client_raw_data")
df_client_raw_data = spark.sql("""
SELECT 
  device_type,
  device_type_index,
  device_id,
  device_id_index,
  reading_1,
  reading_2,
  reading_3
FROM vw_client_raw_data 
""")

# Assemble the data into label and features columns

assembler = VectorAssembler( 
inputCols=["device_type_index", "device_id_index", "reading_1", "reading_2", "reading_3"], 
outputCol="features")

df_client_raw_data = assembler.transform(df_client_raw_data)

display(df_client_raw_data)

# COMMAND ----------

# Now we can actually run the model we just instantiated

df_client_predictions = selected_model.transform(df_client_raw_data)
df_client_predictions.createOrReplaceTempView("vw_client_predictions")
display(df_client_predictions) # Let's take a look at the output... notice the "prediction" column (last column... scroll right)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Write predictions out to a Delta table

# COMMAND ----------

# I'm almost ready to write my data with predictions out to a Delta Lake table.  
# But I don't want to  use those numeric prediction values that the model produces.

# I would like to change them to the friendly names that were in my labeled training data
# Fortunately, Spark ML gives us a way to get these values

df = spark.sql("""
  SELECT 
    device_operational_status
  FROM current_readings_labeled
""")

# Create a numerical index of label values (device status) 
label_indexer = StringIndexer(inputCol="device_operational_status", outputCol="device_operational_status_index")
df = label_indexer.fit(df).transform(df)
    
labelReverse = IndexToString().setInputCol("device_operational_status_index")
df_reversed = labelReverse.transform(df)

df_reversed.createOrReplaceTempView("vw_reversed")
display(spark.sql("""
  SELECT DISTINCT
    device_operational_status,
    device_operational_status_index
  FROM vw_reversed
  ORDER BY device_operational_status_index ASC
"""))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's persist the output of our decision tree application
# MAGIC 
# MAGIC DROP TABLE IF EXISTS application_output;
# MAGIC 
# MAGIC CREATE TABLE application_output
# MAGIC USING DELTA
# MAGIC AS (
# MAGIC   SELECT
# MAGIC     device_type,
# MAGIC     device_id,
# MAGIC     reading_1,
# MAGIC     reading_2,
# MAGIC     reading_3,
# MAGIC     CASE   -- Change the numeric predictions to user-friendly text values
# MAGIC       WHEN prediction = 0 THEN "RISING"
# MAGIC       WHEN prediction = 1 THEN "IDLE"
# MAGIC       WHEN prediction = 2 THEN "NOMINAL"
# MAGIC       WHEN prediction = 3 THEN "HIGH"
# MAGIC       WHEN prediction = 4 THEN "RESETTING"
# MAGIC       WHEN prediction = 5 THEN "FAILURE"
# MAGIC       WHEN prediction = 6 THEN "DESCENDING"
# MAGIC       ELSE 'UNKNOWN'
# MAGIC     END AS predicted_device_operational_status
# MAGIC   FROM vw_client_predictions
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's verify that our new table looks good
# MAGIC 
# MAGIC SELECT * FROM application_output

# COMMAND ----------

# MAGIC  %md
# MAGIC  ### What just happened?
# MAGIC  
# MAGIC  We learned a lot about MLflow Tracking in this module.  We tracked:
# MAGIC  
# MAGIC  - Parameters
# MAGIC  - Metrics
# MAGIC  - Tags
# MAGIC  - The model itself
# MAGIC  
# MAGIC  
# MAGIC  Then we showed how to find a model, instantiate it, and run it to make predictions on a different data set.
# MAGIC  
# MAGIC  __*But we've just scratched the surface of what MLflow can do...*__
# MAGIC  
# MAGIC  To learn more, check out documentation and notebook examples for MLflow Models and MLflow Registry.