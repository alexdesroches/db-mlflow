# Databricks notebook source
# Get our experiement 
import mlflow
team_name = "alexdesroches_mlflow"
model_name = team_name
experiment_name = team_name 
experiment_id = [experiment.experiment_id for experiment in mlflow.list_experiments() if experiment_name in experiment.name][0]
print(experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load an experiment

# COMMAND ----------

# Let's load the experiment... 
# If this were *really* another notebook, I'd have to obtain the Experiment ID from the MLflow page.  
# But since we are in the original notebook, I can get it as a default value

df_client = spark.read.format("mlflow-experiment").load(experiment_id)
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
selected_run_id = df_model_selector.first()[1]
selected_model_accuracy = df_model_selector.first()[3]
selected_model_uri = df_model_selector.first()[4]

print(f"Selected experiment ID: {selected_experiment_id}")
print(f"Selected run ID: {selected_run_id}")
print(f"Selected model accuracy: {selected_model_accuracy}")
print(f"Selected model URI: {selected_model_uri}")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Is this model registered in the model registry, and is it currently in production?

# COMMAND ----------


import mlflow.spark as mlflow_spark
from mlflow.tracking import MlflowClient
client = MlflowClient()

def model_isin_production(model_name):
  try:
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
    model = mlflow_spark.load_model(model_production_uri)
    return True
  except:
    return False


# COMMAND ----------

model_in_prod = model_isin_production(model_name)

if model_in_prod:
  print("Model is already in production.")
else:
  print(f"No model named {model_name} in production")
  print("Registering the best model for the current experiment...")
  # Register Model
  result = mlflow.register_model(f"runs:/{selected_run_id}/spark-model", model_name)
  # Transition to Production
  client.transition_model_version_stage(name=model_name, version=result.version, stage="Production")

# COMMAND ----------

def get_production_model_accuracy(model_name):
  model = client.get_registered_model(model_name)
  model_versions = model.latest_versions
  print(model_versions)
  model_production_run_id = [model_version.run_id for model_version in model_versions if model_version.current_stage=="Production"][0]
  print("Current production model run_id: ", model_production_run_id)
  
  model_production_accuracy = spark.sql(f"""select metrics.Accuracy, run_id from vw_client where run_id = "{model_production_run_id}" """).first()[0]
  
  print("Current production model accuracy: ", model_production_accuracy)
  return model_production_accuracy

# COMMAND ----------

# spark.sql(f"""select metrics.Accuracy, run_id from vw_client where run_id = "9002af91c0b4477f991981e403d42af1" """).show()

# COMMAND ----------

def promote_model_if_better(model_name, selected_run_id):
  model_production_accuracy = get_production_model_accuracy(model_name)
  
  if model_production_accuracy < selected_model_accuracy:
    print("Promoting model to production since it's more accurate")
    result = mlflow.register_model(f"runs:/{selected_run_id}/spark-model", model_name)
    client.transition_model_version_stage(name=model_name, version=result.version, stage="Production")
  
  elif model_production_accuracy >= selected_model_accuracy:
    print(f"Selected model accuracy: {selected_model_accuracy}")
    print("NOT promoting model to production since it's less accurate or the same.")
    
  

# COMMAND ----------

promote_model_if_better(model_name, selected_run_id)
