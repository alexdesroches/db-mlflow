# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Model Tracking Rapidstart  <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/databricks icon.png?raw=true" width=100/> 
# MAGIC 
# MAGIC __NOTE:__ Use a cluster running Databricks 7.3 ML or higher.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Important Note!
# MAGIC **This notebook leverages the notebook: ./includes/setup to set up the following:**
# MAGIC 1. A new database for the demo's tables
# MAGIC 2. Download the sample dataset
# MAGIC 3. Create a few directories used for storing the sample datasets and tables.
# MAGIC 
# MAGIC ** You must have the correct priviliges in order to run this notebook correctly **
# MAGIC 
# MAGIC *__Alternatively,__ you can hard code the database name and create the tables using the code in the ./includes/setup notebook directly.*

# COMMAND ----------

team_name = "alexdesroches_mlflow"

setup_responses = dbutils.notebook.run("./includes/setup", 0, {"team_name": team_name}).split()

local_data_path = setup_responses[0]
dbfs_data_path = setup_responses[1]
database_name = setup_responses[2]

print(f"Path to be used for Local Files: {local_data_path}")
print(f"Path to be used for DBFS Files: {dbfs_data_path}")
print(f"Database Name: {database_name}")

# COMMAND ----------

# Let's set the default database name so we don't have to specify it on every query

# Hard code this if required. 

spark.sql(f"USE {database_name}")


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 1:  Import Required Libraries

# COMMAND ----------

from pyspark.ml.feature        import IndexToString, StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation     import MulticlassClassificationEvaluator 

import mlflow
from mlflow import spark as mlflow_spark # renamed to prevent collisions when doing spark.sql

import time


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Build and track a model with ML Flow!
# MAGIC 
# MAGIC #### What kind of data are we working with?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM current_readings_labeled
# MAGIC -- CAPTEURS

# COMMAND ----------

# MAGIC %md
# MAGIC #### Building a Model with MLflow Tracking
# MAGIC 
# MAGIC The cell below creates a Python function that builds, tests, and trains a Decision Tree model.  
# MAGIC 
# MAGIC We made it a function so that you can easily call it many times with different parameters.  This is a convenient way to create many different __Runs__ of an experiment, and will help us show the value of MLflow Tracking.
# MAGIC 
# MAGIC Read through the code in the cell below, and notice how we use MLflow Tracking in several different ways:
# MAGIC   
# MAGIC - First, we __initiate__ MLflow Tracking like this: 
# MAGIC 
# MAGIC ```with mlflow.start_run() as run:```
# MAGIC 
# MAGIC Then we illustrate several things we can do with Tracking:
# MAGIC 
# MAGIC - __Tags__ let us assign free-form name-value pairs to be associated with the run.  
# MAGIC 
# MAGIC - __Parameters__ let us name and record single values for a run.  
# MAGIC 
# MAGIC - __Metrics__ also let us name and record single numeric values for a run.  We can optionally record *multiple* values under a single name.
# MAGIC 
# MAGIC - Finally, we will log the __Model__ itself.
# MAGIC 
# MAGIC Notice the parameters that the function below accepts:
# MAGIC 
# MAGIC - __p_max_depth__ is used to specify the maximum depth of the decision tree that will be generated.  You can vary this parameter to tune the accuracy of your model
# MAGIC 
# MAGIC - __p_owner__ is the "value" portion of a Tag we have defined.  You can put any string value into this parameter.

# COMMAND ----------

def training_run(p_max_depth = 2, p_owner = "default") :
  with mlflow.start_run() as run:
    # Start a timer to get overall elapsed time for this function
    overall_start_time = time.time()
    
    # Log a Tag for the run
    mlflow.set_tag("Owner", p_owner)
    
    # Log a Parameter
    mlflow.log_param("Maximum Depth", p_max_depth)
    
    ########### STEP 1: Read in the raw data to use for training
    # We'll use an MLflow metric to log the time taken in each step 
    start_time = time.time()
    
    df_raw_data = spark.sql("""
      SELECT 
        device_type,
        device_operational_status AS label,
        device_id,
        reading_1,
        reading_2,
        reading_3
      FROM current_readings_labeled
    """)   
    end_time = time.time()
    elapsed_time = end_time - start_time

    # We'll use an MLflow metric to log the time taken in each step 
    # NOTE: this will be a multi-step metric that shows the elapsed time for each step in this function.
    #       Set this call to be step 1
    
    mlflow.log_metric("Step Elapsed Time", elapsed_time, 1)

    ########### STEP 2: Data Prep (INDEXING)
    
    # Index the Categorical data so the Decision Tree can use it
    start_time = time.time()
    
    
    # Create a numerical index of device_type values (it's a category, but Decision Trees don't need OneHotEncoding)
    device_type_indexer = StringIndexer(inputCol="device_type", outputCol="device_type_index")
    df_raw_data = device_type_indexer.fit(df_raw_data).transform(df_raw_data)
    
    # Create a numerical index of device_id values (it's a category, but Decision Trees don't need OneHotEncoding)
    device_id_indexer = StringIndexer(inputCol="device_id", outputCol="device_id_index")
    df_raw_data = device_id_indexer.fit(df_raw_data).transform(df_raw_data)

    # Create a numerical index of label values (device status) 
    label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
    df_raw_data = label_indexer.fit(df_raw_data).transform(df_raw_data)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    mlflow.log_metric("Step Elapsed Time", elapsed_time, 2)


    ########### STEP 3: INDEX 
    # Create a dataframe with the indexed data ready to be assembled
    start_time = time.time()
    
    # Populated df_raw_data with the all-numeric values
    df_raw_data.createOrReplaceTempView("vw_raw_data")
    df_raw_data = spark.sql("""
    SELECT 
      label_index AS label, 
      device_type_index AS device_type,
      device_id_index AS device_id,
      reading_1,
      reading_2,
      reading_3
    FROM vw_raw_data
    """)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    mlflow.log_metric("Step Elapsed Time", elapsed_time, 3)

    ########### STEP 4: Assemble the data into label and features columns (CREATE VECTORS)
    
    start_time = time.time()
    
    assembler = VectorAssembler( 
    inputCols=["device_type", "device_id", "reading_1", "reading_2", "reading_3"], 
    outputCol="features")

    df_assembled_data = assembler.transform(df_raw_data).select("label", "features")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    mlflow.log_metric("Step Elapsed Time", elapsed_time, 4)

    ########### STEP 5: Randomly split data into training and test sets. Set seed for reproducibility
    start_time = time.time()
    
    (training_data, test_data) = df_assembled_data.randomSplit([0.7, 0.3], seed=100)
    
    # Log the size of the training and test data

    mlflow.log_metric("Training Data Rows", training_data.count())
    mlflow.log_metric("Test Data Rows", test_data.count())
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    mlflow.log_metric("Step Elapsed Time", elapsed_time, 5)

    ########### STEP 6: Train the model (FIT)
    start_time = time.time()
    
    # Select the Decision Tree model type, and set its parameters
    dtClassifier = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    dtClassifier.setMaxDepth(p_max_depth)
    dtClassifier.setMaxBins(20) # This is how Spark decides if a feature is categorical or continuous

    # Train the model
    model = dtClassifier.fit(training_data)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    mlflow.log_metric("Step Elapsed Time", elapsed_time, 6)

    ########### STEP 7: Test the model (TRANSFORM)
    
    # We'll use an MLflow metric to log the time taken in each step 
    start_time = time.time()
    
    df_predictions = model.transform(test_data)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    mlflow.log_metric("Step Elapsed Time", elapsed_time, 7)

    ########### STEP 8: Determine the model's accuracy / precision    (EVALUATE)
    start_time = time.time()
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(df_predictions, {evaluator.metricName: "accuracy"})
    
    # NOTE: this is a 1-time metric, not a series
    
    mlflow.log_metric("Accuracy", accuracy)

    # Log the model's feature importances in MLflow
    mlflow.log_param("Feature Importances", str(model.featureImportances))
    
    # We'll use an MLflow metric to log the time taken in each step 
    end_time = time.time()
    elapsed_time = end_time - start_time

    mlflow.log_metric("Step Elapsed Time", elapsed_time, 8)
    
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time

    mlflow.log_metric("Overall Elapsed Time", overall_elapsed_time)
    # Log the model itself
    mlflow_spark.log_model(model, "spark-model")
    
    return run.info.run_uuid

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 3: Iteratively Train and Test

# COMMAND ----------

dbutils.widgets.text("max_depth", "2", "max_depth")

# COMMAND ----------

# Train and test a model.  Run this several times using different parameter values.  
max_depth = int(dbutils.widgets.get("max_depth"))

for i in range(1, max_depth):
  run_info = training_run(p_max_depth = i, p_owner = team_name)
  print(f"""
------- Run Complete -------
Run ID: {run_info}
Max Depth: {i}""")


# COMMAND ----------


