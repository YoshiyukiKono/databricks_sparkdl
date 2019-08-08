# CDSW Python 2 / 2Core/ 16GB
 
# git clone https://github.com/databricks/spark-deep-learning
# cd spark-deep-learning
# build/sbt assembly
 
# !pip install --upgrade keras
# !pip install --upgrade tensorflow
# !pip install numpy --upgrade
# !pip install image
 
# Read https://medium.com/linagora-engineering/making-image-classification-simple-with-spark-deep-learning-f654a8b876b8
# wget https://github.com/zsellami/images_classification/raw/master/personalities.zip
# unzip to HDFS
 
# Configure the Spark Requirements
import cdsw
import tensorflow as tf
 
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.functions import lit
 
# Establish the Spark Session
conf = SparkConf().set("spark.jars", "/home/cdsw/spark-deep-learning/target/scala-2.11/spark-deep-learning-assembly-1.5.1-SNAPSHOT-spark2.4.jar")
sc = SparkContext( conf=conf)
 
# Add in the sparkdl Dependancies
sys.path.insert(0, "/home/cdsw/spark-deep-learning/target/scala-2.11/spark-deep-learning-assembly-1.5.1-SNAPSHOT-spark2.4.jar")
 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
 
from sparkdl import DeepImageFeaturizer
 
from pyspark.ml.image import ImageSchema
label1_df = ImageSchema.readImages("data/personalities/jobs").withColumn("label", lit(0))
label2_df = ImageSchema.readImages("data/personalities/zuckerberg").withColumn("label", lit(1))
train1_df, test1_df = label1_df.randomSplit([0.6, 0.4])
train2_df, test2_df = label2_df.randomSplit([0.6, 0.4])
train1_df.show()
test1_df.show()
 
train_images_df = train1_df.unionAll(train2_df)
test_images_df = test1_df.unionAll(test2_df)
 
# Training Set
train_images_df.show()
 
# Test Set
test_images_df.show()
 
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
 
model = p.fit(train_images_df)    # train_images_df is a dataset of images and labels
 
# Inspect training error
df = model.transform(test_images_df).select("image", "label", "prediction")
df.show()
 
predictionAndLabels = df.select("label", "prediction")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
 
sc.stop()