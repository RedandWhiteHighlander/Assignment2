import findspark
findspark.init()
findspark.find()
import warnings

#Loading the libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

#This is to begin the spark session
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

#This is to dataset loading
df = spark.read.format("csv").load("file:///home/hadoop/TrainingDataset.csv" , header = True ,sep =";")
df.printSchema()
df.show()

#This is to changing column name from 'quality'  to 'label'
for col_name in df.columns[1:-1]+['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "label")


#This is to getting  features and label seperately and converting them to a numpy array
features =np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('label').collect())

#This is to create  feature vector
VectorAssembler = VectorAssembler(inputCols = df.columns[1:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(df)
df_tr = df_tr.select(['features','label'])

#This following function makes the labeled point and parallelize it to perform RDD conversion
def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points) 

#The dataset of the converted RDD
dataset = to_labeled_point(sc, features, label)

#This is to Split the dataset into training and test
training, test = dataset.randomSplit([0.7, 0.3],seed =11)


#This is to fabricating a random forest training classifier
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)

#This is to predict
predictions = RFmodel.predict(test.map(lambda x: x.features))
#predictionAndLabels = test.map(lambda x: (float(model.predict(x.features)), x.label))

#This is to retrieving a RDD of label and predictions
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

labelsAndPredictions_df = labelsAndPredictions.toDF()
#cpnverting rdd ==> spark dataframe ==> pandas dataframe 

print()
print('^^^^^ Converting RDD - using spark datafrane -- and pandas dataframe ^^^^^') 
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()


#This is to Calculate F1- scoreprint()
print()
print('^^^^^ Calculating F1Score ^^^^^') 
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print()
print('^^^^^ Confusion matrix ^^^^^')
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))

print()
print('^^^^^ Classification report ^^^^^')
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print()
print('^^^^^ Accuracy Score ^^^^^')
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

#This is to calculate test error
print()
print('^^^^^ Calculating Test Errors ^^^^^')
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())    
print('Test Error = ' + str(testErr))

#This is to save training model
RFmodel.save(sc, 's3://mybucket-assignment2/trainingmodel.model')