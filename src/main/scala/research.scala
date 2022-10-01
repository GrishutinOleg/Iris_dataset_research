import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.Pipeline

object research extends App {

  val spark = SparkSession
    .builder()
    .appName("Spark ML for IRIS dataset")
    .config("spark.master", "local")
    .getOrCreate()

  val rawDF = spark.read.format("csv")
    .option("sep", ",")
    .option("inferSchema", "true")
    .option("header", "true")
    .load("src/main/resourses/IRIS.csv")

  rawDF.show(5, truncate = false)

  val columns: Array[String] = rawDF.columns

  columns.foreach(x => println(x))

  rawDF.printSchema

  rawDF.dtypes.foreach { dt => println(f"${dt._1}%25s\t${dt._2}") }

  rawDF.dtypes.groupBy(_._2).mapValues(_.length).foreach(println)

  val numericColumns: Array[String] = rawDF.dtypes.filter(p => p._2.equals("DoubleType") || p._2.equals("IntegerType")).map(_._1)
  rawDF.select(numericColumns.map(col): _*).summary().show

  val indexer = new StringIndexer()
    .setInputCol("species")
    .setOutputCol("indexedLabel")
    .fit(rawDF)

  val indexeddata = indexer.transform(rawDF)

  val inputColSchema = indexeddata.schema(indexer.getOutputCol)

  val assembler = new VectorAssembler()
    .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
    .setOutputCol("features")

  val dfvr = assembler.transform(indexeddata)

  val scaler = new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  val scaledData = scaler.fit(dfvr).transform(dfvr)

  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("scaledFeatures")
    .setNumTrees(10)


  val labelConvertedDF = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(indexer.labelsArray(0))

  val pipeline = new Pipeline()
    .setStages(Array(indexer,
                     assembler,
                     scaler,
                     rf,
                     labelConvertedDF
    ))

  val Array(trainingData, testData) = rawDF.randomSplit(Array(0.7, 0.3))

  val model = pipeline.fit(trainingData)

  val predictions = model.transform(testData)

  predictions.show(7)

  predictions.select("predictedLabel", "species", "probability", "prediction").show(10, truncate = false)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"\n Test Error = ${(1.0 - accuracy)} \n")

  model.write.overwrite().save("src/main/outputmodel")


}
