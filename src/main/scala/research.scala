import org.apache.spark.sql.Dataset
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

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



  val assembler = new VectorAssembler()
    .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
    .setOutputCol("features")

  val dfvr = assembler.transform(rawDF)



  val scaler = new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  val scalerModel = scaler.fit(dfvr)

  val scaledData = scalerModel.transform(dfvr)

  val readydata = scaledData.select("scaledFeatures", "species")

  readydata.show()

  val indexer = new StringIndexer()
    .setInputCol("species")
    .setOutputCol("indexedLabel")
    .fit(readydata)

  val indexeddata = indexer.transform(readydata)

  val inputColSchema = indexeddata.schema(indexer.getOutputCol)

  val Array(trainingData, testData) = indexeddata.randomSplit(Array(0.7, 0.3))


  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("scaledFeatures")
    .setNumTrees(10)

  val model = rf.fit(trainingData)

  val predictions = model.transform(testData)


  val labelConvertedDF = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(indexer.labelsArray(0))
    .transform(predictions)


  labelConvertedDF.select("predictedLabel", "species", "probability", "prediction").show(10, truncate = false)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test Error = ${(1.0 - accuracy)}")

  model.write.overwrite().save("src/main/outputmodel")


}
