import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.regexp_replace



object KafkaHandler {
  def main(args: Array[String]): Unit = {
    //if (args.length != 3) {
      println("Usage: SparkML <path-to-model> <path-to-input> <path-to-output>")
      //sys.exit(-1)
    }

  val spark = SparkSession
    .builder()
    .appName("Kafka handler on Spark ML")
    .config("spark.master", "local")
    .getOrCreate()

  val topicinput = "iris_input"

  import spark.implicits._
  val df = spark
    .read
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", topicinput)
    .load()

  df.show(10)

  val df1 = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    .as[(String, String)]

  import spark.sqlContext.implicits._

  val df2 = df1.withColumn("sepal_length_str", split(col("value"), " ").getItem(0))
    .withColumn("sepal_width_str", split(col("value"), " ").getItem(1))
    .withColumn("petal_length_str", split(col("value"), " ").getItem(2))
    .withColumn("petal_width_str", split(col("value"), " ").getItem(3))


  val df3 = df2.withColumn("sepal_length", col("sepal_length_str").cast("Double"))
                   .withColumn("sepal_width", col("sepal_width_str").cast("Double"))
                   .withColumn("petal_length", col("petal_length_str").cast("Double"))
                   .withColumn("petal_width", col("petal_width_str").cast("Double"))

  df3.show()
  df3.printSchema()


  val model = PipelineModel.load("src/main/outputmodel")
  val prediction = model.transform(df3)


  val selecteddata = prediction.select("sepal_length", "sepal_width", "petal_length", "petal_width", "predictedLabel")

  val assembleddata = selecteddata.withColumn("value",
    concat(col("sepal_length"), lit(" ")
      , col("sepal_width"), lit(" ")
      , col("petal_length"), lit(" ")
      , col("petal_width"), lit(" ")
      , col("predictedLabel")
    ))
    .drop("sepal_length", "sepal_width", "petal_length", "petal_width", "predictedLabel")

  assembleddata.show()

  val topic = "iris_prediction"

  assembleddata.selectExpr( "CAST(value AS STRING)")
    .write
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("topic", topic)
    .save()

  println("\n \n data saved \n \n")

}
