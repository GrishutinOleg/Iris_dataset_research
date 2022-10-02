import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.struct
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._


object KafkaProducer extends App {

  val spark = SparkSession
    .builder()
    .appName("Kafka producer on Spark")
    .config("spark.master", "local")
    .getOrCreate()

  val rawdata = spark.read.format("csv")
    .option("sep", ",")
    .option("inferSchema", "true")
    .option("header", "true")
    .load("src/main/resourses/IRIS.csv")

  val selecteddata = rawdata.select("sepal_length", "sepal_width", "petal_length", "petal_width")

  selecteddata.show(5, truncate = false)

  val topic = "iris_input"


  val assembleddata = selecteddata.withColumn("value",
  concat(col("sepal_length"), lit(" ")
    , col("sepal_width"), lit(" ")
    , col("petal_length"), lit(" ")
    , col("petal_width")
  ))

  assembleddata.show()

  assembleddata.selectExpr( "CAST(value AS STRING)")
    .write
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("topic", topic)
    .save()

  println("\n \n data saved \n \n")

}
