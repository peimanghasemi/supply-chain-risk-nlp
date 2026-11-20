// Aggregates classified risk events per supplier and builds a simple risk score with Spark.
// Input: CSV with columns supplier, event_type, severity, date (e.g. output of the classifiers).
// Run: spark-submit --class RiskAggregation target/scala-2.12/risk-aggregation.jar events.csv out/

import org.apache.spark.sql.{SparkSession, functions => F}
import org.apache.spark.sql.expressions.Window

object RiskAggregation {
  def main(args: Array[String]): Unit = {
    val Array(input, output) = args
    val spark = SparkSession.builder.appName("SupplierRiskAggregation").getOrCreate()
    import spark.implicits._

    val events = spark.read.option("header", "true").csv(input)
      .filter($"event_type" =!= "no_risk")
      .withColumn("weight",
        F.when($"severity" === "high", 3).when($"severity" === "medium", 2).otherwise(1))
      .withColumn("date", F.to_date($"date"))

    val scores = events
      .groupBy($"supplier")
      .agg(
        F.count("*").as("n_events"),
        F.sum("weight").as("risk_score"),
        F.max("date").as("last_event"),
        F.collect_set("event_type").as("event_types"))

    val ranked = scores.withColumn("rank", F.rank().over(Window.orderBy($"risk_score".desc)))

    ranked.write.mode("overwrite").parquet(output)
    ranked.show(20, truncate = false)
    spark.stop()
  }
}
