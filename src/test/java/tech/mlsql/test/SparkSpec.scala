package tech.mlsql.test

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession, SparkUtils}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import tech.mlsql.arrow.python.ispark._
import tech.mlsql.arrow.python.runner.{ArrowPythonRunner, ChainedPythonFunctions, PythonConf, PythonFunction}
import tech.mlsql.common.utils.lang.sc.ScalaMethodMacros.str
import tech.mlsql.common.utils.log.Logging
import tech.mlsql.test.function.SparkFunctions.MockData

import java.util
import scala.collection.JavaConverters._

/**
 * 2019-08-14 WilliamZhu(allwefantasy@gmail.com)
 */
class SparkSpec extends AnyFunSuite with BeforeAndAfterAll with Logging {

  var _session: Option[SparkSession] = None
  var rayEnv = new RayEnv()

  def condaEnv = "source /Users/allwefantasy/opt/anaconda3/bin/activate ray-dev"

  //spark.executor.heartbeatInterval
  test("spark") {
    val session = _session.get
    import session.implicits._
    val timezoneid = session.sessionState.conf.sessionLocalTimeZone
    val df = session.createDataset[String](Seq("a1", "b1")).toDF("value")
    val struct = df.schema
    val abc = df.rdd.mapPartitions { iter =>
      val enconder = RowEncoder.apply(struct).resolveAndBind()
      val envs = new util.HashMap[String, String]()
      envs.put(str(PythonConf.PYTHON_ENV), "source activate ray-1.12.0 && export ARROW_PRE_0_15_IPC_FORMAT=1")
      val batch = new ArrowPythonRunner(
        Seq(ChainedPythonFunctions(Seq(PythonFunction(
          """
            |import pandas as pd
            |import numpy as np
            |for item in data_manager.fetch_once():
            |    print(item)
            |df = pd.DataFrame({'AAA': [4, 5, 6, 7],'BBB': [10, 20, 30, 40],'CCC': [100, 50, -30, -50]})
            |data_manager.set_output([[df['AAA'],df['BBB']]])
          """.stripMargin, envs, "python", "3.6")))), struct,
        timezoneid, Map()
      )
      val toRow = enconder.createSerializer()
      val newIter = iter.map { irow =>
        toRow(irow)
      }
      val commonTaskContext = new SparkContextImp(TaskContext.get(), batch)
      val columnarBatchIter = batch.compute(Iterator(newIter), TaskContext.getPartitionId(), commonTaskContext)
      columnarBatchIter.flatMap { batch =>
        batch.rowIterator.asScala
      }.map(f => f.copy())
    }

    val dataDF = session.createDataset(Range(0, 100).map(i => MockData(s"Title${i}", s"body-${i}"))).toDF()
    rayEnv.startDataServer(dataDF)

    val df2 = session.createDataset(rayEnv.dataServers).toDF()

    val envs = new util.HashMap[String, String]()
    envs.put("PYTHON_ENV", s"${condaEnv};export ARROW_PRE_0_15_IPC_FORMAT=1")
    //envs.put("PYTHONPATH", (os.pwd / "python").toString())

    val aps = new ApplyPythonScript(rayEnv.rayAddress, envs, "asia/harbin")
    val rayAddress = rayEnv.rayAddress
    logInfo(rayAddress)
    val func = aps.execute(
      s"""
         |import ray
         |import time
         |from pyjava.api.mlsql import RayContext
         |import numpy as np;
         |ray_context = RayContext.connect(globals(),"${rayAddress}")
         |def echo(row):
         |    row1 = {}
         |    row1["title"]=row['title'][1:]
         |    row1["body"]= row["body"] + ',' + row["body"]
         |    return row1
         |ray_context.foreach(echo)
          """.stripMargin, df2.schema)

    val outputDF = df2.rdd.mapPartitions(func)

    val pythonServers = SparkUtils.internalCreateDataFrame(session, outputDF, df2.schema).collect()

    val rdd = session.sparkContext.makeRDD[Row](pythonServers, numSlices = pythonServers.length)
    val pythonOutputDF = rayEnv.collectResult(rdd)
    val output = SparkUtils.internalCreateDataFrame(session, pythonOutputDF, dataDF.schema).collect()
    assert(output.length == 100)
    output.zipWithIndex.foreach({
      case (row, index) =>
        assert(row.getString(0) == s"itle${index}")
        assert(row.getString(1) == s"body-${index},body-${index}")
    })
  }

  override def beforeAll(): Unit = {
    _session = Some(SparkSession.builder().master("local[*]").appName("test").getOrCreate())
    super.beforeAll()
    rayEnv.startRay(condaEnv)
  }

  override def afterAll(): Unit = {
    _session.map(item=>item.sparkContext.stop())
    rayEnv.stopRay(condaEnv)
    super.afterAll()
  }

}
