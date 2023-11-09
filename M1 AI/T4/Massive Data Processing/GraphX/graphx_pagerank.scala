import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

//load the graph
val graph = GraphLoader.edgeListFile(sc,"karate.txt")

val outDegrees: VertexRDD[Int] = graph.outDegrees

//add initial values to the attributes (pagerank of 0.0)
val initialGraph = graph.mapVertices((id, _) => 0.0)

//join the vertice with the degrees (for the PR computation)
//atributes will then be (pagerank, degree)
val prGraph = initialGraph.outerJoinVertices(outDegrees) { (id, oldAttr, outDegOpt) =>
  outDegOpt match {
    case Some(outDeg) => (oldAttr,outDeg)
    case None => (oldAttr,0)
}
}
//compute pagerank for 10 iterations
val pr = prGraph.pregel(0.0,10)(
    //new PR value => sum of messages * 0.85
    (id, attr, newSum) => (0.15 + 0.85 * newSum, attr._2),
    //send the pagerank value divided by the degree
    triplet => Iterator((triplet.dstId, triplet.srcAttr._1/triplet.srcAttr._2)),
    //sum all incoming pageranks to use for newSum above
    (a, b) => a+b
)

pr.vertices.collect()