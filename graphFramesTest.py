from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *

from pyspark import SparkContext
sc = SparkContext("local", "First App")

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

vertices = sqlContext.createDataFrame([
  ("a", "Alice", 34),
  ("b", "Bob", 36),
  ("c", "Charlie", 30),
  ("d", "David", 29),
  ("e", "Esther", 32),
  ("f", "Fanny", 36),
  ("g", "Gabby", 60)], ["id", "name", "age"])


edges = sqlContext.createDataFrame([
  ("a", "b", "friend"),
  ("b", "c", "follow"),
  ("c", "b", "follow"),
  ("f", "c", "follow"),
  ("e", "f", "follow"),
  ("e", "d", "friend"),
  ("d", "a", "friend"),
  ("a", "e", "friend")
], ["src", "dst", "relationship"])


print('yesssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
g = GraphFrame(vertices, edges)
print(g)

from igraph import *
ig = Graph.TupleList(g.edges.collect(), directed=True)
plot(ig)