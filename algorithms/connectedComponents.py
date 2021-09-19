
from pyspark.sql.functions import col, lit, when

from pyspark import SparkContext
sc = SparkContext("local", "First App")

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

from pyspark.sql.session import SparkSession
spark = SparkSession(sc)

def isSameComponent(V, src, dst):
    c1 = V.filter(V.id == src).select('component').collect()[0][0]
    c2 = V.filter(V.id == dst).select('component').collect()[0][0]
    return True if c1 == c2 else False


def addEdge(V, src, dst, res):
    if not isSameComponent(V, src, dst):
        res.append((src, dst))


def getCcEdges(V, E):
    """determines edges between components

    Args:
        V (Dataframe[id, component]): nodes and component which they blong to
        E (Dataframe[src, dst]): whole graph edges

    Returns:
        Dataframe[src, dst]: edges between components
    """
    edges = E.join(V, E.src == V.id).select(E.src, E.dst, col('component').alias('srcComponent'))
    edges = edges.join(V, edges.dst == V.id).select('src', 'dst', 'srcComponent', col('component').alias('dstComponent'))
    edges = edges.where('srcComponent <> dstComponent')
    edges.show()


def example():
    E = spark.read.parquet('data/videoGames_Edges.parquet')
    cc = spark.read.parquet('data/videoGames_Connected_Component.parquet')
    E = E.select(E.src, E.dst)
    V = cc.select(cc.id, cc.component)
    res = getCcEdges(V, E)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(len(res))
    print(res)