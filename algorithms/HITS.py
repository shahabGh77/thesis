from pyspark.sql.functions import col, lit, when

from pyspark import SparkContext
sc = SparkContext("local", "First App")

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


def hubsNormalisation(V):
    max_h_score = V.agg({'h_score': 'max'}).collect()[0][0]
    return V.withColumn('h_score', V.h_score/max_h_score)

def authsNormalisation(V):
    max_a_score = V.agg({'a_score': 'max'}).collect()[0][0]
    return V.withColumn('a_score', V.a_score/max_a_score)

def hubsUpdate(V, E):
    tmp = V.join(E, V.id == E.dst).groupBy(E.src).sum('a_score').withColumnRenamed("sum(a_score)", "h_score").withColumnRenamed("src", "id")
    tmp = V.join(tmp, V.id == tmp.id, 'leftouter').withColumn('h_score', when(col('h_score').isNotNull(), tmp.h_score).otherwise(0)).select(V.id, 'h_score')
    return hubsNormalisation(tmp)

def authsUpdate(V, E):
    tmp = V.join(E, V.id == E.src).groupBy(E.dst).sum('h_score').withColumnRenamed("sum(h_score)", "a_score").withColumnRenamed("dst", "id")
    tmp = V.join(tmp, V.id == tmp.id, 'leftouter').withColumn('a_score', when(col('a_score').isNotNull(), tmp.a_score).otherwise(0)).select(V.id, 'a_score')
    return authsNormalisation(tmp)


def HITS(V, E, iteration, hubs=None, auths=None, writeToFile=False):
    if not hubs: hubs  = V.withColumn('h_score', lit(1))
    if not auths: auths = V.withColumn('a_score', lit(0))

    for i in range(iteration):
        auths = authsUpdate(hubs, E)
        hubs  = hubsUpdate(auths, E)
        
        if(writeToFile):
            print(f'\n\n###################### step {i+1} ########################')
            hubs.write.parquet(f'data/videoGames_hubs{i+1}.parquet')
            auths.write.parquet(f'data/videoGames_auths{i+1}.parquet')

    return hubs, auths




vertices = sqlContext.createDataFrame([
    ('A', ),
    ('B', ),
    ('C', ),
    ('D', ),
    ('E', )], ["id", ])


edges = sqlContext.createDataFrame([
    ('A', 'B'),
    ('A', 'C'),
    ('A', 'D'),
    ('B', 'A'),
    ('B', 'D'),
    ('C', 'E'),
    ('D', 'B'),
    ('D', 'C')], ["src", "dst", ])

hubs, auths = HITS(vertices, edges, 5)

hubs.show()
auths.show()
