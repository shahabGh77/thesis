from pyspark.sql.functions import col, when, lit, sum as sqlsum, exp, log, abs as sqlabs
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import numpy as np
import time

# from mGraph import mGraph


sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext.getOrCreate(sc)


class MRF:
    def __init__(self, V, E):
        self.V = V
        self.E = E
        self.Fi = 2
        #potential functions
        self.epsilon = 0.1
        self.si = {                                                 #  epssilon=.1 =>    +     bad good
            '+': np.array([[1-2*self.epsilon, 2*self.epsilon],      #                  fraud   .8  .2
                           [self.epsilon, 1-self.epsilon]]),        #                  honest  .1  .9
            '-': np.array([[2*self.epsilon, 1-2*self.epsilon],      #                    -     bad good
                           [1-self.epsilon, self.epsilon]])         #                  fraud   .2  .8
        }                                                           #                  honest  .9  .1


    def addRandomVariable(self, name, values, defaultProb, on='vertex'):
        for value in values:
            if on == 'vertex':
                self.V = self.V.withColumn(f'{name}={value}', lit(defaultProb))
            elif on == 'edge':
                self.E = self.E.withColumn(f'{name}={value}', lit(defaultProb))

    def initBP(self):
        self.E = self.E.withColumn('Mij[fraud]', lit(1))
        self.E = self.E.withColumn('Mij[honest]', lit(1))
        self.E = self.E.withColumn('Mji[bad]', lit(1))
        self.E = self.E.withColumn('Mji[good]', lit(1))

        self.V = self.V.withColumn('Fi[fraud]', lit(2))
        self.V = self.V.withColumn('Fi[honest]', lit(2))
        self.V = self.V.withColumn('Fi[bad]', lit(2))
        self.V = self.V.withColumn('Fi[good]', lit(2))

    def MUL(self, df, groupByField, columnsAndAliases):
        exprs = [exp(sqlsum(log(x))).alias(y) for x, y in columnsAndAliases]
        df = df.groupBy(groupByField).agg(*exprs)
        return df

    def selectSi(self, column):
        if 'bad' in column:
            si = {'+': self.si['+'][:, 0], '-':self.si['-'][:, 0]}
        elif 'good' in column:
            si = {'+': self.si['+'][:, 1], '-':self.si['-'][:, 1]}
        elif 'fraud' in column:
            si = {'+': self.si['+'][0], '-':self.si['-'][0]}
        elif 'honest' in column:
            si = {'+': self.si['+'][1], '-':self.si['-'][1]}
        return si


    def dot(self, df, columns):
        for column in columns:
            si = self.selectSi(column)
            df = df.withColumn(column, when(col('sign') >= 0,
                                          col('sign')*(self.Fi*col('nfraud')*si['+'][0] + self.Fi*col('nhonest')*si['+'][1])) 
                                          .otherwise( sqlabs('sign')*(self.Fi*col('nfraud')*si['-'][0] + self.Fi*col('nhonest')*si['-'][1])))
        return df
        
    def normaliser(self, df, columns):
        cl = [col(c) for c in columns]
        nsum = [cl[i]+cl[i+1] for i in range(len(cl)-1)][-1]
        df = df.withColumn('nsum', col('Mji[bad]')+col('Mji[good]'))
        for column in columns:
            df = df.withColumn(column, col(column) / col('nsum'))
        return df

    def u2pMsg(self):
        """user to product message

        logic:                              

         Graph:             si:
                                                E            
                 u2         review+|bad good    +---+---+-----+----------+-----------+--------+---------+                
          ↓Mij /    \       -------|--------    |src|dst| sign|Mij[fraud]|Mij[honest]|Mji[bad]|Mji[good]|             
         ↑Mji /      \        fraud|0.8  0.2    +---+---+-----+----------+-----------+--------+---------+    prods         
             /        \      honest|0.1  0.9    | u2| p1|    1|       0.5|        0.5|     0.5|      0.5|    +---+---+-------+--------+
           p1----u3----p2                       | u2| p2|   -1|       0.5|        0.5|     0.5|      0.5|    |src|dst| nfraud| nhonest|
                      /     review-|bad good    | u3| p1|    1|       0.5|        0.5|     0.5|      0.5|    +---+---+-------+--------+
                     /      -------|--------    | u3| p2|    1|       0.5|        0.5|     0.5|      0.5| => | u2| p1|   0.25|    0.25|
                    /         fraud|0.2  0.8    | u4| p2|   -1|       0.5|        0.5|     0.5|      0.5|    | u3| p1|   0.25|    0.25|
                 u4          honest|0.9  0.1    +---+---+-----+----------+-----------+--------+---------+    +---+---+-------+--------+                              

        in order to simulate user to product message passing we do these steps:
        1) grouping by 'src' we have MUL(Mij[fraud]) as 'nfraud' and MUL(Mij[honest]) as 'nhonest', since there isn't such MUL() aggregate function in sql,
        MUL is implimented as described: MUL(colA) = EXP(SUM(LOG(colA))). at this time we've gathered all the messages

        prods * E                                          unormal
        +---+---+----------+-----------+------+-------+    +---+---+------+-------+----+--------+---------+
        |src|dst|Mij[fraud]|Mij[honest]|nfraud|nhonest|    |src|dst|nfraud|nhonest|sign|Mji[bad]|Mji[good]|
        +---+---+----------+-----------+------+-------+    +---+---+------+-------+----+--------+---------+
        | u2| p1|       0.5|        0.5|   0.5|    0.5|    | u2| p1|   0.5|    0.5|   1|    0.45|     0.55|
        | u2| p2|       0.5|        0.5|   0.5|    0.5|    | u2| p2|   0.5|    0.5|  -1|    0.55|     0.45|
        | u3| p1|       0.5|        0.5|   0.5|    0.5| => | u3| p1|   0.5|    0.5|   1|    0.45|     0.55|
        | u3| p2|       0.5|        0.5|   0.5|    0.5|    | u3| p2|   0.5|    0.5|   1|    0.45|     0.55|
        | u4| p2|       0.5|        0.5|      |       |    | u4| p2|      |       |  -1|        |         |
        +---+---+----------+-----------+------+-------+    +---+---+------+-------+----+--------+---------+

        2) for sending a message from 'A' to 'B', we need to collect all the message coming to 'A' except the message that comes from 'B', then do the 
        necessary calculation and send the result to 'B'. as mentioned in step 1, all the messages are gathered. the only thing we need to do is to remove 
        the message coming from the destination that we want to send the message to. to do so, we divide gathered messages
        'nfraud'('nhonest') by Mij[fraud](Mij[honest]) in the other word we have: nfraud/=Mij[fraud], nhonest/=Mij[honest]

        3) the rest is just a dot product: 
            Mji[bad]  = si_bad[0.8, 0.1] . [nfraud, nhonest]
            Mji[good] = si_good[0.2, 0.9] . [nfraud, nhonest]

        """
        self.E.cache()
        moreThan1NeighbourIds = self.E.groupBy(self.E.src).count().where('count > 1').withColumnRenamed('src', 'id').drop('count')
        moreThan1Neighbour = self.E.join(moreThan1NeighbourIds, self.E.src == moreThan1NeighbourIds.id).select(self.E.columns)
        prods = self.MUL(moreThan1Neighbour, 'src', [('Mij[fraud]', 'nfraud'), ('Mij[honest]', 'nhonest')]).withColumnRenamed('src', 'id')
        
        del moreThan1Neighbour, moreThan1NeighbourIds
        
        prods = self.E.join(prods, self.E.src == prods.id)
        prods = prods.withColumn('nfraud', col('nfraud')/col('Mij[fraud]')).withColumn('nhonest', col('nhonest')/col('Mij[honest]'))
        unormal = self.dot(prods, ['Mji[bad]', 'Mji[good]'])

        del prods
        normal = self.normaliser(unormal, ['Mji[bad]', 'Mji[good]'])
        
        del unormal
        diffIds = self.E.select('src', 'dst').subtract(normal.select('src', 'dst')).withColumnRenamed('src', 'srcc').withColumnRenamed('dst', 'dstt')
        diff = self.E.join(diffIds, (self.E.src == diffIds.srcc) & (self.E.dst == diffIds.dstt)).select(self.E.columns)
        self.E = normal.select(self.E.columns).union(diff)


    def p2uMsg(self):
        """product to user message

        logic: it's like u2pMsg()
        """
        self.E.cache()
        moreThan1NeighbourIds = self.E.groupBy(self.E.dst).count().where('count > 1').withColumnRenamed('dst', 'id').drop('count')
        moreThan1Neighbour = self.E.join(moreThan1NeighbourIds, self.E.dst == moreThan1NeighbourIds.id).select(self.E.columns)
        prods = moreThan1Neighbour.groupBy('dst') \
                .agg(exp(sqlsum(log('Mji[bad]'))).alias('nbad'), exp(sqlsum(log('Mji[good]'))).alias('ngood')) \
                .withColumnRenamed('dst', 'id')
        del moreThan1Neighbour, moreThan1NeighbourIds
        prods = self.E.join(prods, self.E.dst == prods.id)
        prods = prods.withColumn('nbad', col('nbad')/col('Mji[bad]')).withColumn('ngood', col('ngood')/col('Mji[good]'))
        unormal = prods.withColumn('Mij[fraud]',
                                     when(col('sign') >= 0,
                                          col('sign')*(self.Fi*col('nbad')*0.8 + self.Fi*col('ngood')*0.2))
                                          .otherwise( sqlabs('sign')*(self.Fi*col('nbad')*0.2 + self.Fi*col('ngood')*0.8) )) \
                       .withColumn('Mij[honest]',
                                    when(col('sign') >= 0,
                                         col('sign')*(self.Fi*col('nbad')*0.1 + self.Fi*col('ngood')*0.9)) 
                                         .otherwise( sqlabs('sign')*(self.Fi*col('nbad')*0.9 + self.Fi*col('ngood')*0.1)) )
        del prods
        normal = unormal.withColumn('psum', col('Mij[fraud]')+col('Mij[honest]')) \
                        .withColumn('Mij[fraud]', col('Mij[fraud]') / col('psum')) \
                        .withColumn('Mij[honest]', col('Mij[honest]') / col('psum'))
        del unormal
        diffIds = self.E.select('src', 'dst').subtract(normal.select('src', 'dst')).withColumnRenamed('src', 'srcc').withColumnRenamed('dst', 'dstt')
        diff = self.E.join(diffIds, (self.E.src == diffIds.srcc) & (self.E.dst == diffIds.dstt)).select(self.E.columns)
        self.E = normal.select(self.E.columns).union(diff)


    def spBelief(self):
        clms = self.V.columns
        unormalU = self.E.groupBy('src') \
                .agg(exp(sqlsum(log('Mij[fraud]'))).alias('P[fraud]'), exp(sqlsum(log('Mij[honest]'))).alias('P[honest]')) \
                .withColumn('P[fraud]', self.Fi*col('P[fraud]')) \
                .withColumn('P[honest]', self.Fi*col('P[honest]'))
        normalU = unormalU.withColumn('usum', col('P[fraud]')+col('P[honest]')) \
                          .withColumn('P[fraud]', col('P[fraud]') / col('usum')) \
                          .withColumn('P[honest]', col('P[honest]') / col('usum'))        
        self.V = self.V.join(normalU, self.V.id == normalU.src, 'leftouter')
        unormalP = self.E.groupBy('dst') \
                .agg(exp(sqlsum(log('Mji[bad]'))).alias('P[bad]'), exp(sqlsum(log('Mji[good]'))).alias('P[good]')) \
                .withColumn('P[bad]', self.Fi*col('P[bad]')) \
                .withColumn('P[good]', self.Fi*col('P[good]'))
        normalP = unormalP.withColumn('psum', col('P[bad]')+col('P[good]')) \
                          .withColumn('P[bad]', col('P[bad]') / col('psum')) \
                          .withColumn('P[good]', col('P[good]') / col('psum')) 
        clms.extend(['P[fraud]', 'P[honest]', 'P[bad]', 'P[good]'])
        self.V = self.V.join(normalP, self.V.id == normalP.dst, 'leftouter').select(clms)


def firstRun():
    V = spark.read.parquet('../data/videoGames_Vertices.parquet')
    E = spark.read.parquet('../data/videoGames_Edges.parquet')
    m = MRF(V, E)
    m.initBP()
    m.u2pMsg()
    m.E.show()




def mrfIterrate(iteration):
    V = spark.read.parquet('../data/videoGames_Vertices.parquet')
    
    for i in range(iteration):
        E = spark.read.parquet(f'../data/MRF-E{i+1}.parquet')
        m = MRF(V, E)
        start_time = time.time()
        m.u2pMsg()
        m.p2uMsg()
        m.E.write.parquet(f'../data/MRF-E{i+2}.parquet')
        del m
        spark.catalog.clearCache()
        print(f"\n\n\n---iteration {i+1} ===> {time.time() - start_time} seconds ---\n\n\n")


def blf(iteration):
    for i in range(iteration):
        V = spark.read.parquet('../data/videoGames_Vertices.parquet')
        E = spark.read.parquet(f'../data/MRF-E{i+1}.parquet')
        m = MRF(V, E)
        start_time = time.time()
        m.spBelief()
        m.V.write.parquet(f'../data/MRF-V{i+1}.parquet')
        del m
        spark.catalog.clearCache()
        print(f"\n\n---iteration {i+1} ===> {time.time() - start_time} seconds ---\n\n")


# E = spark.read.parquet('../data/MRF/MRF-E1.parquet')
# m = MRF(1, E)
# m.normaliser(m.E, ['Mij[fraud]', 'Mij[honest]']).show()
# m.E.show()
firstRun()