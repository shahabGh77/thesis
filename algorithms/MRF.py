from pyspark.sql.functions import col, when, lit, sum as sqlsum, exp, log, abs as sqlabs
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pathlib import Path
from glob import glob
import numpy as np
import time
import random


sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext.getOrCreate(sc)


class MRF:
    def __init__(self, V, E, maxDeg=None, sampleSize=None):
        self.V = V
        self.E = E
        self.epsilon = 0.1
        self.base = 10
        self.maxDeg = 1000 if maxDeg is None else maxDeg
        self.sampleSize = 500 if sampleSize is None else sampleSize
        self.signCol = 's'
        #potential functions
        self.si = {                                                 #  epssilon=.1 =>    +     bad good
            '+': np.array([[1-2*self.epsilon, 2*self.epsilon],      #                  fraud   .8  .2
                           [self.epsilon, 1-self.epsilon]]),        #                  honest  .1  .9
            '-': np.array([[2*self.epsilon, 1-2*self.epsilon],      #                    -     bad good
                           [1-self.epsilon, self.epsilon]])         #                  fraud   .2  .8
        }                                                           #                  honest  .9  .1
        self.moreThan1ProductNeighbour, self.moreThan1UserNeighbour = self.findMoreThan1Neighbours()
        
    def findMoreThan1Neighbours(self):
        moreThan1ProductNeighbour = self.V.filter((col('type') == 'u') & (col('deg') > 1)) \
                                            .withColumn('ratio', self.sampleSize/col('deg')) \
                                            .select('id', 'ratio', 'deg').cache().repartition('deg')

        moreThan1UserNeighbour = self.V.filter((col('type') == 'p') & (col('deg') > 1)) \
                                        .withColumn('ratio', self.sampleSize/col('deg')) \
                                        .select('id', 'ratio', 'deg').cache().repartition('deg')
        return moreThan1ProductNeighbour, moreThan1UserNeighbour    

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

        self.E = self.E.withColumn('Fi[fraud]', lit(2))
        self.E = self.E.withColumn('Fi[honest]', lit(2))
        self.E = self.E.withColumn('Fi[bad]', lit(2))
        self.E = self.E.withColumn('Fi[good]', lit(2))
        
        self.V = self.V.withColumn('Fi[fraud]', lit(2))
        self.V = self.V.withColumn('Fi[honest]', lit(2))
        self.V = self.V.withColumn('Fi[bad]', lit(2))
        self.V = self.V.withColumn('Fi[good]', lit(2))
        self.repartitionEdges()

    def repartitionEdges(self):
        self.E = self.E.join(self.moreThan1ProductNeighbour.select('id', 'count'), col('src') == col('id'), 'leftouter') \
                       .withColumn('scu', when(col('id').isNull(), lit(True)).otherwise(lit(False))) \
                       .withColumn('uDeg', when(col('id').isNull(), lit(1)).otherwise(col('count'))).drop('id', 'count')
        
        self.E = self.E.join(self.moreThan1UserNeighbour.select('id', 'count'), col('dst') == col('id'), 'leftouter') \
                       .withColumn('scp', when(col('id').isNull(), lit(True)).otherwise(lit(False))) \
                       .withColumn('pDeg', when(col('id').isNull(), lit(1)).otherwise(col('count'))).drop('id', 'count')

        self.E = self.E.repartition('scu', 'scp', 'uDeg', 'pDeg')


    def updateFi(self, df):
        eClms = self.E.columns
        vClms = self.V.columns
        df = df.withColumnRenamed('src', 'ssrc').withColumnRenamed('dst', 'ddst').drop('unixReviewTime')       

        self.E = self.E.join(df, ((self.E.dst == df.ddst) & (self.E.src == df.ssrc)), 'leftouter') \
                       .withColumn('Fi[bad]', when(col('count').isNull(), col('Fi[bad]')).otherwise(col('Fi[bad]')*col('count'))) \
                       .withColumn('Fi[fraud]', when(col('count').isNull(), col('Fi[fraud]')).otherwise(col('Fi[fraud]')*col('count'))) \
                       .select(eClms)

        self.V = self.V.join(df, self.V.id == df.ssrc, 'leftouter') \
                       .withColumn('Fi[fraud]', when(col('count').isNull(), col('Fi[fraud]')).otherwise(col('Fi[fraud]')*col('count'))) \
                       .select(vClms)



    def MUL(self, df, groupByField, columnsAndAliases):
        exprs = [exp(sqlsum(log(x))).alias(y) for x, y in columnsAndAliases]
        df = df.groupBy(groupByField).agg(*exprs)
        return df

    def preDot(self, column):
        if 'bad' in column:
            si = {'+': self.si['+'][:, 0], '-':self.si['-'][:, 0]}
            firstCol, secondCol = 'nfraud', 'nhonest'
        elif 'good' in column:
            si = {'+': self.si['+'][:, 1], '-':self.si['-'][:, 1]}
            firstCol, secondCol = 'nfraud', 'nhonest'
        elif 'fraud' in column:
            si = {'+': self.si['+'][0], '-':self.si['-'][0]}
            firstCol, secondCol = 'nbad', 'ngood'
        elif 'honest' in column:
            si = {'+': self.si['+'][1], '-':self.si['-'][1]}
            firstCol, secondCol = 'nbad', 'ngood'
        return si, firstCol, secondCol


    def dot(self, df, columns, signMagEffect):
        for column in columns:
            si, firstCol, secondCol = self.preDot(column)
            FiFirst, FiSecond = f'Fi[{firstCol[1:]}]', f'Fi[{secondCol[1:]}]'
            if signMagEffect:
                df = df.withColumn(column, when(col(self.signCol) > 0,
                                          col(self.signCol)*(col(FiFirst)*col(firstCol)*si['+'][0] + col(FiSecond)*col(secondCol)*si['+'][1])) 
                                          .otherwise( sqlabs(self.signCol)*(col(FiFirst)*col(firstCol)*si['-'][0] + col(FiSecond)*col(secondCol)*si['-'][1])))
            else:
                df = df.withColumn(column, when(col(self.signCol) > 0,
                                          col(FiFirst)*col(firstCol)*si['+'][0] + col(FiSecond)*col(secondCol)*si['+'][1]) 
                                          .otherwise( col(FiFirst)*col(firstCol)*si['-'][0] + col(FiSecond)*col(secondCol)*si['-'][1]))
        return df
        
    def rowNormaliser(self, df, columns, base=1):
        cl = [col(c) for c in columns]
        nsum = [cl[i]+cl[i+1] for i in range(len(cl)-1)][-1]
        df = df.withColumn('nsum', nsum)
        for column in columns:
            df = df.withColumn(column, base * col(column) / col('nsum'))
        return df

    def colNormaliser(self, df, columns):
        for column, alias in columns:
            maxVal = df.agg({column: 'max'}).collect()[0][0]
            df = df.withColumn(alias, col(column)/maxVal)
        return df

    def takeSample(self, sampleGroups, sampleIdx, space):
        fracDict = sampleGroups.toPandas().set_index('id').to_dict()['ratio']
        return space.sampleBy(sampleIdx, fractions=fracDict)

    def getEdges(self, groupByField):
        if groupByField == 'src':
            scCol = 'scu'
            degCol = 'udeg'
            sampleGroups = self.moreThan1ProductNeighbour.filter(col('deg') >= self.maxDeg).select('id', 'ratio')
        elif groupByField == 'dst':
            scCol = 'scp'
            degCol = 'pdeg'
            sampleGroups = self.moreThan1UserNeighbour.filter(col('deg') >= self.maxDeg).select('id', 'ratio')

        moreThan1Neighbour = self.E.filter(col(scCol) == False) # work with edges that their groupByField side has more than 1 neighbour
        normalSelected = moreThan1Neighbour.filter(col(degCol) < self.maxDeg)       # edges with deg < MAX_DEG are selected normally
        sampleSpace = moreThan1Neighbour.filter(col(degCol) >= self.maxDeg)          # we take a sample from edges with deg >= MAX_DEG

        if sampleSpace.rdd.isEmpty():
            return normalSelected
        sampleSelected = self.takeSample(sampleGroups, groupByField, sampleSpace)   #take a sample
        return normalSelected.union(sampleSelected)

    def u2pMsg(self, signMagEffect=True):
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
        moreThan1Neighbour = self.getEdges('src')
        prods = self.MUL(moreThan1Neighbour, 'src', [('Mij[fraud]', 'nfraud'), ('Mij[honest]', 'nhonest')]).withColumnRenamed('src', 'id')
        del moreThan1Neighbour
        prods = self.E.join(prods, self.E.src == prods.id)
        prods = prods.withColumn('nfraud', col('nfraud')/col('Mij[fraud]')).withColumn('nhonest', col('nhonest')/col('Mij[honest]'))
        unormal = self.dot(prods, ['Mji[bad]', 'Mji[good]'], signMagEffect)
        del prods
        normal = self.rowNormaliser(unormal, ['Mji[bad]', 'Mji[good]']).select(self.E.columns)
        del unormal

        lessThan1Neighbour = self.E.filter(col('scu') == True)
        self.E = normal.union(lessThan1Neighbour)


    def p2uMsg(self, signMagEffect=True):
        """product to user message

        logic: it's like u2pMsg()
        """
        self.E.cache()
        moreThan1Neighbour = self.getEdges('dst')
        prods = self.MUL(moreThan1Neighbour, 'dst', [('Mji[bad]', 'nbad'), ('Mji[good]', 'ngood')]).withColumnRenamed('dst', 'id')    
        del moreThan1Neighbour
        prods = self.E.join(prods, self.E.dst == prods.id)
        prods = prods.withColumn('nbad', col('nbad')/col('Mji[bad]')).withColumn('ngood', col('ngood')/col('Mji[good]'))
        unormal = self.dot(prods, ['Mij[fraud]', 'Mij[honest]'], signMagEffect)
        del prods
        normal = self.rowNormaliser(unormal, ['Mij[fraud]', 'Mij[honest]']).select(self.E.columns)
        del unormal

        lessThan1Neighbour = self.E.filter(col('scp') == True)
        self.E = normal.union(lessThan1Neighbour)


    def spBelief(self):
        clms = self.V.columns
        self.V = self.V.drop('P[fraud]').drop('P[honest]').drop('P[bad]').drop('P[good]')
        unormalU = self.MUL(self.getEdges('src'), 'src', [('Mij[fraud]', 'P[fraud]'), ('Mij[honest]', 'P[honest]')])
        self.V = self.V.join(unormalU, self.V.id == unormalU.src, 'leftouter').withColumn('P[fraud]', col('Fi[fraud]')*col('P[fraud]')) \
                                                                            .withColumn('P[honest]', col('Fi[honest]')*col('P[honest]'))
        # self.V = self.colNormaliser(self.V, [('P[fraud]', 'fFac'), ('P[honest]', 'hFac')])
        self.V = self.rowNormaliser(self.V, ['P[fraud]', 'P[honest]'])

        unormalP = self.MUL(self.getEdges('dst'), 'dst', [('Mji[bad]', 'P[bad]'), ('Mji[good]', 'P[good]')])
        # if not 'P[bad]' in clms:
        #     clms.extend(['P[fraud]', 'P[honest]', 'P[bad]', 'P[good]', 'fFac', 'hFac', 'bFac', 'gFac'])

        self.V = self.V.join(unormalP, self.V.id == unormalP.dst, 'leftouter').withColumn('P[bad]', col('Fi[bad]')*col('P[bad]')) \
                                                                            .withColumn('P[good]', col('Fi[good]')*col('P[good]'))     
        # self.V = self.colNormaliser(self.V, [('P[bad]', 'bFac'), ('P[good]', 'gFac')])  
        clms.extend(['P[fraud]', 'P[honest]', 'P[bad]', 'P[good]'])                                      
        self.V = self.rowNormaliser(self.V, ['P[bad]', 'P[good]']).select(clms)

        


def firstRun():
    V = spark.read.parquet('../data/videoGames_Vertices.parquet')
    E = spark.read.parquet('../data/videoGames_Edges_Sentiment.parquet')
    E = E.select('src', 'dst', 'overall', 'verified', 'vote', 'reviewText', col('compound').alias('sign'))
    moreThan1Review = spark.read.parquet('../data/videoGames_MoreThan1ReviewAtATime.parquet')
    m = MRF(V, E)
    m.initBP()
    m.updateFi(moreThan1Review)

    m.u2pMsg()
    # m.E.show()
    m.p2uMsg()
    return m
    # m.spBelief()
    # # m.E.show()
    # m.V.filter(col('P[fraud]').isNull() & col('P[bad]').isNull()).show()
    # m.V.filter(m.V.id == 'A71Z5AIGEFK11').show()
    print('\n\n\n\n\n')
    # m.E.filter(m.E.src == 'A71Z5AIGEFK11').show()

def messagePassing(iteration, maxDeg=None, sampleSize=None, startIndex=1, signMagEffect=True, vPath=None, ePath=None, rPath=None):
    if ePath is None:
        vPath = '../data/videoGames_VerticesLBP.parquet'
        ePath = '../data/videoGames_EdgesLBP.parquet'
    if vPath is None:
        rPath = '../data/MRF/'
    Path(rPath).mkdir(parents=True, exist_ok=True)

    V = spark.read.parquet(vPath)
    E = spark.read.parquet(ePath)
    
    # moreThan1Review = spark.read.parquet('../data/videoGames_MoreThan1ReviewAtATime.parquet')
    m = MRF(V, E, maxDeg, sampleSize)
    # m.updateFi(moreThan1Review)

    for i in range(startIndex, startIndex+iteration):
        start_time = time.time()
        m.u2pMsg(signMagEffect)
        m.p2uMsg(signMagEffect)
        print(f"\n---iteration {i} before writing===> {time.time() - start_time} seconds ---\n")
        m.E.write.parquet(rPath + f'MRF-E{i}.parquet')
        del m
        spark.catalog.clearCache()
        E = spark.read.parquet(rPath + f'MRF-E{i}.parquet')
        m = MRF(V, E, maxDeg, sampleSize)
        print(f"\n---iteration {i} after writing ===> {time.time() - start_time} seconds ---\n")
    
    # print('[Writing Final Result]')
    # start_time = time.time()
    # m.E.write.parquet(rPath + 'MRF-E_final.parquet')
    # print(f"\n---final write ===> {time.time() - start_time} seconds ---\n")


def addSingletoneBelief():
    pass    
        

def beliefExtraction(vPath, ePath, rPath):
    V = spark.read.parquet(vPath)
    E = spark.read.parquet(ePath)
    m = MRF(V, E)
    start_time = time.time()
    m.spBelief()
    m.V.write.parquet(rPath)
    print(f"\n---Beleifs Extracted ===> {time.time() - start_time} seconds ---\n")


def convergence(path):
    results = sorted(glob(path))
    report = {'columns': ['Fm', 'Hm', 'Bm', 'Gm']}
    for i, res in enumerate(results):
        print('reading: ', res)
        e = spark.read.parquet(res)
        d = e.agg(F.sum('Mij[fraud]').alias('F'), F.sum('Mij[honest]').alias('H'),
                    F.sum('Mji[bad]').alias('B'), F.sum('Mji[good]').alias('G')).first().asDict()

        report[f'MRF{i}'] = list(d.values())
        if i != 0:
            report[f'Diff{i}, {i-1}'] = [x2-x1 for (x2, x1) in zip(od, d.values())]
        od = d.values()
    return report

def convergencePrint(path):
    print('in path:', path)
    res = convergence(path)
    for k, v in res.items():
        print(k, ': ', v)