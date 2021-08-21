from pyspark.sql.functions import col, when, lit, sum as sqlsum, exp, log, abs as sqlabs
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import pandas as pd
import time

# from mGraph import mGraph


sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext.getOrCreate(sc)


def updateRow(df, columnName, condition, value):
    return df.withColumn(columnName , when(condition, value).otherwise(col(columnName)))

class MRF:
    def __init__(self, V, E):
        self.V = V
        self.E = E
        self.Fi = 2
        #potential functions
        self.epsilon = 0.1
        self.si_pos = pd.DataFrame({'bad': [1-2*self.epsilon, self.epsilon], 'good': [2*self.epsilon, 1-self.epsilon]}, index=['fraud', 'honest'])
        self.si_neg = pd.DataFrame({'bad': [2*self.epsilon, 1-self.epsilon], 'good': [1-2*self.epsilon, self.epsilon]}, index=['fraud', 'honest'])
        pass

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
        
        # self.E.cache()
        # self.V.cache()


    def getNeighbours(self, id, side):
        # return spark.sql(f'SELECT `Mij[fraud]`, `Mij[honest]` from E where {side}="{id}"')
        return self.E.where(col(side) == id).withColumn('id', col(side))


    def userToProduct(self):
        for e in self.E.rdd.toLocalIterator():
            nbs = self.getNeighbours(e['src'], 'src').filter(col('id') != e['dst']).select('Mij[fraud]', 'Mij[honest]')
            if nbs.count() == 0:
                continue
            nbsToPan = nbs.toPandas()
            if nbsToPan.shape[0] == 1:
                prods = nbsToPan[['Mij[fraud]', 'Mij[honest]']].T.rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'})            
            else:
                prods = nbsToPan.prod(axis=0).rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'}).to_frame()

            Fi = self.V.filter(self.V.id == e['src']).select(col('Fi[fraud]').alias('fraud'), col('Fi[honest]').alias('honest')).toPandas().T
            sign = float(e['sign'])
            if sign > 0:
                unormal = sign * self.si_pos.T @ (Fi * prods)
            elif sign == 0:
                unormal = 0.0001 * self.si_pos.T @ (Fi * prods)
            else:
                unormal = abs(sign) * self.si_neg.T @ (Fi * prods)

            normal = unormal / unormal.sum()
            self.E = updateRow(self.E, 'Mji[bad]', (self.E.src == e["src"]) & (self.E.dst == e["dst"]), normal[0]['bad'])
            self.E = updateRow(self.E, 'Mji[good]', (self.E.src == e["src"]) & (self.E.dst == e["dst"]), normal[0]['good'])


    def u2pMsg(self):
        self.E.cache()
        moreThan1NeighbourIds = self.E.groupBy(self.E.src).count().where('count > 1').withColumnRenamed('src', 'id').drop('count')
        moreThan1Neighbour = self.E.join(moreThan1NeighbourIds, self.E.src == moreThan1NeighbourIds.id).select(self.E.columns)
        prods = moreThan1Neighbour.groupBy('src') \
                .agg(exp(sqlsum(log('Mij[fraud]'))).alias('nfraud'), exp(sqlsum(log('Mij[honest]'))).alias('nhonest')) \
                .withColumnRenamed('src', 'id')
        del moreThan1Neighbour, moreThan1NeighbourIds
        
        prods = self.E.join(prods, self.E.src == prods.id)
        prods = prods.withColumn('nfraud', col('nfraud')/col('Mij[fraud]')).withColumn('nhonest', col('nhonest')/col('Mij[honest]'))
        unormal = prods.withColumn('Mji[bad]',
                                     when(col('sign') >= 0,
                                          col('sign')*(self.Fi*col('nfraud')*0.8 + self.Fi*col('nhonest')*0.1)) 
                                          .otherwise( sqlabs('sign')*(self.Fi*col('nfraud')*0.2 + self.Fi*col('nhonest')*0.9))) \
                       .withColumn('Mji[good]',
                                    when(col('sign') >= 0,
                                         col('sign')*(self.Fi*col('nfraud')*0.2 + self.Fi*col('nhonest')*0.9)) 
                                         .otherwise( sqlabs('sign')*(self.Fi*col('nfraud')*0.8 + self.Fi*col('nhonest')*0.1))) 
        del prods
        normal = unormal.withColumn('usum', col('Mji[bad]')+col('Mji[good]')) \
                        .withColumn('Mji[bad]', col('Mji[bad]') / col('usum')) \
                        .withColumn('Mji[good]', col('Mji[good]') / col('usum'))
        del unormal
        diffIds = self.E.select('src', 'dst').subtract(normal.select('src', 'dst')).withColumnRenamed('src', 'srcc').withColumnRenamed('dst', 'dstt')
        diff = self.E.join(diffIds, (self.E.src == diffIds.srcc) & (self.E.dst == diffIds.dstt)).select(self.E.columns)
        self.E = normal.select(self.E.columns).union(diff)




    def p2uMsg(self):
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

    def productToUser(self):
        for e in self.E.rdd.toLocalIterator():
            nbs = self.getNeighbours(e['dst'], 'dst').filter(col('id') != e['src']).select('Mji[bad]', 'Mji[good]')
            if nbs.count() == 0:
                continue
            nbsToPan = nbs.toPandas()
            if nbsToPan.shape[0] == 1:
                prods = nbsToPan[['Mji[bad]', 'Mji[good]']].T.rename(index={'Mji[bad]': 'bad', 'Mji[good]': 'good'})            
            else:
                prods = nbsToPan.prod(axis=0).rename(index={'Mji[bad]': 'bad', 'Mji[good]': 'good'}).to_frame()

            Fi = self.V.filter(self.V.id == e['dst']).select(col('Fi[bad]').alias('bad'), col('Fi[good]').alias('good')).toPandas().T
            sign = float(e['sign'])
            if sign > 0:
                unormal = sign * self.si_pos @ (Fi * prods)
            elif sign == 0:
                unormal = 0.0001 * self.si_pos @ (Fi * prods)
            else:
                unormal = abs(sign) * self.si_neg @ (Fi * prods)
            normal = unormal / unormal.sum()
            self.E = updateRow(self.E, 'Mij[fraud]', (self.E.src == e["src"]) & (self.E.dst == e["dst"]), normal[0]['fraud'])
            self.E = updateRow(self.E, 'Mij[honest]', (self.E.src == e["src"]) & (self.E.dst == e["dst"]), normal[0]['honest'])
    
    def sumProduct(self):
        for v in self.V.rdd.toLocalIterator():
            if v['type'] == 'user':
                ps = self.E.filter(self.E.src == v['id']).select('Mij[fraud]', 'Mij[honest]').toPandas().prod(axis=0) \
                    .rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'}).to_frame()
                Fi = pd.DataFrame({0: [v['Fi[fraud]'], v['Fi[honest]']]}, index=['fraud', 'honest'])
                unormal = Fi * ps
                normal = unormal / unormal.sum()
                self.V = updateRow(self.V, 'Fi[fraud]', self.V.id == v["id"], normal[0]['fraud'])
                self.V = updateRow(self.V, 'Fi[honest]', self.V.id == v["id"], normal[0]['honest'])
                
            else:
                us = self.E.filter(self.E.dst == v['id']).select('Mji[bad]', 'Mji[good]').toPandas().prod(axis=0) \
                    .rename(index={'Mji[bad]': 'bad', 'Mji[good]': 'good'}).to_frame()
                Fi = pd.DataFrame({0: [v['Fi[bad]'], v['Fi[good]']]}, index=['bad', 'good'])
                unormal = Fi * us
                normal = unormal / unormal.sum()
                self.V = updateRow(self.V, 'Fi[bad]', self.V.id == v["id"], normal[0]['bad'])
                self.V = updateRow(self.V, 'Fi[good]', self.V.id == v["id"], normal[0]['good'])

    def mmap(self, e):
        nbs = self.getNeighbours(e['src'], 'src')
        if nbs.count() == 0:
            print(f'{i}] ', e['src'], e['dst'], 'passed')
        print(f'{i}] ', e['src'], e['dst'])


    def uu(self, e):
        nbs = self.getNeighbours(e['src'], 'src').filter(col('id') != e['dst']).select('Mij[fraud]', 'Mij[honest]')
        if nbs.count() == 0:
            print(f'{i}] ', e['src'], e['dst'], 'passed')
            return
        nbsToPan = nbs.toPandas()
        if nbsToPan.shape[0] == 1:
            prods = nbsToPan[['Mij[fraud]', 'Mij[honest]']].T.rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'})            
        else:
            prods = nbsToPan.prod(axis=0).rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'}).to_frame()

        Fi = self.V.filter(self.V.id == e['src']).select(col('Fi[fraud]').alias('fraud'), col('Fi[honest]').alias('honest')).toPandas().T
        sign = float(e['sign'])
        if sign > 0:
            unormal = sign * self.si_pos.T @ (Fi * prods)
        elif sign == 0:
            unormal = 0.0001 * self.si_pos.T @ (Fi * prods)
        else:
            unormal = abs(sign) * self.si_neg.T @ (Fi * prods)

        normal = unormal / unormal.sum()
        print(f'{i}] ', e['src'], e['dst'], sign, normal[0]['bad'], normal[0]['good'])
        self.E = updateRow(self.E, 'Mji[bad]', (self.E.src == e["src"]) & (self.E.dst == e["dst"]), normal[0]['bad'])
        self.E = updateRow(self.E, 'Mji[good]', (self.E.src == e["src"]) & (self.E.dst == e["dst"]), normal[0]['good'])

    def test(self):
        self.initBP()
        self.E.foreach(self.mmap)


    def LBP(self, iteration):
        self.initBP()
        for i in range(iteration):
            self.userToProduct()
            self.productToUser()
        self.sumProduct()

    def BP(self):
        self.initBP()
        self.userToProduct()
        self.E.write.parquet('data/userToProduct.parquet')
        self.productToUser()
        self.sumProduct()
        self.V.write.parquet('data/MRF_V.parquet')
        self.E.write.parquet('data/MRF_E.parquet')

    





def neighbours(E, id):
    return E.filter((col('src') == id) | (col('dst') == id)).withColumn('id', when(col('src') == id, col('dst')).otherwise(col('src')))

si_pos = pd.DataFrame({'bad': [0.8, 0.1], 'good': [0.2, 0.9]}, index=['fraud', 'honest'])
si_neg = pd.DataFrame({'bad': [0.2, 0.9], 'good': [0.8, 0.1]}, index=['fraud', 'honest'])

def userToProduct(V, E):
    for e in E.rdd.toLocalIterator():
        nbsToPan = neighbours(E, e['src']).filter(col('id') != e['dst']).select('Mij[fraud]', 'Mij[honest]').toPandas()
        if nbsToPan.shape[0] == 1:
            product = nbsToPan[['Mij[fraud]', 'Mij[honest]']].T.rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'})            
        else:
            product = nbsToPan.prod(axis=0).rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'}).to_frame()

        print('product\n---------------\n\n', product)
        Fi = V.filter(V.id == e['src']).select(col('Fi[fraud]').alias('fraud'), col('Fi[honest]').alias('honest')).toPandas().T
        print( '\n\n', e['sign'], type(e['sign']), '\n\n')
        if int(e['sign']) == 1:
            print(si_pos, '\n', Fi)
            unormal = si_pos.T @ (Fi * product)
        else:
            print('elllllllllllllllllllllllllllllllllllllllllllllllllllllse')
            print(si_neg, '\n', Fi)
            unormal = si_neg.T @ (Fi * product)
        normal = unormal / unormal.sum()
        print(unormal)
        print('\n\n\n')
        print(normal)
        print(normal[0]['bad'])
        print(normal[0]['good'])
        E = updateRow(E, 'Mji[bad]', (E.src == e["src"]) & (E.dst == e["dst"]), normal[0]['bad'])
        E = updateRow(E, 'Mji[good]', (E.src == e["src"]) & (E.dst == e["dst"]), normal[0]['good'])

    return V, E


def productToUser(V, E):
    for e in E.rdd.toLocalIterator():
        nbsPan = neighbours(E, e['dst']).filter(col('id') != e['src']).select('Mji[bad]', 'Mji[good]').toPandas()
        if nbsPan.shape[0] == 1:
            product = nbsPan[['Mji[bad]', 'Mji[good]']].T.rename(index={'Mji[bad]': 'bad', 'Mji[good]': 'good'})            
        else:
            product = nbsPan.prod(axis=0).rename(index={'Mji[bad]': 'bad', 'Mji[good]': 'good'}).to_frame()

        print('product\n---------------\n\n', product)
        Fi = V.filter(V.id == e['dst']).select(col('Fi[bad]').alias('bad'), col('Fi[good]').alias('good')).toPandas().T
        print( '\n\n', e['sign'], type(e['sign']), '\n\n')
        if int(e['sign']) == 1:
            print(si_pos, '\n', Fi)
            unormal = si_pos @ (Fi * product)
        else:
            print('elllllllllllllllllllllllllllllllllllllllllllllllllllllse')
            print(si_neg, '\n', Fi)
            unormal = si_neg @ (Fi * product)
        normal = unormal / unormal.sum()
        print('\n\n\n')
        print(normal)
        E = updateRow(E, 'Mij[fraud]', (E.src == e["src"]) & (E.dst == e["dst"]), normal[0]['fraud'])
        E = updateRow(E, 'Mij[honest]', (E.src == e["src"]) & (E.dst == e["dst"]), normal[0]['honest'])
    return V, E
    
def belief(V, E):
    for v in V.rdd.toLocalIterator():
        if v['type'] == 'user':
            ps = E.filter(E.src == v['id']).select('Mij[fraud]', 'Mij[honest]').toPandas().prod(axis=0) \
                 .rename(index={'Mij[fraud]': 'fraud', 'Mij[honest]': 'honest'}).to_frame()
            Fi = pd.DataFrame({0: [v['Fi[fraud]'], v['Fi[honest]']]}, index=['fraud', 'honest'])
            unormal = Fi * ps
            normal = unormal / unormal.sum()
            V = updateRow(V, 'Fi[fraud]', V.id == v["id"], normal[0]['fraud'])
            V = updateRow(V, 'Fi[honest]', V.id == v["id"], normal[0]['honest'])
            
        else:
            us = E.filter(E.dst == v['id']).select('Mji[bad]', 'Mji[good]').toPandas().prod(axis=0) \
                 .rename(index={'Mji[bad]': 'bad', 'Mji[good]': 'good'}).to_frame()
            Fi = pd.DataFrame({0: [v['Fi[bad]'], v['Fi[good]']]}, index=['bad', 'good'])
            unormal = Fi * us
            normal = unormal / unormal.sum()
            V = updateRow(V, 'Fi[bad]', V.id == v["id"], normal[0]['bad'])
            V = updateRow(V, 'Fi[good]', V.id == v["id"], normal[0]['good'])
    return V, E


def ttt(V, E):
    # V = spark.read.parquet('../data/videoGames_Vertices.parquet')
    # E = spark.read.parquet('../data/videoGames_Edges_Sentiment.parquet')
    m = MRF(V, E)
    m.initBP()
    start_time = time.time()
    m.test()
    # spark.sql('SELECT `Mij[fraud]` from E').show()
    print("\n\n\n--- %s seconds ---\n\n\n" % (time.time() - start_time))

def t1(V, E):
    V, E = userToProduct(V, E)
    V, E = productToUser(V, E)
    E.show()
    V.show()















def u2pMsg(E):
    moreThan1NeighbourIds = E.groupBy(E.src).count().where('count > 1').withColumnRenamed('src', 'id').drop('count')
    moreThan1Neighbour = E.join(moreThan1NeighbourIds, E.src == moreThan1NeighbourIds.id).select(E.columns)
    prods = moreThan1Neighbour.groupBy('src') \
            .agg(exp(sqlsum(log('Mij[fraud]'))).alias('nfraud'), exp(sqlsum(log('Mij[honest]'))).alias('nhonest')) \
            .withColumnRenamed('src', 'id')
    del moreThan1Neighbour, moreThan1NeighbourIds
    
    prods = E.join(prods, E.src == prods.id)
    prods = prods.withColumn('nfraud', col('nfraud')/col('Mij[fraud]')).withColumn('nhonest', col('nhonest')/col('Mij[honest]'))
    unormal = prods.withColumn('Mji[bad]',
                                    when(col('sign') >= 0,
                                        2*col('nfraud')*0.8 + 2*col('nhonest')*0.1) 
                                        .otherwise(2*col('nfraud')*0.2 + 2*col('nhonest')*0.9)) \
                    .withColumn('Mji[good]',
                                when(col('sign') >= 0,
                                        2*col('nfraud')*0.2 + 2*col('nhonest')*0.9) 
                                        .otherwise(2*col('nfraud')*0.8 + 2*col('nhonest')*0.1))
    del prods
    normal = unormal.withColumn('usum', col('Mji[bad]')+col('Mji[good]')) \
                    .withColumn('Mji[bad]', col('Mji[bad]') / col('usum')) \
                    .withColumn('Mji[good]', col('Mji[good]') / col('usum'))
    del unormal
    diffIds = E.select('src', 'dst').subtract(normal.select('src', 'dst')).withColumnRenamed('src', 'srcc').withColumnRenamed('dst', 'dstt')
    diff = E.join(diffIds, (E.src == diffIds.srcc) & (E.dst == diffIds.dstt)).select(E.columns)
    
    return normal.select(E.columns).union(diff)




def p2uMsg(E):
    moreThan1NeighbourIds = E.groupBy(E.dst).count().where('count > 1').withColumnRenamed('dst', 'id').drop('count')
    moreThan1Neighbour = E.join(moreThan1NeighbourIds, E.dst == moreThan1NeighbourIds.id).select(E.columns)
    prods = moreThan1Neighbour.groupBy('dst') \
            .agg(exp(sqlsum(log('Mji[bad]'))).alias('nbad'), exp(sqlsum(log('Mji[good]'))).alias('ngood')) \
            .withColumnRenamed('dst', 'id')
    del moreThan1Neighbour, moreThan1NeighbourIds
    prods = E.join(prods, E.dst == prods.id)
    prods = prods.withColumn('nbad', col('nbad')/col('Mji[bad]')).withColumn('ngood', col('ngood')/col('Mji[good]'))
    unormal = prods.withColumn('Mij[fraud]',
                                    when(col('sign') >= 0,
                                        2*col('nbad')*0.8 + 2*col('ngood')*0.2) 
                                        .otherwise(2*col('nbad')*0.2 + 2*col('ngood')*0.8)) \
                    .withColumn('Mij[honest]',
                                when(col('sign') >= 0,
                                        2*col('nbad')*0.1 + 2*col('ngood')*0.9) 
                                        .otherwise(2*col('nbad')*0.9 + 2*col('ngood')*0.1))
    del prods
    normal = unormal.withColumn('psum', col('Mij[fraud]')+col('Mij[honest]')) \
                    .withColumn('Mij[fraud]', col('Mij[fraud]') / col('psum')) \
                    .withColumn('Mij[honest]', col('Mij[honest]') / col('psum'))
    del unormal
    diffIds = E.select('src', 'dst').subtract(normal.select('src', 'dst')).withColumnRenamed('src', 'srcc').withColumnRenamed('dst', 'dstt')
    diff = E.join(diffIds, (E.src == diffIds.srcc) & (E.dst == diffIds.dstt)).select(E.columns)
    return normal.select(E.columns).union(diff)











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








# blf()

# V = sqlContext.createDataFrame([
#     ('u1', 'user'),
#     ('u2', 'user'),
#     ('u3', 'user'),
#     ('u4', 'user'),
#     ('u5', 'user'),
#     ('u6', 'user'),
#     ('p1', 'product'),
#     ('p2', 'product'),
#     ('p3', 'product'),
#     ('p4', 'product')], ['id', 'type'])

# E = sqlContext.createDataFrame([
#     ('u1', 'p1', '1'),
#     ('u1', 'p3', '-1'),
#     ('u2', 'p1', '1'),
#     ('u2', 'p2', '1'),
#     ('u2', 'p4', '-1'),
#     ('u3', 'p1', '1'),
#     ('u3', 'p2', '1'),
#     ('u3', 'p3', '-1'),
#     ('u4', 'p2', '1'),
#     ('u5', 'p1', '-1'),
#     ('u5', 'p3', '1'),
#     ('u6', 'p2', '1'),
#     ('u6', 'p3', '1'),
#     ('u6', 'p4', '1')], ['src', 'dst', 'sign'])
# V = spark.read.parquet('../data/videoGames_Vertices.parquet')
# E = spark.read.parquet('../data/videoGames_Edges_Sentiment.parquet')
# E = E.select('src', 'dst', col('compound').alias('sign'))
# E = E.withColumn('sign', when(col('sign') == 0, 0.0001).otherwise(col('sign')))
# # E.filter(E.sign == 0.0001).show(20, False)
# # exit()
# # E = E.withColumn('Mij[fraud]', lit(1))
# # E = E.withColumn('Mij[honest]', lit(1))
# # E = E.withColumn('Mji[bad]', lit(1))
# # E = E.withColumn('Mji[good]', lit(1))

# # V = V.withColumn('Fi[fraud]', lit(2))
# # V = V.withColumn('Fi[honest]', lit(2))
# # V = V.withColumn('Fi[bad]', lit(2))
# # V = V.withColumn('Fi[good]', lit(2))
# m = MRF(V, E)
# m.initBP()
# start_time = time.time()
# for i in range(3):
#     m.u2pMsg()
#     m.p2uMsg()
#     spark.catalog.clearCache()
# m.E.write.parquet('../data/MRF-E3.parquet')
# # spark.sql('SELECT `Mij[fraud]` from E').show()
# print("\n\n\n--- %s seconds ---\n\n\n" % (time.time() - start_time))












# V, E = userToProduct(V, E)
# V, E = productToUser(V, E)

# for i in range(5):
#     V, E = userToProduct(V, E)
#     V, E = productToUser(V, E)
# V, E = belief(V, E)


# # V, E = belief(V, E)
# E.show()
# V.show()

# V = updateRow(V, 'Fi[honest]', V.id == 'u3', 0.9)
# p = E.toPanda()
# n = neighbours(E, 'u1')
# print(p)
# n.show()




# from igraph import *
# print(E.collect())
# ig = Graph.TupleList(E.collect(), directed=False)
# ig.vs["label"] = ig.vs["name"]
# ig.es["label"] = ['+', '-', '+', '+', '-', '+', '+', '-', '+', '-', '+', '+', '+', '+']

# ig.vs["color"] = '#00FFFF'
# layout = ig.layout("kk")
# print('\n\n\n\n\n\n\n')
# print(ig.vs["name"])

# plot(ig, layout=layout, margin = 20)
