from shutil import rmtree
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import DoubleType, IntegerType, MapType, StringType
# from pyspark.sql.functions import col, lit, when, row_number, concat, udf
import pyspark.sql.functions as F
from nltk.sentiment import SentimentIntensityAnalyzer

spark = SparkSession.builder.appName("transform json to desirable parquet").getOrCreate()
sia = SentimentIntensityAnalyzer()

def _createTmp(rPath, mPath):
    df = spark.read.json(rPath)
    df = df.select(F.col('reviewerID').alias('src'), F.col('asin').alias('dst'), 'overall', 'reviewerName',
                   'verified', 'vote', 'reviewTime', 'unixReviewTime', 'summary', 'reviewText')
    df = df.dropDuplicates()
    df = df.repartition('src', 'dst')
    df.write.parquet('tmpE')

    mt = spark.read.json(mPath).dropDuplicates().select('asin', 'title')
    mt = mt.repartition('asin')
    mt.write.parquet('tmpP')
    del df, mt

def _createVertices(mPath, oPath):
    df = spark.read.parquet('tmpE')
    users = df.groupBy('src').count().select(F.col('src').alias('id'), F.col('count').alias('deg')).withColumn('type', F.lit('u')).cache()
    products = df.groupBy('dst').count().select(F.col('dst').alias('id'), F.col('count').alias('deg')).withColumn('type', F.lit('p')).cache()

    users_names = df.select('src', 'reviewerName')
    unique_users_names = users_names.dropDuplicates(['src'])
    dup_users_names = users_names.exceptAll(unique_users_names).select('src')

    users = users.join(unique_users_names, F.col('id') == F.col('src')).drop('src')
    users = users.join(dup_users_names, F.col('id') == F.col('src'), 'left_outer') \
                 .withColumn('severalName', F.when(F.col('src').isNull(), False).otherwise(True)) \
                 .drop('src').dropDuplicates().withColumnRenamed('reviewerName', 'name')

    products_names = spark.read.parquet('tmpP')
    unique_products_names = products_names.dropDuplicates(['asin'])
    dup_products_names = products_names.exceptAll(unique_products_names).select('asin')

    products = products.join(unique_products_names, F.col('id') == F.col('asin'), 'left_outer') \
                                    .withColumn('title', F.when(F.col('title').isNull(), F.lit('')).otherwise(F.col('title'))) \
                                    .select('id', 'deg', 'type', F.col('title').alias('name'))
    products = products.join(dup_products_names, F.col('id') == F.col('asin'), 'left_outer') \
                       .withColumn('severalName', F.when(F.col('asin').isNull(), False).otherwise(True)) \
                       .drop('asin').dropDuplicates()

    vertices = users.union(products)
    vertices.write.parquet(oPath + '_Vertices.parquet')
    del df, users, users_names, unique_users_names, dup_users_names
    del products, products_names, unique_products_names, dup_products_names


def _createEdges(oPath, edgeIDLabel):
    df = spark.read.parquet('tmpE')
    df = df.drop('reviewerName')

    w = Window.orderBy(df.unixReviewTime)
    df = df.withColumn('edgeID', F.row_number().over(w)).withColumn('edgeID', F.concat(F.lit(edgeIDLabel), F.col('edgeID')))
    df.select('edgeID', 'src', 'dst', 'overall', 'verified', 'vote', 'reviewTime', 'unixReviewTime', 'summary', 'reviewText') \
      .repartition('edgeID').write.parquet(oPath + '_Edges.parquet')
    del df, w


def _voteCastToInt(vote):
    if vote :
        x = vote.replace(',', '')
        res = int(x)
    else:
        res = 0
    return res

def _ratingMap(x):
    return {
        1: -1.0,
        2: -0.8,
        3: 0.3,
        4: 0.8,
        5: 1.0
    }[x]

def _sentimentExtraction(text):
    snt = sia.polarity_scores(text) if text else sia.polarity_scores('')
    snt['posOrNeg'] = snt['pos'] if snt['pos'] >= snt['neg'] else -snt['neg']
    return snt

def _sentimentScore(rs, ts, rw, tw, vn, am, hp):
    s = (rs*rw + ts*tw)/(rw+tw) 
    if s < 0:
        f = s - vn*am
        s = f if f > -1 else -1.0   #you must use -1.0 !!! not -1 otherwise it returns null 
    else:   #not elif becasue mabe vn could make s not to be 0
        f = s + vn*am
        s = f if f < 1 else 1.0     #you must use 1.0 !!! not 1 otherwise it returns null

        if s == 0: #avoid s=0
            s = rs/(rw+tw) if hp == 'rating' else ts/(rw+tw) 
    return s

def _sentiment(rating, text, vote, rw, tw, am, hp):
    rs = _ratingMap(int(rating)) #rating_score
    vn = _voteCastToInt(vote) if vote else 0
    snt = _sentimentExtraction(text)    #sentiment extraction from text
    ts = snt['compound']    #text_score
    s = _sentimentScore(rs, ts, rw, tw, vn, am, hp)
    snt.update({
        'nvote': vn,
        's': s
    })
    return list(snt.values())

def _sentimentMap(x, rw, tw, am, hp):
    current = list(x)
    res = _sentiment(x['overall'], x['reviewText'], x['vote'], rw, tw, am, hp)
    current.extend(res)
    return current

def addSentiment(ePath, rw=1, tw=1, am=0.01, hp='rating'):
    df = spark.read.parquet(ePath)
    cols = df.columns
    cols.extend(['neg', 'neu', 'pos', 'compound', 'posOrNeg', 'nvote', 's'])
    nDf = df.rdd.map(lambda x: _sentimentMap(x, rw, tw, am, hp)).toDF(cols)
    nDf = nDf.withColumn('vote', F.col('nvote')).drop('nvote')

    d = ePath.partition('.parquet')
    oPath = d[0] + '_Sentiment' + d[1]

    nDf.write.parquet(oPath)
    


def transform(rPath, mPath, oPath, edgeIDLabel):
    _createTmp(rPath, mPath)
    _createVertices(mPath, oPath)
    _createEdges(oPath, edgeIDLabel)
    rmtree('tmpE')
    rmtree('tmpP')


def edgePlus(ePath, vPath):
    v = spark.read.parquet(vPath)
    e = spark.read.parquet(ePath)

    cols = e.columns
    cols.extend(['scu', 'udeg'])
    e = e.join(v.filter(v.type == 'u'), F.col('src') == F.col('id')) \
          .withColumn('scu', F.when(F.col('deg') == 1, True).otherwise(False)) \
          .withColumn('udeg', F.col('deg')).select(cols)
    cols.extend(['scp', 'pdeg'])
    e = e.join(v.filter(v.type == 'p'), F.col('dst') == F.col('id')) \
            .withColumn('scp', F.when(F.col('deg') == 1, True).otherwise(False)) \
            .withColumn('pdeg', F.col('deg')).select(cols)

    d = ePath.partition('_Sentiment')
    oPath = d[0] + 'Plus' + d[2]
    e.write.parquet(oPath)
    
def creatLBP(ePath, vPath):
    v = spark.read.parquet(vPath).select('id', 'type', 'deg')
    e = spark.read.parquet(ePath).select('edgeID', 'src', 'dst', 'scu', 'udeg', 'scp', 'pdeg', 's')

    #here you can add prior knowledge
    v = v.select('*',
                 F.lit(2).alias('Fi[fraud]'), F.lit(2).alias('Fi[honest]'),
                 F.lit(2).alias('Fi[bad]'), F.lit(2).alias('Fi[good]')
    )
    e = e.select('*',
                 F.lit(2).alias('Fi[fraud]'), F.lit(2).alias('Fi[honest]'),
                 F.lit(2).alias('Fi[bad]'), F.lit(2).alias('Fi[good]')
    )
    #initialise messages
    e = e.select('*',
                 F.lit(1).alias('Mij[fraud]'), F.lit(1).alias('Mij[honest]'),
                 F.lit(1).alias('Mji[bad]'), F.lit(1).alias('Mji[good]')
    )

    #repartitioning makes LBP runs faster
    e = e.repartition('scu', 'scp', 'uDeg', 'pDeg')

    dv = vPath.partition('.parquet')
    oVPath = dv[0] + 'LBP' + dv[1]
    de = ePath.partition('Plus')
    oPPath = de[0] + 'LBP' + de[2]
    v.write.parquet(oVPath)
    e.write.parquet(oPPath)

def mergeVertices(pathL, oPath):
    for i, p in enumerate(pathL):
        print(f'################### in = {p[0]} ###########################')
        
        df = spark.read.parquet(p[0]) \
                       .select('id', 'type', 'deg', 'Fi[fraud]', 'Fi[honest]', 'Fi[bad]', 'Fi[good]') 
        print(f'################### df = {df.count()} ###########################')
        if i == 0:
            res = df.withColumn('category', F.array(F.lit(p[1])))
            continue
        intersection = res.drop('category').intersect(df)
                          
        print(f'################### res = {res.count()} ###########################')
        print(f'################### intersection = {intersection.count()} ###########################')
        pure = df.exceptAll(intersection).withColumn('category', F.array(F.lit(p[1])))
        print(f'################### pure = {pure.count()} ###########################')
        if intersection.count() != 0:
            intersection = intersection.select('id', 'deg')
            newDeg = res.select(res.id, res.deg).join(intersection, res.id == intersection.id) \
                        .dropDuplicates().select(res.id.alias('idd'), (intersection.deg + res.deg).alias('degg'))
            print(f'################### newDeg = {newDeg.count()} ###########################')
            res = res.join(newDeg, res.id == newDeg.idd, 'left_outer') \
                    .withColumn('deg', F.when(F.col('idd').isNull(), F.col('deg')).otherwise(newDeg.degg)) \
                    .withColumn('category', F.when(F.col('idd').isNull(), F.col('category'))
                                            .otherwise(F.array_union(F.col('category'), F.array(F.lit(p[1]))))) \
                    .drop('idd', 'degg')
            print(f'################### res = {res.count()} ###########################')
        res = res.union(pure)
        print(f'################### res = {res.count()} ###########################')
    
    res.write.parquet(oPath)


def mergeCategories(vPathL, ePathL):
    for i, eP in enumerate(ePathL):
        e = spark.read.parquet(eP)
        if i == 0:
            edges = e
            continue
        edges = edges.unionByName(e)
    users = edges.select('src', 'dst').groupBy('src').count() \
             .select(F.col('src').alias('id'), F.col('count').alias('deg')).withColumn('type', F.lit('u'))
    products = edges.select('src', 'dst').groupBy('dst').count() \
                .select(F.col('dst').alias('id'), F.col('count').alias('deg')).withColumn('type', F.lit('p'))
    vertices = users.union(products)
    catL = []
    for i, vP in enumerate(vPathL):
        v = spark.read.parquet(vP[0]).withColumn(vP[1], F.lit(1))
        if i == 0:
            vertices = vertices.join(v, vertices.id == v.id, 'left_outer') \
                 .select(vertices.id, vertices.type, vertices.deg, 
                         v['Fi[fraud]'], v['Fi[honest]'], v['Fi[bad]'], v['Fi[good]'], 
                         F.when(F.col(vP[1]).isNull(), 0).otherwise(F.col(vP[1])).alias(vP[1]))
            catL.append(vP[1])
        else:
            vertices = vertices.join(v, vertices.id == v.id, 'left_outer') \
                 .select(vertices.id, vertices.type, vertices.deg,
                         F.when(vertices['Fi[fraud]'].isNull(), v['Fi[fraud]']).otherwise(vertices['Fi[fraud]']).alias('Fi[fraud]'),
                         F.when(vertices['Fi[honest]'].isNull(), v['Fi[honest]']).otherwise(vertices['Fi[honest]']).alias('Fi[honest]'),
                         F.when(vertices['Fi[bad]'].isNull(), v['Fi[bad]']).otherwise(vertices['Fi[bad]']).alias('Fi[bad]'),
                         F.when(vertices['Fi[good]'].isNull(), v['Fi[good]']).otherwise(vertices['Fi[good]']).alias('Fi[good]'),
                         *catL,
                         F.when(F.col(vP[1]).isNull(), 0).otherwise(F.col(vP[1])).alias(vP[1])
                        #  F.when(vertices['category'].isNull(), v['cat'])
                        #   .otherwise(F.array_union(vertices['category'], v['cat'])).alias('category')
                         )
            catL.append(vP[1])
    edges = edges.join(vertices.select('id', 'deg'), edges.src == F.col('id'))\
                 .withColumn('udeg', F.col('deg')) \
                 .withColumn('scu', F.when(F.col('deg') == 1, True).otherwise(False)) \
                 .drop('id', 'deg')

    vertices.write.parquet('data/Vf')
    edges.write.parquet('data/Ef')
    vertices.repartition('type', 'deg').write.parquet('data/Vf_repartition')
    edges.repartition('scu', 'scp', 'uDeg', 'pDeg').write.parquet('data/Ef_repartition')
    

def mergeRun():
    vPathL = [('data/cellPhonesAndAccessories_VerticesLBP.parquet', 'C'), ('data/electronics_VerticesLBP.parquet', 'E'),
             ('data/toysAndGames_VerticesLBP.parquet', 'T'), ('data/videoGames_VerticesLBP.parquet', 'V')]
    ePathL = ['data/cellPhonesAndAccessories_EdgesLBP.parquet', 'data/electronics_EdgesLBP.parquet', 
              'data/toysAndGames_EdgesLBP.parquet', 'data/videoGames_EdgesLBP.parquet']
    return mergeCategories(vPathL, ePathL)


def run(rPath, mPath, oPath, edgeIDLabel):
    transform(rPath, mPath, oPath, edgeIDLabel)
    addSentiment(oPath+'_Edges.parquet')
    edgePlus(oPath + '_Edges_Sentiment.parquet', oPath + '_Vertices.parquet')
    creatLBP(oPath + '_EdgesPlus.parquet', oPath + '_Vertices.parquet')

