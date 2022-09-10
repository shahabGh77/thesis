#Dempster Shafer Theory

from shutil import rmtree
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, MapType, StringType
import pyspark.sql.functions as F


spark = SparkSession.builder.appName("extract belief for reviews").getOrCreate()


@F.udf(returnType=MapType(StringType(), DoubleType()))
def fusion(udeg, pdeg, fraudster, honest, bad, good, s):
    mf1, mg1 = fraudster, honest
    mf_g1, mf_g2 = 0, 0
    if s > 0:
        mf2, mg2 = bad, good
    else:
        mf2, mg2 = good, bad

    if udeg <= 5:
        frag = udeg/5
        shift = (mf1 - mf1*frag) + (mg1 - mg1*frag)
        #shift = 1-frag if shift > 1-frag else shift
        mf1, mg1, mf_g1 =  mf1*frag, mg1*frag, mf_g1+shift

    mf2, mg2, mf_g2 =  mf2*.7, mg2*.7, .3
    if pdeg <= 30:
        frag = pdeg/30
        shift = (mf2 - mf2*frag) + (mg2 - mg2*frag)
        #shift = 1-frag if shift > 1-frag else shift
        mf2, mg2, mf_g2 =  mf2*frag, mg2*frag, mf_g2+shift

    k = mf1*mg2 + mg1*mf2
    if k >= 1:
        print('#########################################')
        print(udeg, pdeg, fraudster, honest, bad, good, s)
        print(mf1 ,mg1, mf_g1, mf2, mg2, mf_g2)
        exit(1)
    fake = (mf1*mf2 + mf1*mf_g2 + mf_g1*mf2) / (1-k)
    genuine = (mg1*mg2 + mg1*mf_g2 + mf_g1*mg2) / (1-k)
    fakeOrGenuine = (mf_g1*mf_g2) / (1-k)

    return {'fake': round(fake, 2), 'genuine': round(genuine, 2), 'uncertainty': round(fakeOrGenuine, 2)} 
     


def addEdgeBelief(vPath, ePath):
    v = spark.read.parquet(vPath).select('id', 'P[fraud]', 'P[honest]', 'P[bad]', 'P[good]')
    # e = spark.read.parquet(ePath).select('edgeID', 'src', 'dst', 'udeg', 'pdeg', 's', 'Mij[fraud]', 'Mij[honest]', 'Mji[bad]', 'Mji[good]')
    e = spark.read.parquet(ePath).select('src', 'dst', 'udeg', 'pdeg', 's', 'Mij[fraud]', 'Mij[honest]', 'Mji[bad]', 'Mji[good]', 'label')

    e = e.join(v, e.src == v.id).select(*e.columns, v['P[fraud]'].alias('fraudster'), v['P[honest]'].alias('honest'))
    e = e.join(v, e.dst == v.id).select(*e.columns, v['P[bad]'].alias('bad'), v['P[good]'].alias('good'))
    # e = e.select('edgeID', 'src', 'dst', 'udeg', 'pdeg',
    e = e.select('src', 'dst', 'udeg', 'pdeg', 'label',

                 *[F.round(c, 4).alias(c) 
                   for c in
                   ['s', 'Mij[fraud]', 'Mij[honest]', 'Mji[bad]', 'Mji[good]', 'fraudster', 'honest', 'bad', 'good']
                   ]
                ).withColumn('fraudster', F.when(F.col('fraudster').isNull(), F.col('Mij[fraud]')).otherwise(F.col('fraudster'))) \
                .withColumn('honest', F.when(F.col('honest').isNull(), F.col('Mij[honest]')).otherwise(F.col('honest'))) \
                .withColumn('bad', F.when(F.col('bad').isNull(), F.col('Mji[bad]')).otherwise(F.col('bad'))) \
                .withColumn('good', F.when(F.col('good').isNull(), F.col('Mji[good]')).otherwise(F.col('good')))
    # fraudster + honest and bad + good should be 1. normalise singleton initiall value for messages. [1, 1] => [0.5, 0.5].
    e = e.withColumn('fraudster', F.when((F.col('udeg') == 1)&(F.col('pdeg') == 1), F.col('fraudster')/2).otherwise(F.col('fraudster'))) \
        .withColumn('honest', F.when((F.col('udeg') == 1)&(F.col('pdeg') == 1), F.col('honest')/2).otherwise(F.col('honest'))) \
        .withColumn('bad', F.when((F.col('udeg') == 1)&(F.col('pdeg') == 1), F.col('bad')/2).otherwise(F.col('bad'))) \
        .withColumn('good', F.when((F.col('udeg') == 1)&(F.col('pdeg') == 1), F.col('good')/2).otherwise(F.col('good')))
    # return e
    ds = e.withColumn('label1', fusion(e.udeg, e.pdeg, e.fraudster, e.honest, e.bad, e.good, e.s))
    return ds


def _reviewDist(dstPath, baseCol, labelCol='label1'):
    a = spark.read.parquet(dstPath)
    res = {}
    for i in range(10, 0, -1):
        ub, lb = i/10, i/10 - 0.1
        if i == 10:
            key = f'{lb} ≤ {baseCol} ≤ {ub}'
            df = a.filter((F.col(baseCol) <= ub) & (F.col(baseCol) >= lb))
        else: 
            key = f'{lb} ≤ {baseCol} < {ub}'
            df = a.filter((F.col(baseCol) < ub) & (F.col(baseCol) >= lb))
        
        intervalCount = df.count()
        uncertainCount = df.filter(F.col(labelCol).uncertainty > F.col(labelCol).fake + F.col(labelCol).genuine).count()
        cdf = df.filter(F.col(labelCol).uncertainty <= F.col(labelCol).fake + F.col(labelCol).genuine)
        fakeCount = cdf.filter(F.col(labelCol).fake > F.col(labelCol).genuine).count()
        genuineCount = cdf.count() - fakeCount
        
        res[key] = [round(fakeCount/intervalCount, 2)*100, round(uncertainCount/intervalCount, 2)*100, round(genuineCount/intervalCount, 2)*100]
    return res