from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType

spark = SparkSession.builder.appName("sentiment extraction").getOrCreate()
sia = SentimentIntensityAnalyzer()

def fromOverall(overall):
    score = int(overall)
    if score > 3:
        score = score * 0.02
        return {'neg': 0.0, 'neu': 0.0, 'pos': score, 'compound': score}
    else:
        score = (5-score) * 0.02 
        return {'neg': score, 'neu': 0.0, 'pos': 0.0, 'compound': -score}


def addSentiment(x):
    """adds sentiment based on 'nltk.sentiment' module to raw 'x' in a dataframe

    Args:
        x (Row('reviewText', ...)): each row should have a 'reviewText' field

    Returns:
        list: return the same x with its sentiment attached
    """
    snt = sia.polarity_scores(x['reviewText']) if x['reviewText'] else sia.polarity_scores('')
    # component = 0 fix 
    if snt['compound'] == 0.0:
        if  snt['neu'] == 1.0 or ( snt['pos'] == 0.0 and snt['neg'] == 0.0):
            snt = fromOverall(x['overall'])
        else:
            snt['compound'] = snt['pos'] if snt['pos'] >= snt['neg'] else -snt['neg']
    posOrNeg = snt['pos'] if snt['pos'] >= snt['neg'] else -snt['neg']
    res = list(x)
    res.extend(list(snt.values()))
    res.append(posOrNeg)
    return res


@udf(IntegerType())
def voteCast(x):
    if x :
        x = x.replace(',', '')
        res = int(x)
    else:
        res = 0
    return res

def ratingMap(x):
    return {
        1: -1,
        2: -0.8,
        3: 0.3,
        4: 0.8,
        5: 1
    }[x]

@udf(returnType=DoubleType())
def sentimentScore(rating, text, vote, rw=1, tw=1, am=0.01, hp='rating'):
    rs = ratingMap(int(rating)) #rating_score
    vn = int(vote) if vote else 0
    snt = sia.polarity_scores(text) if text else sia.polarity_scores('')    #sentiment extraction from text
    ts = snt['compound']    #text_score
    
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

def run():
    E = spark.read.parquet('../data/videoGames_Edges.parquet')

    cols = E.columns
    cols.extend(['neg', 'neu', 'pos', 'compound', 'posOrNeg'])
    nE = E.rdd.map(lambda x: addSentiment(x)).toDF(cols)
    nE.write.parquet('../data/videoGames_Edges_Sentiment.parquet')