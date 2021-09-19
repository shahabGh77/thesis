from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

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


def run():
    E = spark.read.parquet('../data/videoGames_Edges.parquet')

    cols = E.columns
    cols.extend(['neg', 'neu', 'pos', 'compound', 'posOrNeg'])
    nE = E.rdd.map(lambda x: addSentiment(x)).toDF(cols)
    nE.write.parquet('../data/videoGames_Edges_Sentiment.parquet')