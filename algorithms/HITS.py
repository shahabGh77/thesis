from pyspark.sql.functions import col, lit, when


def hubsNormalisation(V):
    """This function will normalaise hubs scores, deviding every h_score value by max(h_score)

    Args:
        V (Dataframe[id, h_score]): a dataframe containing every nodes with their h_score

    Returns:
        Dataframe[id, h_score]: normalised hubs scores( 0 <= h_score <= 1 )
    """
    max_h_score = V.agg({'h_score': 'max'}).collect()[0][0]
    return V.withColumn('h_score', V.h_score/max_h_score)


def authsNormalisation(V):
    """This function will normalaise authorities scores, deviding every a_score value by max(a_score)

    Args:
        V (Dataframe[id, a_score]): a dataframe containing every nodes with their a_score

    Returns:
        Dataframe[id, a_score]: normalised authorities scores( 0 <= a_score <= 1 )
    """
    max_a_score = V.agg({'a_score': 'max'}).collect()[0][0]
    return V.withColumn('a_score', V.a_score/max_a_score)


def hubsUpdate(V, E):
    """This function will calculate new values for hubs scores based on current authorities scores

    Args:
        V (Dataframe[id, a_score]): nodes with thier current authorities scores
        E (Dataframe[src, dst]): a dataframe with src, dst columns which represents graph edges
    Logic:

    +---+---+                                +---+-------+---+---+
    |src|dst|                                | id|a_score|src|dst|
    +---+---+                                +---+-------+---+---+
    |  A|  B|    +---+-------+               |  A|    0.5|  B|  A|                             +---+-------+                            +---+-------+
    |  A|  C|    | id|a_score|               |  B|      1|  A|  B|                             | id|h_score|                            | id|h_score|
    |  A|  D|    +---+-------+               |  B|      1|  D|  B|                             +---+-------+                            +---+-------+
    |  B|  A|    |  A|    0.5|               |  C|      1|  A|  C|                             |  A|      3|                            |  A|      1|
    |  B|  D|    |  B|      1|   join on     |  C|      1|  D|  C|   groupBy src               |  B|    1.5|    normalaise              |  B|    0.5|
    |  C|  E|    |  C|      1|   id = dst    |  D|      1|  A|  D|   h_score = sum(a_score)    |  C|    0.5|    h_score/max(h_score)    |  C| 0.1666|
    |  D|  B|    |  D|      1|   ========>   |  D|      1|  B|  D|   ======================>   |  D|      2|    ====================>   |  D| 0.6666|
    |  D|  C|    |  E|    0.5|               |  E|    0.5|  C|  E|                             |   |       |    0 <= h_score <= 1       |   |       |
    +---+---+    +---+-------+               +---+-------+---+---+                             +---+-------+                            +---+-------+  

    Returns:
        Dataframe: new values for hubs scores
    """
    tmp = V.join(E, V.id == E.dst).groupBy(E.src).sum('a_score').withColumnRenamed("sum(a_score)", "h_score").withColumnRenamed("src", "id")
    return hubsNormalisation(tmp)

def authsUpdate(V, E):
    """This function will calculate new values for authorities scores based on current hubs scores
    Args:
        V (Dataframe[id, h_score]): nodes with thier current hubs scores
        E (Dataframe[src, dst]): a dataframe with src, dst columns which represents graph edges
    Logic:

    +---+---+                                +---+-------+---+---+
    |src|dst|                                | id|h_score|src|dst|
    +---+---+                                +---+-------+---+---+
    |  A|  B|    +---+-------+               |  B|      1|  B|  A|                             +---+-------+                            +---+-------+
    |  A|  C|    | id|h_score|               |  B|      1|  B|  D|                             | id|a_score|                            | id|a_score|
    |  A|  D|    +---+-------+               |  D|      1|  D|  B|                             +---+-------+                            +---+-------+
    |  B|  A|    |  A|      1|   join on     |  D|      1|  D|  C|   groupBy dst               |  A|      1|    normalaise              |  A|    0.5|
    |  B|  D|    |  B|      1|   id = srs    |  C|      1|  C|  E|   a_score = sum(h_score)    |  B|      2|    a_score/max(a_score)    |  B|      1|
    |  C|  E|    |  C|      1|   ========>   |  A|      1|  A|  B|   ======================>   |  C|      2|    ====================>   |  C|      1|
    |  D|  B|    |  D|      1|               |  A|      1|  A|  C|                             |  D|      2|    0 <= a_score <= 1       |  D|      1|
    |  D|  C|    |  E|      1|               |  A|      1|  A|  D|                             |  E|      1|                            |  E|    0.5|
    +---+---+    +---+-------+               +---+-------+---+---+                             +---+-------+                            +---+-------+  

    Returns:
        Dataframe[id, a_score]: new values for authorities scores
    """    
    tmp = V.join(E, V.id == E.src).groupBy(E.dst).sum('h_score').withColumnRenamed("sum(h_score)", "a_score").withColumnRenamed("dst", "id")
    return authsNormalisation(tmp)


def completeHAscore(V, hubs, auths):
    """Add nodes which aren't in the results with score = 0(a_score or h_score) 

    Args:
        V (Dataframe[id, ...]): all nodes of graph
        hubs (Dataframe[id, h_score]): hubs scores
        auths (Dataframe[id, a_score]): authorities scores

    Returns:
        Dataframe[id, h_score], Dataframe[id, a_score]: complete hubs and authorities scores
    """
    hDiff = V.select('id').subtract(hubs.select('id')).withColumn('h_score', lit(0))
    aDiff = V.select('id').subtract(auths.select('id')).withColumn('a_score', lit(0))

    return hubs.union(hDiff), auths.union(aDiff)


def HITS(V, E, iteration, hubs=None, auths=None):
    """hits algorithm

    Args:
        V (Dataframe[id]): vertices
        E (Dataframe[src, dst]): edges
        iteration (Int): number of iteration
        hubs (Dataframe[id, h_score], optional): initial value for hubs scores. Defaults to None.
        auths (Dataframe[id, a_score], optional): initial value for authorities scores. Defaults to None.
        
    Returns:
        Dataframe[id, h_score], Dataframe[id, a_score]: hubs and authorities scores
    """
    if not hubs: hubs  = V.withColumn('h_score', lit(1))
    if not auths: auths = V.withColumn('a_score', lit(0))

    for i in range(iteration):
        auths = authsUpdate(hubs, E)
        hubs  = hubsUpdate(auths, E)

    return completeHAscore(V, hubs, auths)


def HITS_saveMode(V, E, iteration, outputFilePath, hubs=None, auths=None,):
    """hits algorithm

    Args:
        V (Dataframe[id]): vertices
        E (Dataframe[src, dst]): edges
        iteration (Int): number of iteration
        hubs (Dataframe[id, h_score], optional): initial value for hubs scores. Defaults to None.
        auths (Dataframe[id, a_score], optional): initial value for authorities scores. Defaults to None.
        outputFilePath (Str, optional): output file path with prefix name(example: 'output/videoGames'). Defaults to None.

    Returns:
        Dataframe[id, h_score], Dataframe[id, a_score]: hubs and authorities scores
    """
    if not hubs: hubs  = V.withColumn('h_score', lit(1))
    if not auths: auths = V.withColumn('a_score', lit(0))

    for i in range(iteration):
        auths = authsUpdate(hubs, E)
        hubs  = hubsUpdate(auths, E)
        hubs, auths = completeHAscore(V, hubs, auths)
        
        print(f'\n\n###################### step {i+1} ########################')
        hubs.write.parquet(outputFilePath + f'_hubs{i+1}.parquet')
        auths.write.parquet(outputFilePath + f'_auths{i+1}.parquet')