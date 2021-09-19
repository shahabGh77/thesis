from pyspark.sql.functions import col, when, lit
from pyspark import SparkContext
from pyspark.sql import SQLContext


sc = SparkContext("local", "Graph")
sqlContext = SQLContext(sc)

class mGraph:
    def __init__(self, V, E):
        self.V = V
        self.E = E

    def adjVertices(self, id):
        """returns adjacent vertices of node 'id'

        Args:
            id (str/int): id of desired node

        Returns:
            Dataframe[id]: a Dataframe containing all adjacent vertices of node 'id'
        """
        return self.E.filter((col('src') == id) | (col('dst') == id)).withColumn('id', when(col('src') == id, col('dst')).otherwise(col('src')))

    def _DFSTraverse(self, startingNode, adjDf, colourDf):
        """this function will traverse Graph in DFS order and return True if it sees a cycle

        Args:
            startingNode (str/int): id of a node which we want to start DFS from 
            adjDf (Dataframe[id]): adjacent vertices of startingNode 
            colourDf (Dataframe[id, colour]): nodes and thier colours

        Returns:
            bool, Dataframe[id, colour]: returns (True, colourDf) if it finds a cycle in the graph, otherwise (False, colourDf)
        """
        #0=White(unseen), 1=Gray(seen but still working on its children), 2=Black(node and all of its children are seen)
        colourDf = colourDf.withColumn('colour', when(col('id') == startingNode, 1).otherwise(col('colour')))   #set startingNode colour to Gray
        for v in adjDf.rdd.toLocalIterator():
            colour = colourDf.filter(colourDf.id == v['id']).select('colour').collect()[0][0]   #colour of 'v'
            if colour == 1: #if there is a node that its colour is Gray, it means we've seen it before therefore there is a cycle in the graph
                return True, colourDf
            if colour == 0:
                nAdjDf = self.adjVertices(v['id']).select('id').filter(col('id') != startingNode)    #remove 'v' from its child adjacent vertices
                cycle, colourDf = self._DFSTraverse(v['id'], nAdjDf, colourDf)
                if cycle:
                    return True, colourDf
        colourDf = colourDf.withColumn('colour', when(col('id') == startingNode, 2).otherwise(col('colour')))   #set startingNode colour to Black
        return False, colourDf

    def hasCycle(self):
        """runs _DFSTraverse for all the nodes in the graph

        Returns:
            bool: True if there is a cycle in the graph, otherwise False
        """
        colourDf = self.V.withColumn('colour', lit(0))  #set all nodes colour to white
        for v in self.V.rdd.toLocalIterator():
            if colourDf.filter(colourDf.id == v['id']).select('colour').collect()[0][0] == 0:   #if node 'v' is unseen then run _DFSTraverse on it
                cycle, colourDf = self._DFSTraverse(v['id'], self.adjVertices(v['id']).select('id'), colourDf)
                if cycle:
                    return True
        return False


def example():
    V = sqlContext.createDataFrame([
        ('0', ),
        ('1', ),
        ('2', ),
        ('3', ),
        ('4', )], ['id', ])
        
    E = sqlContext.createDataFrame([
        ('0', '1'),
        ('0', '2'),
        ('0', '3'),
        ('1', '2'),
        ('3', '4')], ['src', 'dst'])

    G = mGraph(V, E)

    if G.hasCycle():
        print('\n\n\nyessssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss\n\n\n')
    else:
        print('\n\n\n noooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo\n\n\n')

    G.adjVertices(1).show()
