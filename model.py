#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import math
import numpy as np
from collections import Counter

class KnnBalltreeClassifier(object):
    '''Классификатор реализует взвешенное голосование по ближайшим соседям. 
    При подсчете расcтояния используется l2-метрика.
    Поиск ближайшего соседа осуществляется поиском по ball-дереву.
    Параметры
    ----------
    n_neighbors : int, optional
        Число ближайших соседей, учитывающихся в голосовании
    weights : str, optional (default = 'uniform')
        веса, используемые в голосовании. Возможные значения:
        - 'uniform' : все веса равны.
        - 'distance' : веса обратно пропорциональны расстоянию д о классифицируемого объекта
        -  функция, которая получает на вход массив расстояний и возвращает массив весов
    leaf_size: int, optional
        Максимально допустимый размер листа дерева
    '''
    class Node(object):        
        def __init__(self, pivot = None, points = [], radius = None): #Индекс корня - 1
            self.pivot = pivot #Индекс левой верщины - 2*v
            self.radius = radius #Индекс правой вершины - 2*v + 1
            self.points = points
        def __str__(self):
            self.stringList = ''.join(str(symbol) for symbol in self.points)
            return("pivot = " + str(self.pivot) + " radius = " + str(self.radius) + " points = " 
                + str(self.stringList))
        
    class Point(object):
            def __init__(self, point, index):
                self.point = point
                self.index = index
            def __str__(self):
                return("Vector " + str(self.point) + " with index " + str(self.index))

    def __init__(self, n_neighbors=1, weights='uniform', leaf_size=30):
        if(n_neighbors <= 0):
            print("Ошибка! Соседей не может быть меньше, чем 1")
            return
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.nodes = {}
        self.kNSet = set()
        self.points = []
        self.classes = []
        self.pointsOriginal = []

    def GetnpArrayOfPoints(self, xStruct):
        x = []
        for i in range(len(xStruct)):
            x.append(xStruct[i].point)
        return(np.array(x))
    
    def Distance(self, a, b):
        if(len(a) != len(b)):
            print("Размерности объектов не одинаковые!")
            return("Error")
        return np.linalg.norm(a-b)

    def GetMaxDistance(self, point, points):
        points = self.GetnpArrayOfPoints(points)
        point = point.point
        #print(points)

        max = 0
        for i in range(len(points)):
            currentDist = self.Distance(point, points[i])
            if(currentDist > max):
                max = currentDist
        return(max)

    def GetMaxSpreadDimension(self, x): #Нужно передать подотрезок array[a:b]
        x = self.GetnpArrayOfPoints(x)       
        difference = []
        for i in range(x.shape[1]):
            maximum = x[0][i]
            minimum = x[0][i]
            for j in range(x.shape[0]):
                maximum = max(maximum, x[j][i])
                minimum = min(minimum, x[j][i])
            difference.append((maximum - minimum, i))
        return(max(difference)[1])

    def SortingByDimension(self, x, startPoint, endPoint, dimension): #Нужно передать начало и 
        left = x[:startPoint]                                  
        rigth = x[endPoint:]
        x = np.array(sorted(x[startPoint:endPoint], key = lambda Point: Point.point[dimension]))
        x = np.concatenate((left, x, rigth), 0)
        return(x)

    def GetCentroidIndex(self, x, dimension): #Нужно передать подотрезок array[a:b]
        x = self.GetnpArrayOfPoints(x)
        if(x.shape[0] == 0):
            print("Ошибка! Передан нулевой массив в поиск центройда")
        sum = 0
        for i in range(x.shape[0]): #Размер x не нулевой
            sum = sum + x[i][dimension]
        sum = sum / x.shape[0]
        minimum = abs(sum - x[0][dimension])
        index = 0
        for i in range(x.shape[0]):
            if(abs(x[i][dimension] - sum) < minimum):
                minimum = abs(x[i][dimension] - sum)
                index = i
        return(index)

    def ConstructTree(self, leafSize, vertexIndex, startPoint, endPoint):
        if(endPoint - startPoint <= 0): 
            return #endPoint не включается в отрезок [ )
        
        #print("ConstructTree" + str(leafSize) + " " + str(startPoint) + " " + str(endPoint))

        dimension = self.GetMaxSpreadDimension(self.points[startPoint:endPoint])
        self.points = self.SortingByDimension(self.points, startPoint, endPoint, dimension)
        centroidIndex = self.GetCentroidIndex(self.points[startPoint:endPoint], dimension) + startPoint 
        # Передаём подотрезок, индексация внутри функции будет с нуля, поэтому + startPoint делаем
        #if(startPoint > centroidIndex or centroidIndex >= endPoint):
        #   print("ошибка")
        #   return

        #print(centroidIndex)

        radius = self.GetMaxDistance(self.points[centroidIndex], self.points[startPoint:endPoint])
        self.nodes[vertexIndex] = self.Node(self.points[centroidIndex], [], radius)
        if(endPoint - startPoint <= leafSize):
            for i in range(startPoint, endPoint):
                self.nodes[vertexIndex].points.append(self.points[i])
            return
        else:
            self.nodes[vertexIndex].points.append(self.points[centroidIndex])
        self.ConstructTree(leafSize, vertexIndex * 2, startPoint, centroidIndex)
        self.ConstructTree(leafSize, vertexIndex * 2 + 1, centroidIndex + 1, endPoint)

    def searchBallSubtree(self, vertexIndex, newPoint, kN):
        if(len(self.nodes[vertexIndex].points) > 1):
            #last = 0
            for point in self.nodes[vertexIndex].points:
                distance = self.Distance(newPoint, point.point)
                if(len(self.kNSet) >= kN and max(self.kNSet)[0] > distance):
                    self.kNSet.remove(max(self.kNSet))
                    #last = point
                if(len(self.kNSet) < kN):
                    self.kNSet.add((distance, point.index))
                    #last = point
            #print(last)
            return
        leftChild = vertexIndex * 2
        rightChild = vertexIndex * 2 + 1
        distance = self.Distance(newPoint, self.nodes[vertexIndex].pivot.point)

        if(len(self.kNSet) >= kN and max(self.kNSet)[0] > distance):
            self.kNSet.remove(max(self.kNSet))
        if(len(self.kNSet) < kN):
            self.kNSet.add((distance, self.nodes[vertexIndex].pivot.index))
        
        if(len(self.nodes[leftChild].points) != 0):
            distance = self.Distance(newPoint, self.nodes[leftChild].pivot.point)
            if(len(self.kNSet) < kN or max(self.kNSet)[0] > distance - self.nodes[leftChild].radius):
                self.searchBallSubtree(leftChild, newPoint, kN)

        if(len(self.nodes[rightChild].points) != 0):
            distance = self.Distance(newPoint, self.nodes[rightChild].pivot.point)
            if(len(self.kNSet) < kN or max(self.kNSet)[0] > distance - self.nodes[rightChild].radius):
                self.searchBallSubtree(rightChild, newPoint, kN)
        
     
    def fit(self, x, y):
        if (len(x) != len(y)):
            print("Ошибка - не у всех точек определены классы")
            return("fit Error")
        self.points = []
        self.classes = []
        self.pointsOriginal = x
        for i in range(len(x)):
            self.points.append(self.Point(x[i], i))
        self.classes = y

        self.ConstructTree(self.leaf_size, 1, 0, len(self.points))
        return self
    
    def predict(self, x):
        predictions = []
        for i in range(len(x)):
            self.kNSet.clear()
            self.searchBallSubtree(1, x[i], self.n_neighbors)
            results = []
            for index in self.kNSet: #kNSet хранит пару (дистанция, индекс)
                results.append(self.classes[index[1]])
                #print(index[1])
                #print(classes[index[1]])
            resultClass = Counter(results).most_common()[0][0]
            predictions.append(resultClass)
        return np.array(predictions)
        
    def predict_proba(self, X):    
        classesCount = np.unique(self.classes)
        results = []
        for point in X:
            neighborsClasses = self.get_kneighbors_classes(point)
            result = []
            for clas in classesCount:
                count = 0
                for neighborClass in neighborsClasses:
                    if(clas == neighborClass):
                        count += 1
                result.append(count/self.n_neighbors * 100)
            results.append(result)
        return(np.array(results))



    def get_kneighbors_classes(self, x):
        self.kNSet.clear()
        self.searchBallSubtree(1, x, self.n_neighbors)
        result = []
        for index in self.kNSet:
            result.append(self.classes[index[1]]) 
        return np.array(result)
        
    def kneighbors(self, x, n_neighbors):
        results = []
        for i in range(len(x)):
            self.kNSet.clear()
            self.searchBallSubtree(1, x[i], n_neighbors)
            result = []

            for index in self.kNSet:
                result.append(self.pointsOriginal[index[1]])
            results.append(result)    
        return np.array(np.array(results))


# In[ ]:




