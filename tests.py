#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import csv
import matplotlib.pyplot as plt

import random
import math
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pdb
from collections import Counter
import matplotlib.colors as mcolors

class KnnBalltreeClassifierTester(object):
    def __init__(self, model):
        self.model = model
        return

    def Gen2dDataset(self, points_num):
        points = []
        classes = []
        for i in range(points_num):
            tempX = np.random.uniform(-5,5)
            tempY = np.random.uniform(-5,5)
            points.append((tempX, tempY))
            if tempX <= 0 and tempY <= 0:
                classes.append(1)
            elif tempX <= 0 and tempY >= 0:
                classes.append(2)
            elif tempX >= 0 and tempY <= 0:
                classes.append(3)
            elif tempX >= 0 and tempY >= 0:
                classes.append(4)
        return np.array(points), np.array(classes)

    def Gen11dDataset(self, points_num):
        points = []
        classes = []
        for i in range(points_num):
            temp1 = np.random.uniform(-5,5)
            temp2 = np.random.uniform(-5,5)
            temp3 = np.random.uniform(-5,5)
            temp4 = np.random.uniform(-5,5)
            temp5 = np.random.uniform(-5,5)
            temp6 = np.random.uniform(-5,5)
            temp7 = np.random.uniform(-5,5)
            temp8 = np.random.uniform(-5,5)
            temp9 = np.random.uniform(-5,5)
            temp10 = np.random.uniform(-5,5)
            temp11 = np.random.uniform(-5,5)
            points.append((temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11))
            if temp1 <= 0 and temp3 <= 0:
                classes.append(1)
            elif temp2 <= 0 and temp4 >= 0:
                classes.append(2)
            elif temp5 >= 0 and temp9 <= 0:
                classes.append(3)
            elif temp10 >= 0 and temp3 >= 0:
                classes.append(4)
            elif temp6 >= 0 and temp5 >= 0:
                classes.append(5)
            elif temp2 >= 0 and temp9 >= 0:
                classes.append(3)
            elif temp10 >= 0 and temp11 >= 0:
                classes.append(4)
            elif temp7 >= 0 and temp3 >= 0:
                classes.append(2)
            elif temp5 >= 0 and temp3 >= 0:
                classes.append(1)
            elif temp6 >= 0 and temp2 >= 0:
                classes.append(4)
            else:
                classes.append(5)
        return np.array(points), np.array(classes)    
    
    def FitTest(self):
        np.random.seed(42)
        points, classes = self.Gen2dDataset(100)
        fig, ax = plt.subplots()
        ax.axis('equal')
        self.model.fit(points, classes)
        print("Точки в модели:")
        for point in self.model.points:
            print(point)
        ax.set_title('Объекты и классы')
        ax.scatter(points[:,0], points[:,1], c = classes, 
                norm = plt.Normalize(0, 5),
                s = 10)    


    def SortingByDimensionTest(self):
        np.random.seed(42)
        points, classes = self.Gen2dDataset(100)
        fig, ax = plt.subplots(2, 1)
        ax[0].axis('equal')
        ax[1].axis('equal')
        self.model.fit(points, classes)

        points2 = []
        for i in range(points.shape[0]):
            points2.append(self.model.Point(points[i], i))

        points3 = []
        for point in points2:
            points3.append(point.point[0])
        print(points3)

        ax[0].set_title('До сортировки')
        ax[0].scatter(range(0,100), points3, c = 'blue', s = 10)
        
        points2 = self.model.SortingByDimension(points2, 0, 100, 0)

        points3 = []
        for point in points2:
            points3.append(point.point[0])
        print(points3)

        ax[1].set_title('После сортировки')
        ax[1].scatter(range(0,100), points3, c = 'blue', s = 10)

    def ConstructTreeTest(self):
        np.random.seed(42)
        points, classes = self.Gen2dDataset(100)
        fig, ax = plt.subplots()
        ax.axis('equal')
        self.model.fit(points, classes)
        print("Узлы(вершины) дерева:")
        for node in self.model.nodes.values():
            print(node)
        for node in self.model.nodes.values():
            arrayOfPoints = []
            for point in node.points:
                arrayOfPoints.append(point)
                ax.scatter(points[:,0], points[:,1], c = classes, 
                norm = plt.Normalize(0, 5),
                s = 10)
            ax.scatter(node.pivot.point[0], node.pivot.point[1], c = 'red', 
                norm = plt.Normalize(0, 5),
                s = 25)
            circle1 = plt.Circle(node.pivot.point, node.radius, color='r', fill = False)
            ax.add_patch(circle1)
        
    def Visual2dPointClassTest(self):
        np.random.seed(42)
        points, classes = self.Gen2dDataset(100)

        fig, ax = plt.subplots()

        #ax.axis('equal')
        #colors = {1: 'r', 2: 'g', 3: 'b', 4: 'y'}
        ax.set_title('Объекты и классы')
        ax.scatter(points[:,0], points[:,1], c = classes, 
                norm = plt.Normalize(0, 5),
                #cmap = mcolors.ListedColormap(["b", "g","r","y","c","k"]), 
                s = 10)


        self.model.fit(points, classes)

        #print(len(model.points))
        checkPoints = [[2, 1], [-2, -2], [-3, 4.5], [0, -1]]
        #print(points)
        #print(classes)
        print("Точки для классфикации:")
        print(np.array(checkPoints))
        checkClasses = self.model.predict(checkPoints)
        print("Ближайшие соседи:")
        print(self.model.kneighbors(checkPoints, 3))
        print("Их классы:")
        print(checkClasses)
        #print(model.get_kneighbors_classes(checkPoints[0]))
        print("Вероятности (по классам):")
        print(self.model.predict_proba(checkPoints))
        print("Жирные точки - тестовые точки для классфикации:")
        ax.scatter(np.array(checkPoints)[:,0], np.array(checkPoints)[:,1], 
                c = checkClasses, norm = plt.Normalize(0, 5), 
                #cmap = mcolors.ListedColormap(["b", "g","r","y","c","k"]), 
                s = 40)
    
    def Score11dPointTest(self):
        np.random.seed(42)
        trainX, trainY = self.Gen11dDataset(1279)
        testX, testY = self.Gen11dDataset(320)
        print("Размер тренировачного искуственного датасета:")
        print(trainX.shape)
        print(trainY.shape)
        print("Размер тестового искуственного датасета:")
        print(testX.shape)
        print(testY.shape)
        self.model.fit(trainX, trainY)
        predictClasses = self.model.predict(testX)
        #print(model.predict_proba(testX))
        trueClasses = testY
        #print(predictClasses)
        #print(trueClasses)
        table = predictClasses == trueClasses
        print("Вероятности (по классам):")
        print(self.model.predict_proba(testX))
        #print(table)
        score = Counter(table)[True] / len(table)
        print("Иаблица истинной классификации тестовых данных(Если True - ")
        print("значит этот объект был классифицирован правильно): ")
        print(table)
        print("Отношение правильных ко всем (accuracy): ")
        print(score)


# In[ ]:




