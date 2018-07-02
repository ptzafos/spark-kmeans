#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

from numpy import array
import numpy as np
import sys
from math import sqrt
from datetime import datetime

from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from matplotlib import pyplot
import time

def error(point, clusters):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  

    start_time = time.time()
    data = sc.textFile("pointdata2018.txt")
    parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))

    
    minK = 0
    minWSSE = float("+inf")
    for K in range(2,10):
        clusters = KMeans.train(parsedData, K, maxIterations=10, initializationMode="random", seed=int(time.time()))
        WSSSE = parsedData.map(lambda point: error(point, clusters)).reduce(lambda x, y: x + y)
        print("Kernel", K, "WSSSE=", WSSSE)
        if(WSSSE<minWSSE):
            minK=K

    print("Optimal number of kernels:", minK)

    clusters = KMeans.train(parsedData, minK, maxIterations=10, initializationMode="random", seed=int(time.time()))

    WSSSE = parsedData.map(lambda point: error(point,clusters)).reduce(lambda x, y: x + y)
    center = clusters.clusterCenters
    print("Cluster centers:", center)

    elapsed_time = time.time() - start_time
    print("MLlib kmean WSSSE = {}, time elapsed= {}\n".format(str(WSSSE), str(elapsed_time)))
    sc.stop()

    
    