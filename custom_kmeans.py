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

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession


def parseVector(line):
    return np.array([float(x) for x in line.split(',')])


def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        #Manhattan distance
#         tempDist = np.sum((p - centers[i]))
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def compute_error(p, centers):
    center = centers[closestPoint(p,centers)]
    return np.sqrt(sum([x**2 for x in (p - center)]))


if __name__ == "__main__":

    spark = SparkSession\
    .builder\
    .appName("PythonKMeans")\
    .getOrCreate()

    # lines = spark.read.text('gs://dataproc-99f4856f-2ef2-4239-ab93-7e952ddaa6a8-europe-north1/pointdata2018.txt').rdd.map(lambda r: r[0])
    lines = spark.read.text('pointdata2018.txt').rdd.map(lambda r: r[0])
    # lines.takeSample(True, 100000, seed=int(datetime.timestamp(datetime.now())))

    data = lines.map(parseVector).cache()
    convergeDist = 0.0001
    tempDist = 1.0
    i = 0
    sse_list = list()
    for K in range(2,10):
        kPoints = data.takeSample(False, K, 5)
        while tempDist > convergeDist:
            closest = data.map(lambda p: (closestPoint(p, kPoints), (p, 1)))
            pointStats = closest.reduceByKey(lambda p1_c1, p2_c2:
                                            (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
            newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
            tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)
            ##updates new points coordinates
            for (iK, p) in newPoints:
                kPoints[iK] = p
        WSSSE = data.map(lambda point: compute_error(point, kPoints)).reduce(lambda x, y: x + y)
        sse_list.append(WSSSE)
        print(K, "- WSSE = ", WSSSE)
    print("Final centers: " + str(kPoints))
    sse_vector = np.array(sse_list)
    plt.plot(sse_vector)
    plt.show()
    spark.stop()