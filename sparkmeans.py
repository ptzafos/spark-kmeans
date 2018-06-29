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

# $example on$
from numpy import array
import numpy as np
import sys
from math import sqrt
from datetime import datetime
# $example off$

from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
from matplotlib import pyplot
import time
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  # SparkContext

    # # $example on$
    # # Load and parse the data
    start_time = time.time()
    data = sc.textFile("pointdata2018.txt")
    parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))

    # # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, 4, maxIterations=10, initializationMode="random", seed=int(time.time()))


    ## Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("MLlib = " + str(WSSSE))

# # Save and load model
    elapsed_time = time.time() - start_time
    with open("mllib.txt", "a") as mllib_file:
        mllib_file.write(str(WSSSE))
        mllib_file.write(str(elapsed_time))
    # clusters.save(sc, "kmeans_mllib_model")
    # sameModel = KMeansModel.load(sc, "KmeansResults")
    # $example off$
    sc.stop()

    
    