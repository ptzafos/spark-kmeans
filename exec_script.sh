#!/bin/bash
for ((number=0; number <5; number++)){
	gcloud dataproc jobs submit pyspark --cluster cluster-4bae-spark --region europe-north1 sparkmeans.py --labels type=mllib,instance_num=2,iter_num=$number
}
for ((number=0; number <5; number++)){
	gcloud dataproc jobs submit pyspark --cluster cluster-4bae-spark --region europe-north1 euclidean_kmeans.py --labels type=euclidean,instance_num=2,iter_num=$number
}
for ((number=0; number <5; number++)){
	gcloud dataproc jobs submit pyspark --cluster cluster-4bae-spark --region europe-north1 manhattan_kmeans.py --labels type=manhattan,instance_num=2,iter_num=$number
}
