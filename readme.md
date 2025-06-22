Za korištenje koda potrebno je nalaziti se u istom direktoriju kao i readme.md datoteka.

Za kreaciju skupa podataka:

python dataset/augment.py

Za particioniranje na skupove podataka za treniranje, testiranje i evaluaciju:

python dataset/partition.py

Za izrezivanje ploče:

Potrebno je dodati sliku ploče u direktorij images.

python code/cutout.py <ime_slike>

Za klasifikaciju izrezane ploče:

python -m code.classify.measurePerformance <model> <ime_slike>

Za određivanje mjera uspješnosti modela:

python -m code.classify.measurePerformance <model> measure

Mogući modeli:
kmeans-base
simple
ResNet50
ResNet50-ImageNet-1k
DINOv2-small
DINOv2-base
