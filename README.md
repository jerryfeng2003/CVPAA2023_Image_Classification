# CVPPA2023_Image_Classification

In the challenge for "Image Classification of Nutrient Deficiencies in Winter Wheat and Winter Rye" organized for the CVPPA 2023 Workshop, we constructed a method that is based on the pre-trained swin-transformer v2 and utilizes tricks such as EMA and data argumentation. Our method achieves the mean accuracy of 90.1\% on the test set and gets 6-th place in the competition.

The detailed information has been shown in the [Github repository](https://github.com/jh-yi/DND-Diko-WWWR) of competition official and the dataset can be downloaded from [PhenoRoam](https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/1272b197-11ad-4138-a872-dc31d8051726).

# Installation

Our experiments are tested on the following environments: 4 TITAN V GPU, Python:3.10, Pytorch:2.0.1, CUDA:11.7
```
conda create --name CVPPA2023 python=3.10
conda activate CVPPA2023
git clone https://github.com/jerryfeng2003/CVPPA2023_Image_Classification
cd CVPPA2023_Image_Classification
unzip 1769319269-DND-Diko-WWWR.zip -d Image
conda install --yes --file requirements.txt

```
