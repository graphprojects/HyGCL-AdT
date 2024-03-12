# HyGCL-AdT

Dual-level Hypergraph Contrastive Learning with Adaptive Temperature Enhancement
====
Official source code of "Dual-level Hypergraph Contrastive Learning with Adaptive Temperature Enhancement" (WWW 2024) [Source Code](XXXXXXXX)


![HyGCL-AdT Framework](https://github.com/Tianyi-Billy-Ma/AdT/blob/main/images/HyGCL-AdT%20Framework.png)
Figure 1: The framework of HyGCL-AdT: (i) given a hypergraph $\mathcal{G}$, HyGCL-AdT samples $`A_{1}, A_{2}`$ from hypergraph augmentation set $\mathcal{T}$. Here $`A_1`$ and $`A_2`$ denote hyperedge removal and node dropping, respectively. With the augmented hypergraphs, it performs noise $`\delta`$ over augmented graphs for generating challenging hypergraph pairs; (ii) augmented graphs $`\widetilde{G}_{1}`$ and $`\widetilde{G}_{2}`$ are fed into HyGNNs encoder $f(\cdot)$ and projection head $`h(\cdot)`$ to get node and hyperedge embeddings. (iii) a dual-level contrastive strategy is designed to reach agreements among node embeddings from a local view and agreements among community embeddings from a global perspective. The dual-level contrast optimization is enhanced via the adaptive temperature $`\tau_{nd}`$ and $`\tau_{cm}`$, respectively.


## Usage

### Requirements

This code is developed and tested with python 3.11.5 with torch 2.1.2, and the required packages are listed in the `requirements.txt`.

Please run `pip install -r requirements.txt` to install all the dependencies. 

For datasets, please run `unzip ./data/raw_data.zip -d ./data/` to unzip data into folder `./data/`.

### HyGCL-AdT

To reproduce our experiments, simply run:
```Python
python main.py --dname dataset
```
Available datasets are: [ "Twitter-HyDrug", "cora", "citeseer", "coauthor_cora", "zoo", "Mushroom", "NTU2012"]

Available data augmentations are: [ "edge", "mask", "hyperedge", "mask_col", "adapt", "adapt_feat", "adapt_edge"]

For instance, for benchmark hypergraph dataset cora, please run:
```Python
python main.py --dname cora
```
As shown in Table I, Twitter-HyDrug has relatively large number of hyperedges, and requires large GPU memory for some calculation. We provides an option to move part of calculations to CPU. Our program will auto detect the dataset, and ask whether move part of calculation to CPU. If so, it may take longer for some calculation on CPU. 
For example, to reproduce our experiment on Twitter-HyDrug wtih some calculation on CPU, please run:
```
>> python main.py --dname Twitter-HyDrug
## "Due to the limited GPU memory, do you want to move some calculations to CPU? (y/n)"
>> y
## "Move some calculations to CPU!"
```

## Dataset
We adopt seven benchmark hypergraph datasets. The data statistics are listed in Table I. 

<div align="center">

| Dataset   | TWitter-HyDrug | Cora  | Citeseer | Cora-CA | Zoo | Mushroom | NTU2012 |
|-----------|----------------|-------|----------|---------|-----|----------|---------|
| # nodes       | 2,936          | 2,708 | 3,312    | 2,708   | 101 | 8,124    | 2012    |
| # hyperedges  | 33,893         | 1,579 | 1,079    | 1,072   | 43  | 298      | 2,012   |
| # features | 384            | 1,433 | 3,703    | 1,433   | 16  | 22       | 100     |
| # classes   | 4              | 7     | 6        | 7       | 7   | 2        | 67      |
| max $d(e)$ | 635  | 5    | 26       | 43      | 93  | 1808     | 5       |
| min $d(e)$ | 1    | 2    | 2        | 2       | 1   | 1        | 5       |
| avg $d(e)$ | 2.34 |3.03 | 3.20      | 4.28    | 39.93 | 136.31 | 5       |
| max $d(v)$ | 3,340 | 145  | 88       | 23      | 17  | 5        | 19      |
| min $d(v)$ | 0    | 0    | 0        | 0       | 17  | 5        | 1       |
| avg $d(v)$ | 31.49 | 1.77 | 1.04     | 1.69    | 17  | 5        | 5       |

Table I: Data Statistics of benchmark hypergraph datasets. d(e) and d(v) indicate the degree of hyperedges and nodes, respectively. 
</div>


## Twitter-HyDrug

We adopt the benchmark hypergraph dataset Twitter-HyDrug from [HyGCL-DC](https://github.com/GraphResearcher/HyGCL-DC). Twitter-HyDrug is a real-world hypergraph data that describes the drug trafficking communities on Twitter. Unlike HyGCL-DC that targets at drug trafficking community detection task (a multi-label node classification), we aim to identify drug user roles in drug trafficking activities on social media. 
To this end, we categorize node labels into four distinct roles: drug seller, drug buyer, drug user, and drug discussant, and each node is assigned to one and only one label. Consequently, we frame our problem as a multi-class node classification task. 

More detailed statistic of Twitter-HyDrug is listed in Table II.

<div align="center">

| Class Types    | # of Nodes | Hyperedge Types | # of Hyperedges |
|----------------|------------|-----------------|-----------------|
| Drug seller    | 455        | User-contain-emoji | 1,616          |
| Drug buyer     | 319        | User-engage-conversation | 13,650   |
| Drug user      | 1,650      | User-include-hashtag | 17,018       |
| Drug discussant| 512        | User-follow-user | 1,609           |
| Total Nodes    | 2,936      | Total Hyperedges | 33,893          |

Table II: Detailed data statistic of Twitter-HyDrug
</div>





## Contact

Yiyue Qian - yqian5@nd.edu

Tianyi Ma - tma2@nd.edu 

Discussions, suggestions and questions are always welcome!

## Citation

```
@inproceedings{,
  title={Dual-level Hypergraph Contrastive Learning with Adaptive Temperature Enhancement},
  author={Qian, Yiyue and Ma, Tianyi and Zhang, Chuxu  and Ye, Yanfang },
  booktitle={The International World Wide Web Conference},
  year={2024}
}
```


### Logger

We provide the completed running logger for the Twitter-HyDrug data in `./result/Twitter-HyDrug.log`.

Besides, this is a sample running logger which records the output and the model performance for Cora data:
```
Epoch: 00, Train Loss: 18.5445, Valid Loss: 1.8159, Test  Loss: 1.7983, Train Acc: 47.04%, Valid Acc: 33.58%, Test  Acc: 35.07%, Train F1: 29.06%, Valid F1: 19.62%, Test  F1: 20.25%, 
Epoch: 01, Train Loss: 18.3816, Valid Loss: 1.7099, Test  Loss: 1.6771, Train Acc: 60.00%, Valid Acc: 37.27%, Test  Acc: 39.22%, Train F1: 47.90%, Valid F1: 23.79%, Test  F1: 24.35%, 
Epoch: 02, Train Loss: 18.1994, Valid Loss: 1.6081, Test  Loss: 1.5668, Train Acc: 66.30%, Valid Acc: 39.85%, Test  Acc: 43.19%, Train F1: 55.90%, Valid F1: 26.61%, Test  F1: 29.63%, 
Epoch: 03, Train Loss: 18.0156, Valid Loss: 1.5061, Test  Loss: 1.4622, Train Acc: 73.33%, Valid Acc: 45.02%, Test  Acc: 46.47%, Train F1: 64.68%, Valid F1: 33.65%, Test  F1: 33.35%, 
Epoch: 04, Train Loss: 17.8567, Valid Loss: 1.3941, Test  Loss: 1.3519, Train Acc: 82.59%, Valid Acc: 50.55%, Test  Acc: 51.41%, Train F1: 78.08%, Valid F1: 39.35%, Test  F1: 39.72%, 
Epoch: 05, Train Loss: 17.6785, Valid Loss: 1.2853, Test  Loss: 1.2470, Train Acc: 87.78%, Valid Acc: 53.87%, Test  Acc: 56.81%, Train F1: 84.89%, Valid F1: 42.57%, Test  F1: 45.86%, 
Epoch: 06, Train Loss: 17.5324, Valid Loss: 1.1924, Test  Loss: 1.1575, Train Acc: 92.96%, Valid Acc: 59.41%, Test  Acc: 60.50%, Train F1: 91.31%, Valid F1: 48.81%, Test  F1: 51.00%, 
Epoch: 07, Train Loss: 17.3587, Valid Loss: 1.1174, Test  Loss: 1.0832, Train Acc: 95.19%, Valid Acc: 64.94%, Test  Acc: 63.91%, Train F1: 93.90%, Valid F1: 58.68%, Test  F1: 54.83%, 
Epoch: 08, Train Loss: 17.1775, Valid Loss: 1.0573, Test  Loss: 1.0210, Train Acc: 97.04%, Valid Acc: 65.68%, Test  Acc: 66.54%, Train F1: 96.10%, Valid F1: 59.76%, Test  F1: 58.11%, 
Epoch: 09, Train Loss: 17.0377, Valid Loss: 1.0125, Test  Loss: 0.9716, Train Acc: 97.78%, Valid Acc: 68.27%, Test  Acc: 68.44%, Train F1: 96.72%, Valid F1: 63.07%, Test  F1: 60.89%, 
Epoch: 10, Train Loss: 16.9099, Valid Loss: 0.9822, Test  Loss: 0.9388, Train Acc: 98.15%, Valid Acc: 69.74%, Test  Acc: 69.91%, Train F1: 97.15%, Valid F1: 65.48%, Test  F1: 63.61%, 
Epoch: 11, Train Loss: 16.8113, Valid Loss: 0.9721, Test  Loss: 0.9236, Train Acc: 99.63%, Valid Acc: 69.37%, Test  Acc: 70.79%, Train F1: 99.72%, Valid F1: 64.57%, Test  F1: 65.13%, 
Epoch: 12, Train Loss: 16.6706, Valid Loss: 0.9757, Test  Loss: 0.9233, Train Acc: 99.63%, Valid Acc: 69.37%, Test  Acc: 71.48%, Train F1: 99.72%, Valid F1: 65.71%, Test  F1: 66.62%, 
Epoch: 13, Train Loss: 16.5034, Valid Loss: 0.9936, Test  Loss: 0.9366, Train Acc: 100.00%, Valid Acc: 70.48%, Test  Acc: 72.50%, Train F1: 100.00%, Valid F1: 66.29%, Test  F1: 68.22%, 
Epoch: 14, Train Loss: 16.4938, Valid Loss: 1.0216, Test  Loss: 0.9594, Train Acc: 100.00%, Valid Acc: 71.22%, Test  Acc: 72.73%, Train F1: 100.00%, Valid F1: 67.14%, Test  F1: 68.90%, 
Epoch: 15, Train Loss: 16.4133, Valid Loss: 1.0524, Test  Loss: 0.9828, Train Acc: 100.00%, Valid Acc: 71.96%, Test  Acc: 73.70%, Train F1: 100.00%, Valid F1: 67.72%, Test  F1: 70.57%, 
Epoch: 16, Train Loss: 16.3382, Valid Loss: 1.0838, Test  Loss: 1.0076, Train Acc: 100.00%, Valid Acc: 71.96%, Test  Acc: 73.74%, Train F1: 100.00%, Valid F1: 67.66%, Test  F1: 70.92%, 
Epoch: 17, Train Loss: 16.3427, Valid Loss: 1.1149, Test  Loss: 1.0382, Train Acc: 100.00%, Valid Acc: 72.32%, Test  Acc: 74.16%, Train F1: 100.00%, Valid F1: 67.69%, Test  F1: 71.58%, 
Epoch: 18, Train Loss: 16.2599, Valid Loss: 1.1485, Test  Loss: 1.0734, Train Acc: 100.00%, Valid Acc: 71.22%, Test  Acc: 74.34%, Train F1: 100.00%, Valid F1: 66.97%, Test  F1: 71.80%, 
Epoch: 19, Train Loss: 16.2701, Valid Loss: 1.1779, Test  Loss: 1.1136, Train Acc: 100.00%, Valid Acc: 70.48%, Test  Acc: 74.43%, Train F1: 100.00%, Valid F1: 66.43%, Test  F1: 72.02%, 
Epoch: 20, Train Loss: 16.2513, Valid Loss: 1.2094, Test  Loss: 1.1559, Train Acc: 100.00%, Valid Acc: 71.59%, Test  Acc: 74.25%, Train F1: 100.00%, Valid F1: 68.12%, Test  F1: 71.81%, 
Epoch: 21, Train Loss: 16.1494, Valid Loss: 1.2388, Test  Loss: 1.1977, Train Acc: 100.00%, Valid Acc: 71.22%, Test  Acc: 73.88%, Train F1: 100.00%, Valid F1: 67.51%, Test  F1: 71.43%, 
Epoch: 22, Train Loss: 16.2073, Valid Loss: 1.2660, Test  Loss: 1.2396, Train Acc: 100.00%, Valid Acc: 71.59%, Test  Acc: 73.93%, Train F1: 100.00%, Valid F1: 67.73%, Test  F1: 71.58%, 
Epoch: 23, Train Loss: 16.1869, Valid Loss: 1.2894, Test  Loss: 1.2736, Train Acc: 100.00%, Valid Acc: 71.96%, Test  Acc: 73.88%, Train F1: 100.00%, Valid F1: 68.18%, Test  F1: 71.57%, 
Epoch: 24, Train Loss: 16.1406, Valid Loss: 1.3150, Test  Loss: 1.3087, Train Acc: 100.00%, Valid Acc: 72.32%, Test  Acc: 74.02%, Train F1: 100.00%, Valid F1: 68.38%, Test  F1: 71.53%, 
Epoch: 25, Train Loss: 16.0886, Valid Loss: 1.3429, Test  Loss: 1.3404, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 73.93%, Train F1: 100.00%, Valid F1: 69.80%, Test  F1: 71.51%, 
Epoch: 26, Train Loss: 16.1030, Valid Loss: 1.3705, Test  Loss: 1.3693, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 73.93%, Train F1: 100.00%, Valid F1: 70.26%, Test  F1: 71.52%, 
Epoch: 27, Train Loss: 16.0474, Valid Loss: 1.3942, Test  Loss: 1.3940, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 74.02%, Train F1: 100.00%, Valid F1: 70.69%, Test  F1: 71.59%, 
Epoch: 28, Train Loss: 16.0372, Valid Loss: 1.4164, Test  Loss: 1.4163, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 74.16%, Train F1: 100.00%, Valid F1: 70.69%, Test  F1: 71.78%, 
Epoch: 29, Train Loss: 16.0939, Valid Loss: 1.4415, Test  Loss: 1.4350, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 74.53%, Train F1: 100.00%, Valid F1: 70.99%, Test  F1: 72.26%, 
Epoch: 30, Train Loss: 16.0699, Valid Loss: 1.4657, Test  Loss: 1.4522, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 74.48%, Train F1: 100.00%, Valid F1: 70.99%, Test  F1: 72.15%, 
Epoch: 31, Train Loss: 16.0326, Valid Loss: 1.4867, Test  Loss: 1.4714, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 74.25%, Train F1: 100.00%, Valid F1: 70.15%, Test  F1: 71.86%, 
Epoch: 32, Train Loss: 16.0215, Valid Loss: 1.5065, Test  Loss: 1.4871, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 74.20%, Train F1: 100.00%, Valid F1: 71.22%, Test  F1: 71.73%, 
Epoch: 33, Train Loss: 16.0036, Valid Loss: 1.5233, Test  Loss: 1.5029, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 74.30%, Train F1: 100.00%, Valid F1: 71.16%, Test  F1: 71.71%, 
Epoch: 34, Train Loss: 15.9739, Valid Loss: 1.5370, Test  Loss: 1.5142, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 74.43%, Train F1: 100.00%, Valid F1: 72.18%, Test  F1: 71.81%, 
Epoch: 35, Train Loss: 15.9941, Valid Loss: 1.5568, Test  Loss: 1.5216, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 74.39%, Train F1: 100.00%, Valid F1: 71.48%, Test  F1: 71.70%, 
Epoch: 36, Train Loss: 16.0092, Valid Loss: 1.5724, Test  Loss: 1.5248, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 74.48%, Train F1: 100.00%, Valid F1: 71.19%, Test  F1: 71.73%, 
Epoch: 37, Train Loss: 15.9712, Valid Loss: 1.5839, Test  Loss: 1.5251, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 74.57%, Train F1: 100.00%, Valid F1: 71.55%, Test  F1: 71.85%, 
Epoch: 38, Train Loss: 15.9792, Valid Loss: 1.5919, Test  Loss: 1.5232, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 74.62%, Train F1: 100.00%, Valid F1: 71.99%, Test  F1: 71.92%, 
Epoch: 39, Train Loss: 15.9465, Valid Loss: 1.5951, Test  Loss: 1.5214, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 74.80%, Train F1: 100.00%, Valid F1: 72.39%, Test  F1: 72.02%, 
Epoch: 40, Train Loss: 15.9550, Valid Loss: 1.5981, Test  Loss: 1.5159, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 75.08%, Train F1: 100.00%, Valid F1: 71.54%, Test  F1: 72.37%, 
Epoch: 41, Train Loss: 15.9455, Valid Loss: 1.5962, Test  Loss: 1.5132, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 75.31%, Train F1: 100.00%, Valid F1: 71.84%, Test  F1: 72.74%, 
Epoch: 42, Train Loss: 15.9750, Valid Loss: 1.5939, Test  Loss: 1.5119, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 75.45%, Train F1: 100.00%, Valid F1: 71.84%, Test  F1: 73.01%, 
Epoch: 43, Train Loss: 15.9169, Valid Loss: 1.5962, Test  Loss: 1.5117, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 75.68%, Train F1: 100.00%, Valid F1: 72.01%, Test  F1: 73.28%, 
Epoch: 44, Train Loss: 15.9647, Valid Loss: 1.5993, Test  Loss: 1.5151, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 75.77%, Train F1: 100.00%, Valid F1: 72.50%, Test  F1: 73.38%, 
Epoch: 45, Train Loss: 15.9644, Valid Loss: 1.5989, Test  Loss: 1.5132, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.00%, Train F1: 100.00%, Valid F1: 71.68%, Test  F1: 73.79%, 
Epoch: 46, Train Loss: 15.9206, Valid Loss: 1.6024, Test  Loss: 1.5098, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.71%, Test  F1: 74.24%, 
Epoch: 47, Train Loss: 15.9062, Valid Loss: 1.6076, Test  Loss: 1.5085, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 72.10%, Test  F1: 74.28%, 
Epoch: 48, Train Loss: 15.8676, Valid Loss: 1.6168, Test  Loss: 1.5095, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.08%, Test  F1: 74.27%, 
Epoch: 49, Train Loss: 15.8867, Valid Loss: 1.6237, Test  Loss: 1.5103, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.89%, Test  F1: 74.37%, 
Epoch: 50, Train Loss: 15.8735, Valid Loss: 1.6340, Test  Loss: 1.5134, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 71.50%, Test  F1: 74.60%, 
Epoch: 51, Train Loss: 15.8642, Valid Loss: 1.6419, Test  Loss: 1.5211, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.50%, Test  F1: 74.36%, 
Epoch: 52, Train Loss: 15.8927, Valid Loss: 1.6461, Test  Loss: 1.5297, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 74.26%, 
Epoch: 53, Train Loss: 15.9086, Valid Loss: 1.6525, Test  Loss: 1.5383, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.90%, Test  F1: 74.19%, 
Epoch: 54, Train Loss: 15.8567, Valid Loss: 1.6611, Test  Loss: 1.5455, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 72.20%, Test  F1: 73.85%, 
Epoch: 55, Train Loss: 15.8541, Valid Loss: 1.6739, Test  Loss: 1.5528, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 75.96%, Train F1: 100.00%, Valid F1: 72.15%, Test  F1: 73.50%, 
Epoch: 56, Train Loss: 15.8915, Valid Loss: 1.6860, Test  Loss: 1.5489, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 75.91%, Train F1: 100.00%, Valid F1: 71.71%, Test  F1: 73.47%, 
Epoch: 57, Train Loss: 15.8321, Valid Loss: 1.6985, Test  Loss: 1.5492, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 75.77%, Train F1: 100.00%, Valid F1: 71.71%, Test  F1: 73.32%, 
Epoch: 58, Train Loss: 15.8673, Valid Loss: 1.7088, Test  Loss: 1.5493, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 75.96%, Train F1: 100.00%, Valid F1: 71.66%, Test  F1: 73.49%, 
Epoch: 59, Train Loss: 15.9079, Valid Loss: 1.7113, Test  Loss: 1.5425, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.00%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 73.50%, 
Epoch: 60, Train Loss: 15.8330, Valid Loss: 1.7101, Test  Loss: 1.5346, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 74.08%, 
Epoch: 61, Train Loss: 15.8057, Valid Loss: 1.7108, Test  Loss: 1.5285, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 71.03%, Test  F1: 73.86%, 
Epoch: 62, Train Loss: 15.8504, Valid Loss: 1.7080, Test  Loss: 1.5233, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 71.03%, Test  F1: 73.95%, 
Epoch: 63, Train Loss: 15.7778, Valid Loss: 1.7040, Test  Loss: 1.5176, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.90%, Test  F1: 74.11%, 
Epoch: 64, Train Loss: 15.8321, Valid Loss: 1.7050, Test  Loss: 1.5113, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 70.85%, Test  F1: 74.36%, 
Epoch: 65, Train Loss: 15.8056, Valid Loss: 1.7079, Test  Loss: 1.5099, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.60%, Test  F1: 74.06%, 
Epoch: 66, Train Loss: 15.8470, Valid Loss: 1.7121, Test  Loss: 1.5084, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 70.47%, Test  F1: 74.04%, 
Epoch: 67, Train Loss: 15.8164, Valid Loss: 1.7145, Test  Loss: 1.5056, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 70.71%, Test  F1: 74.07%, 
Epoch: 68, Train Loss: 15.8388, Valid Loss: 1.7155, Test  Loss: 1.5018, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.71%, Test  F1: 74.40%, 
Epoch: 69, Train Loss: 15.8063, Valid Loss: 1.7153, Test  Loss: 1.4993, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 71.54%, Test  F1: 74.34%, 
Epoch: 70, Train Loss: 15.8121, Valid Loss: 1.7174, Test  Loss: 1.4976, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 71.54%, Test  F1: 74.41%, 
Epoch: 71, Train Loss: 15.7804, Valid Loss: 1.7169, Test  Loss: 1.4950, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.89%, Test  F1: 74.70%, 
Epoch: 72, Train Loss: 15.7794, Valid Loss: 1.7163, Test  Loss: 1.4962, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 74.68%, 
Epoch: 73, Train Loss: 15.8139, Valid Loss: 1.7131, Test  Loss: 1.4962, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.38%, Test  F1: 74.50%, 
Epoch: 74, Train Loss: 15.8029, Valid Loss: 1.7074, Test  Loss: 1.4952, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.17%, Test  F1: 74.32%, 
Epoch: 75, Train Loss: 15.7920, Valid Loss: 1.6945, Test  Loss: 1.4945, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.17%, Test  F1: 74.26%, 
Epoch: 76, Train Loss: 15.7970, Valid Loss: 1.6825, Test  Loss: 1.4934, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.74%, Test  F1: 74.25%, 
Epoch: 77, Train Loss: 15.8049, Valid Loss: 1.6778, Test  Loss: 1.4987, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.69%, Test  F1: 74.09%, 
Epoch: 78, Train Loss: 15.7586, Valid Loss: 1.6721, Test  Loss: 1.4983, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.72%, Test  F1: 74.15%, 
Epoch: 79, Train Loss: 15.7507, Valid Loss: 1.6615, Test  Loss: 1.4980, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.10%, Train F1: 100.00%, Valid F1: 71.08%, Test  F1: 74.00%, 
Epoch: 80, Train Loss: 15.7414, Valid Loss: 1.6540, Test  Loss: 1.4968, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 70.71%, Test  F1: 74.12%, 
Epoch: 81, Train Loss: 15.7614, Valid Loss: 1.6501, Test  Loss: 1.4979, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.10%, Train F1: 100.00%, Valid F1: 70.25%, Test  F1: 74.01%, 
Epoch: 82, Train Loss: 15.7744, Valid Loss: 1.6485, Test  Loss: 1.5001, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.10%, Train F1: 100.00%, Valid F1: 70.23%, Test  F1: 74.01%, 
Epoch: 83, Train Loss: 15.7540, Valid Loss: 1.6510, Test  Loss: 1.5025, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.51%, Test  F1: 74.25%, 
Epoch: 84, Train Loss: 15.7826, Valid Loss: 1.6583, Test  Loss: 1.5057, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.15%, Test  F1: 74.21%, 
Epoch: 85, Train Loss: 15.7923, Valid Loss: 1.6615, Test  Loss: 1.5068, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.43%, Test  F1: 74.28%, 
Epoch: 86, Train Loss: 15.7714, Valid Loss: 1.6670, Test  Loss: 1.5069, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.43%, Test  F1: 74.42%, 
Epoch: 87, Train Loss: 15.7579, Valid Loss: 1.6744, Test  Loss: 1.5052, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 70.34%, Test  F1: 74.54%, 
Epoch: 88, Train Loss: 15.7884, Valid Loss: 1.6790, Test  Loss: 1.5013, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 70.81%, Test  F1: 74.47%, 
Epoch: 89, Train Loss: 15.7724, Valid Loss: 1.6817, Test  Loss: 1.4959, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.23%, Test  F1: 74.79%, 
Epoch: 90, Train Loss: 15.7096, Valid Loss: 1.6813, Test  Loss: 1.4893, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 70.18%, Test  F1: 75.07%, 
Epoch: 91, Train Loss: 15.7460, Valid Loss: 1.6832, Test  Loss: 1.4861, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.66%, Test  F1: 74.95%, 
Epoch: 92, Train Loss: 15.7555, Valid Loss: 1.6830, Test  Loss: 1.4844, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 70.42%, Test  F1: 74.98%, 
Epoch: 93, Train Loss: 15.6988, Valid Loss: 1.6831, Test  Loss: 1.4842, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 70.80%, Test  F1: 75.05%, 
Epoch: 94, Train Loss: 15.7311, Valid Loss: 1.6856, Test  Loss: 1.4878, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 70.16%, Test  F1: 75.03%, 
Epoch: 95, Train Loss: 15.7324, Valid Loss: 1.6999, Test  Loss: 1.4957, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.50%, Test  F1: 75.06%, 
Epoch: 96, Train Loss: 15.7423, Valid Loss: 1.7152, Test  Loss: 1.5025, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.82%, Test  F1: 75.09%, 
Epoch: 97, Train Loss: 15.7338, Valid Loss: 1.7283, Test  Loss: 1.5084, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 70.82%, Test  F1: 74.98%, 
Epoch: 98, Train Loss: 15.7335, Valid Loss: 1.7370, Test  Loss: 1.5113, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.63%, Test  F1: 74.92%, 
Epoch: 99, Train Loss: 15.7206, Valid Loss: 1.7444, Test  Loss: 1.5129, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.15%, Test  F1: 74.83%, 
Epoch: 100, Train Loss: 15.7782, Valid Loss: 1.7515, Test  Loss: 1.5165, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 70.42%, Test  F1: 74.92%, 
Epoch: 101, Train Loss: 15.7298, Valid Loss: 1.7666, Test  Loss: 1.5230, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.06%, Test  F1: 74.57%, 
Epoch: 102, Train Loss: 15.6904, Valid Loss: 1.7816, Test  Loss: 1.5291, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.22%, Test  F1: 74.63%, 
Epoch: 103, Train Loss: 15.7120, Valid Loss: 1.7918, Test  Loss: 1.5332, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.18%, Test  F1: 74.47%, 
Epoch: 104, Train Loss: 15.7438, Valid Loss: 1.7962, Test  Loss: 1.5337, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.39%, Test  F1: 74.29%, 
Epoch: 105, Train Loss: 15.7557, Valid Loss: 1.8020, Test  Loss: 1.5324, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.39%, Test  F1: 74.20%, 
Epoch: 106, Train Loss: 15.6973, Valid Loss: 1.8089, Test  Loss: 1.5321, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 69.97%, Test  F1: 74.11%, 
Epoch: 107, Train Loss: 15.7271, Valid Loss: 1.8173, Test  Loss: 1.5337, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 69.74%, Test  F1: 74.08%, 
Epoch: 108, Train Loss: 15.6995, Valid Loss: 1.8250, Test  Loss: 1.5362, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 70.28%, Test  F1: 74.14%, 
Epoch: 109, Train Loss: 15.7674, Valid Loss: 1.8249, Test  Loss: 1.5365, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.00%, Test  F1: 74.22%, 
Epoch: 110, Train Loss: 15.7167, Valid Loss: 1.8225, Test  Loss: 1.5358, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.00%, Test  F1: 74.25%, 
Epoch: 111, Train Loss: 15.6849, Valid Loss: 1.8197, Test  Loss: 1.5354, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.00%, Test  F1: 74.27%, 
Epoch: 112, Train Loss: 15.7191, Valid Loss: 1.8155, Test  Loss: 1.5349, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.00%, Test  F1: 74.23%, 
Epoch: 113, Train Loss: 15.7056, Valid Loss: 1.8109, Test  Loss: 1.5340, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 70.58%, Test  F1: 74.15%, 
Epoch: 114, Train Loss: 15.7207, Valid Loss: 1.8063, Test  Loss: 1.5310, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 70.58%, Test  F1: 74.07%, 
Epoch: 115, Train Loss: 15.7081, Valid Loss: 1.8077, Test  Loss: 1.5292, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 70.21%, Test  F1: 74.12%, 
Epoch: 116, Train Loss: 15.7223, Valid Loss: 1.8074, Test  Loss: 1.5255, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.21%, Test  F1: 74.23%, 
Epoch: 117, Train Loss: 15.7346, Valid Loss: 1.8031, Test  Loss: 1.5220, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 70.21%, Test  F1: 74.16%, 
Epoch: 118, Train Loss: 15.6890, Valid Loss: 1.7969, Test  Loss: 1.5189, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.95%, Test  F1: 74.38%, 
Epoch: 119, Train Loss: 15.7190, Valid Loss: 1.7916, Test  Loss: 1.5160, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.99%, Test  F1: 74.44%, 
Epoch: 120, Train Loss: 15.6848, Valid Loss: 1.7877, Test  Loss: 1.5142, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.76%, Test  F1: 74.54%, 
Epoch: 121, Train Loss: 15.7453, Valid Loss: 1.7838, Test  Loss: 1.5109, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 70.68%, Test  F1: 74.74%, 
Epoch: 122, Train Loss: 15.6963, Valid Loss: 1.7791, Test  Loss: 1.5082, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.68%, Test  F1: 74.98%, 
Epoch: 123, Train Loss: 15.7000, Valid Loss: 1.7733, Test  Loss: 1.5054, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 70.94%, Test  F1: 74.81%, 
Epoch: 124, Train Loss: 15.7297, Valid Loss: 1.7639, Test  Loss: 1.4993, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.94%, Test  F1: 75.04%, 
Epoch: 125, Train Loss: 15.6683, Valid Loss: 1.7560, Test  Loss: 1.4959, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 70.94%, Test  F1: 74.95%, 
Epoch: 126, Train Loss: 15.6757, Valid Loss: 1.7472, Test  Loss: 1.4922, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 70.59%, Test  F1: 75.06%, 
Epoch: 127, Train Loss: 15.6892, Valid Loss: 1.7389, Test  Loss: 1.4882, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 70.59%, Test  F1: 75.21%, 
Epoch: 128, Train Loss: 15.7006, Valid Loss: 1.7336, Test  Loss: 1.4842, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.59%, Test  F1: 75.00%, 
Epoch: 129, Train Loss: 15.6764, Valid Loss: 1.7315, Test  Loss: 1.4827, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 70.60%, Test  F1: 74.90%, 
Epoch: 130, Train Loss: 15.7306, Valid Loss: 1.7303, Test  Loss: 1.4781, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.86%, Test  F1: 75.07%, 
Epoch: 131, Train Loss: 15.6895, Valid Loss: 1.7286, Test  Loss: 1.4736, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.45%, Test  F1: 75.05%, 
Epoch: 132, Train Loss: 15.6493, Valid Loss: 1.7290, Test  Loss: 1.4709, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.20%, Test  F1: 75.03%, 
Epoch: 133, Train Loss: 15.6902, Valid Loss: 1.7270, Test  Loss: 1.4692, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.68%, Test  F1: 75.08%, 
Epoch: 134, Train Loss: 15.6697, Valid Loss: 1.7260, Test  Loss: 1.4696, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.14%, Test  F1: 75.17%, 
Epoch: 135, Train Loss: 15.6768, Valid Loss: 1.7240, Test  Loss: 1.4735, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.64%, Test  F1: 75.19%, 
Epoch: 136, Train Loss: 15.7090, Valid Loss: 1.7276, Test  Loss: 1.4788, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 71.04%, Test  F1: 74.81%, 
Epoch: 137, Train Loss: 15.7170, Valid Loss: 1.7339, Test  Loss: 1.4856, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 71.05%, Test  F1: 74.97%, 
Epoch: 138, Train Loss: 15.7000, Valid Loss: 1.7384, Test  Loss: 1.4910, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 71.05%, Test  F1: 75.08%, 
Epoch: 139, Train Loss: 15.7107, Valid Loss: 1.7415, Test  Loss: 1.4927, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 75.17%, 
Epoch: 140, Train Loss: 15.6843, Valid Loss: 1.7430, Test  Loss: 1.4944, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 75.06%, 
Epoch: 141, Train Loss: 15.6639, Valid Loss: 1.7434, Test  Loss: 1.4947, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 75.11%, 
Epoch: 142, Train Loss: 15.7212, Valid Loss: 1.7368, Test  Loss: 1.4977, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 75.19%, 
Epoch: 143, Train Loss: 15.6480, Valid Loss: 1.7330, Test  Loss: 1.5025, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.70%, Test  F1: 75.05%, 
Epoch: 144, Train Loss: 15.7075, Valid Loss: 1.7318, Test  Loss: 1.5091, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 70.92%, Test  F1: 75.08%, 
Epoch: 145, Train Loss: 15.6745, Valid Loss: 1.7329, Test  Loss: 1.5160, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.56%, Test  F1: 75.03%, 
Epoch: 146, Train Loss: 15.6758, Valid Loss: 1.7285, Test  Loss: 1.5251, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 70.95%, Test  F1: 74.91%, 
Epoch: 147, Train Loss: 15.6901, Valid Loss: 1.7259, Test  Loss: 1.5300, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 70.95%, Test  F1: 74.91%, 
Epoch: 148, Train Loss: 15.6593, Valid Loss: 1.7259, Test  Loss: 1.5340, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 70.95%, Test  F1: 74.86%, 
Epoch: 149, Train Loss: 15.6710, Valid Loss: 1.7331, Test  Loss: 1.5387, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 71.73%, Test  F1: 74.67%, 
Epoch: 150, Train Loss: 15.6691, Valid Loss: 1.7390, Test  Loss: 1.5429, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.98%, Test  F1: 74.59%, 
Epoch: 151, Train Loss: 15.6461, Valid Loss: 1.7440, Test  Loss: 1.5434, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 74.62%, 
Epoch: 152, Train Loss: 15.6612, Valid Loss: 1.7522, Test  Loss: 1.5465, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 74.93%, 
Epoch: 153, Train Loss: 15.7086, Valid Loss: 1.7609, Test  Loss: 1.5499, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 75.08%, 
Epoch: 154, Train Loss: 15.6607, Valid Loss: 1.7721, Test  Loss: 1.5514, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 72.30%, Test  F1: 75.13%, 
Epoch: 155, Train Loss: 15.6662, Valid Loss: 1.7847, Test  Loss: 1.5535, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 72.30%, Test  F1: 75.24%, 
Epoch: 156, Train Loss: 15.6476, Valid Loss: 1.7962, Test  Loss: 1.5547, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.01%, Test  F1: 75.16%, 
Epoch: 157, Train Loss: 15.6701, Valid Loss: 1.8012, Test  Loss: 1.5538, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 72.01%, Test  F1: 75.32%, 
Epoch: 158, Train Loss: 15.6757, Valid Loss: 1.8064, Test  Loss: 1.5508, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.77%, Test  F1: 75.11%, 
Epoch: 159, Train Loss: 15.6468, Valid Loss: 1.8103, Test  Loss: 1.5499, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 71.38%, Test  F1: 75.31%, 
Epoch: 160, Train Loss: 15.6492, Valid Loss: 1.8097, Test  Loss: 1.5502, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 71.38%, Test  F1: 75.26%, 
Epoch: 161, Train Loss: 15.6587, Valid Loss: 1.8074, Test  Loss: 1.5495, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 71.38%, Test  F1: 75.19%, 
Epoch: 162, Train Loss: 15.6520, Valid Loss: 1.8103, Test  Loss: 1.5500, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.03%, Test  F1: 75.22%, 
Epoch: 163, Train Loss: 15.6805, Valid Loss: 1.8126, Test  Loss: 1.5499, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.39%, Test  F1: 75.05%, 
Epoch: 164, Train Loss: 15.6334, Valid Loss: 1.8163, Test  Loss: 1.5508, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.43%, Test  F1: 75.23%, 
Epoch: 165, Train Loss: 15.6550, Valid Loss: 1.8193, Test  Loss: 1.5529, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.43%, Test  F1: 75.08%, 
Epoch: 166, Train Loss: 15.6744, Valid Loss: 1.8176, Test  Loss: 1.5536, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 70.66%, Test  F1: 75.04%, 
Epoch: 167, Train Loss: 15.6415, Valid Loss: 1.8183, Test  Loss: 1.5574, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 70.07%, Test  F1: 75.07%, 
Epoch: 168, Train Loss: 15.6407, Valid Loss: 1.8157, Test  Loss: 1.5623, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 70.07%, Test  F1: 74.94%, 
Epoch: 169, Train Loss: 15.6135, Valid Loss: 1.8121, Test  Loss: 1.5668, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 70.05%, Test  F1: 74.91%, 
Epoch: 170, Train Loss: 15.6465, Valid Loss: 1.8074, Test  Loss: 1.5712, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 70.05%, Test  F1: 74.65%, 
Epoch: 171, Train Loss: 15.6511, Valid Loss: 1.7999, Test  Loss: 1.5743, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.55%, Test  F1: 74.51%, 
Epoch: 172, Train Loss: 15.6111, Valid Loss: 1.7922, Test  Loss: 1.5763, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.20%, Test  F1: 74.35%, 
Epoch: 173, Train Loss: 15.6506, Valid Loss: 1.7755, Test  Loss: 1.5738, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.54%, Test  F1: 74.35%, 
Epoch: 174, Train Loss: 15.6634, Valid Loss: 1.7625, Test  Loss: 1.5716, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 71.02%, Test  F1: 74.21%, 
Epoch: 175, Train Loss: 15.6923, Valid Loss: 1.7580, Test  Loss: 1.5704, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.26%, Test  F1: 74.34%, 
Epoch: 176, Train Loss: 15.5949, Valid Loss: 1.7584, Test  Loss: 1.5707, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.30%, Test  F1: 74.19%, 
Epoch: 177, Train Loss: 15.6419, Valid Loss: 1.7594, Test  Loss: 1.5715, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 74.18%, 
Epoch: 178, Train Loss: 15.6601, Valid Loss: 1.7619, Test  Loss: 1.5721, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 74.17%, 
Epoch: 179, Train Loss: 15.6437, Valid Loss: 1.7645, Test  Loss: 1.5760, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 74.31%, 
Epoch: 180, Train Loss: 15.6407, Valid Loss: 1.7679, Test  Loss: 1.5810, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 74.21%, 
Epoch: 181, Train Loss: 15.6443, Valid Loss: 1.7708, Test  Loss: 1.5852, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 71.23%, Test  F1: 74.25%, 
Epoch: 182, Train Loss: 15.6726, Valid Loss: 1.7730, Test  Loss: 1.5822, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.23%, Test  F1: 74.46%, 
Epoch: 183, Train Loss: 15.6176, Valid Loss: 1.7751, Test  Loss: 1.5788, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 74.52%, 
Epoch: 184, Train Loss: 15.6769, Valid Loss: 1.7752, Test  Loss: 1.5766, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.78%, Test  F1: 74.42%, 
Epoch: 185, Train Loss: 15.6156, Valid Loss: 1.7767, Test  Loss: 1.5755, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 71.46%, Test  F1: 74.37%, 
Epoch: 186, Train Loss: 15.6495, Valid Loss: 1.7795, Test  Loss: 1.5747, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.46%, Test  F1: 74.49%, 
Epoch: 187, Train Loss: 15.6441, Valid Loss: 1.7805, Test  Loss: 1.5691, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.44%, Test  F1: 74.57%, 
Epoch: 188, Train Loss: 15.6493, Valid Loss: 1.7833, Test  Loss: 1.5625, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 71.94%, Test  F1: 74.24%, 
Epoch: 189, Train Loss: 15.6440, Valid Loss: 1.7878, Test  Loss: 1.5597, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.87%, Test  F1: 74.58%, 
Epoch: 190, Train Loss: 15.6776, Valid Loss: 1.7905, Test  Loss: 1.5600, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.87%, Test  F1: 74.49%, 
Epoch: 191, Train Loss: 15.6000, Valid Loss: 1.7919, Test  Loss: 1.5600, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 72.28%, Test  F1: 74.59%, 
Epoch: 192, Train Loss: 15.6467, Valid Loss: 1.7863, Test  Loss: 1.5585, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 72.28%, Test  F1: 74.53%, 
Epoch: 193, Train Loss: 15.6085, Valid Loss: 1.7795, Test  Loss: 1.5585, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 72.28%, Test  F1: 74.64%, 
Epoch: 194, Train Loss: 15.6235, Valid Loss: 1.7676, Test  Loss: 1.5577, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 72.28%, Test  F1: 74.76%, 
Epoch: 195, Train Loss: 15.6305, Valid Loss: 1.7537, Test  Loss: 1.5554, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.23%, Test  F1: 74.90%, 
Epoch: 196, Train Loss: 15.6248, Valid Loss: 1.7402, Test  Loss: 1.5546, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.88%, Test  F1: 75.06%, 
Epoch: 197, Train Loss: 15.6220, Valid Loss: 1.7343, Test  Loss: 1.5582, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.91%, Test  F1: 74.96%, 
Epoch: 198, Train Loss: 15.6442, Valid Loss: 1.7324, Test  Loss: 1.5613, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 71.91%, Test  F1: 75.08%, 
Epoch: 199, Train Loss: 15.5899, Valid Loss: 1.7335, Test  Loss: 1.5651, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 71.48%, Test  F1: 75.24%, 
Epoch: 200, Train Loss: 15.5879, Valid Loss: 1.7366, Test  Loss: 1.5698, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.53%, Test  F1: 75.16%, 
Epoch: 201, Train Loss: 15.5964, Valid Loss: 1.7412, Test  Loss: 1.5733, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.93%, Test  F1: 75.00%, 
Epoch: 202, Train Loss: 15.6452, Valid Loss: 1.7434, Test  Loss: 1.5742, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.94%, Test  F1: 75.03%, 
Epoch: 203, Train Loss: 15.5939, Valid Loss: 1.7449, Test  Loss: 1.5764, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.61%, Test  F1: 75.05%, 
Epoch: 204, Train Loss: 15.6147, Valid Loss: 1.7374, Test  Loss: 1.5781, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.97%, Test  F1: 75.19%, 
Epoch: 205, Train Loss: 15.6000, Valid Loss: 1.7306, Test  Loss: 1.5809, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.97%, Test  F1: 75.19%, 
Epoch: 206, Train Loss: 15.6499, Valid Loss: 1.7260, Test  Loss: 1.5828, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.73%, Test  F1: 75.08%, 
Epoch: 207, Train Loss: 15.6028, Valid Loss: 1.7207, Test  Loss: 1.5823, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.90%, Test  F1: 75.00%, 
Epoch: 208, Train Loss: 15.5914, Valid Loss: 1.7161, Test  Loss: 1.5820, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.33%, Test  F1: 74.79%, 
Epoch: 209, Train Loss: 15.5845, Valid Loss: 1.7132, Test  Loss: 1.5843, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.63%, Test  F1: 75.07%, 
Epoch: 210, Train Loss: 15.6024, Valid Loss: 1.7160, Test  Loss: 1.5884, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.60%, Test  F1: 75.17%, 
Epoch: 211, Train Loss: 15.5612, Valid Loss: 1.7198, Test  Loss: 1.5935, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.48%, Test  F1: 75.12%, 
Epoch: 212, Train Loss: 15.5815, Valid Loss: 1.7246, Test  Loss: 1.5970, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.74%, Test  F1: 75.13%, 
Epoch: 213, Train Loss: 15.6054, Valid Loss: 1.7199, Test  Loss: 1.6012, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 72.33%, Test  F1: 74.64%, 
Epoch: 214, Train Loss: 15.6215, Valid Loss: 1.7105, Test  Loss: 1.6038, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.22%, Test  F1: 74.60%, 
Epoch: 215, Train Loss: 15.6094, Valid Loss: 1.7153, Test  Loss: 1.6072, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 71.45%, Test  F1: 74.26%, 
Epoch: 216, Train Loss: 15.5944, Valid Loss: 1.7236, Test  Loss: 1.6138, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.94%, Test  F1: 74.56%, 
Epoch: 217, Train Loss: 15.5911, Valid Loss: 1.7333, Test  Loss: 1.6227, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.94%, Test  F1: 74.74%, 
Epoch: 218, Train Loss: 15.6194, Valid Loss: 1.7440, Test  Loss: 1.6290, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.94%, Test  F1: 74.73%, 
Epoch: 219, Train Loss: 15.5805, Valid Loss: 1.7519, Test  Loss: 1.6308, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.69%, Test  F1: 74.74%, 
Epoch: 220, Train Loss: 15.6151, Valid Loss: 1.7562, Test  Loss: 1.6311, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.36%, Test  F1: 74.73%, 
Epoch: 221, Train Loss: 15.6095, Valid Loss: 1.7587, Test  Loss: 1.6287, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.36%, Test  F1: 74.51%, 
Epoch: 222, Train Loss: 15.5998, Valid Loss: 1.7624, Test  Loss: 1.6256, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.01%, Test  F1: 74.56%, 
Epoch: 223, Train Loss: 15.6200, Valid Loss: 1.7673, Test  Loss: 1.6240, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.44%, Test  F1: 74.48%, 
Epoch: 224, Train Loss: 15.5854, Valid Loss: 1.7763, Test  Loss: 1.6236, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.11%, Test  F1: 74.51%, 
Epoch: 225, Train Loss: 15.6041, Valid Loss: 1.7899, Test  Loss: 1.6251, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.23%, Test  F1: 74.50%, 
Epoch: 226, Train Loss: 15.5854, Valid Loss: 1.8017, Test  Loss: 1.6286, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 70.27%, Test  F1: 74.38%, 
Epoch: 227, Train Loss: 15.6054, Valid Loss: 1.8068, Test  Loss: 1.6311, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.93%, Test  F1: 74.50%, 
Epoch: 228, Train Loss: 15.5868, Valid Loss: 1.8145, Test  Loss: 1.6343, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 71.28%, Test  F1: 74.54%, 
Epoch: 229, Train Loss: 15.5875, Valid Loss: 1.8173, Test  Loss: 1.6356, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.28%, Test  F1: 74.43%, 
Epoch: 230, Train Loss: 15.5947, Valid Loss: 1.8173, Test  Loss: 1.6336, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 71.39%, Test  F1: 74.48%, 
Epoch: 231, Train Loss: 15.5931, Valid Loss: 1.8130, Test  Loss: 1.6298, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.39%, Test  F1: 74.55%, 
Epoch: 232, Train Loss: 15.5573, Valid Loss: 1.8039, Test  Loss: 1.6262, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.39%, Test  F1: 74.55%, 
Epoch: 233, Train Loss: 15.5789, Valid Loss: 1.7965, Test  Loss: 1.6237, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 71.79%, Test  F1: 74.30%, 
Epoch: 234, Train Loss: 15.5997, Valid Loss: 1.7895, Test  Loss: 1.6215, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.00%, Train F1: 100.00%, Valid F1: 71.59%, Test  F1: 74.05%, 
Epoch: 235, Train Loss: 15.5643, Valid Loss: 1.7865, Test  Loss: 1.6200, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.10%, Train F1: 100.00%, Valid F1: 71.98%, Test  F1: 74.17%, 
Epoch: 236, Train Loss: 15.5542, Valid Loss: 1.7874, Test  Loss: 1.6184, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 71.98%, Test  F1: 74.24%, 
Epoch: 237, Train Loss: 15.5836, Valid Loss: 1.7955, Test  Loss: 1.6171, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 71.98%, Test  F1: 74.14%, 
Epoch: 238, Train Loss: 15.5685, Valid Loss: 1.8041, Test  Loss: 1.6142, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 71.43%, Test  F1: 74.18%, 
Epoch: 239, Train Loss: 15.5501, Valid Loss: 1.8160, Test  Loss: 1.6115, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.43%, Test  F1: 74.40%, 
Epoch: 240, Train Loss: 15.6046, Valid Loss: 1.8259, Test  Loss: 1.6106, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.43%, Test  F1: 74.81%, 
Epoch: 241, Train Loss: 15.5995, Valid Loss: 1.8371, Test  Loss: 1.6131, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.54%, Test  F1: 74.93%, 
Epoch: 242, Train Loss: 15.6012, Valid Loss: 1.8514, Test  Loss: 1.6199, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.65%, Test  F1: 74.93%, 
Epoch: 243, Train Loss: 15.6103, Valid Loss: 1.8671, Test  Loss: 1.6273, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.49%, Test  F1: 74.85%, 
Epoch: 244, Train Loss: 15.5731, Valid Loss: 1.8801, Test  Loss: 1.6340, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.82%, Test  F1: 74.81%, 
Epoch: 245, Train Loss: 15.5992, Valid Loss: 1.8907, Test  Loss: 1.6359, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 74.82%, 
Epoch: 246, Train Loss: 15.5978, Valid Loss: 1.8934, Test  Loss: 1.6359, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 74.90%, 
Epoch: 247, Train Loss: 15.5753, Valid Loss: 1.8891, Test  Loss: 1.6354, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.66%, Test  F1: 74.75%, 
Epoch: 248, Train Loss: 15.6031, Valid Loss: 1.8820, Test  Loss: 1.6359, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.33%, Test  F1: 74.47%, 
Epoch: 249, Train Loss: 15.5663, Valid Loss: 1.8844, Test  Loss: 1.6424, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.88%, Test  F1: 74.20%, 
Epoch: 250, Train Loss: 15.5489, Valid Loss: 1.8905, Test  Loss: 1.6453, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.88%, Test  F1: 74.29%, 
Epoch: 251, Train Loss: 15.5927, Valid Loss: 1.8936, Test  Loss: 1.6421, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.93%, Test  F1: 74.34%, 
Epoch: 252, Train Loss: 15.5794, Valid Loss: 1.8951, Test  Loss: 1.6396, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.41%, Test  F1: 74.48%, 
Epoch: 253, Train Loss: 15.5829, Valid Loss: 1.8868, Test  Loss: 1.6373, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 71.72%, Test  F1: 74.69%, 
Epoch: 254, Train Loss: 15.5695, Valid Loss: 1.8765, Test  Loss: 1.6319, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.72%, Test  F1: 74.51%, 
Epoch: 255, Train Loss: 15.5776, Valid Loss: 1.8686, Test  Loss: 1.6270, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.97%, Test  F1: 74.60%, 
Epoch: 256, Train Loss: 15.5626, Valid Loss: 1.8624, Test  Loss: 1.6212, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.92%, Test  F1: 74.59%, 
Epoch: 257, Train Loss: 15.5843, Valid Loss: 1.8601, Test  Loss: 1.6161, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.47%, Test  F1: 74.58%, 
Epoch: 258, Train Loss: 15.5501, Valid Loss: 1.8592, Test  Loss: 1.6131, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.47%, Test  F1: 74.71%, 
Epoch: 259, Train Loss: 15.5801, Valid Loss: 1.8534, Test  Loss: 1.6098, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 71.39%, Test  F1: 74.52%, 
Epoch: 260, Train Loss: 15.5662, Valid Loss: 1.8466, Test  Loss: 1.6070, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.91%, Test  F1: 74.50%, 
Epoch: 261, Train Loss: 15.5749, Valid Loss: 1.8413, Test  Loss: 1.6044, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 70.91%, Test  F1: 74.49%, 
Epoch: 262, Train Loss: 15.5498, Valid Loss: 1.8343, Test  Loss: 1.6013, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.91%, Test  F1: 74.35%, 
Epoch: 263, Train Loss: 15.6017, Valid Loss: 1.8312, Test  Loss: 1.6004, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.67%, Test  F1: 74.29%, 
Epoch: 264, Train Loss: 15.5984, Valid Loss: 1.8269, Test  Loss: 1.6045, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.91%, Test  F1: 74.22%, 
Epoch: 265, Train Loss: 15.5657, Valid Loss: 1.8248, Test  Loss: 1.6094, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 70.91%, Test  F1: 74.42%, 
Epoch: 266, Train Loss: 15.5832, Valid Loss: 1.8229, Test  Loss: 1.6137, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.91%, Test  F1: 74.43%, 
Epoch: 267, Train Loss: 15.5403, Valid Loss: 1.8232, Test  Loss: 1.6181, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 69.70%, Test  F1: 74.54%, 
Epoch: 268, Train Loss: 15.5620, Valid Loss: 1.8216, Test  Loss: 1.6195, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 69.99%, Test  F1: 74.68%, 
Epoch: 269, Train Loss: 15.5647, Valid Loss: 1.8170, Test  Loss: 1.6194, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.62%, Test  F1: 74.61%, 
Epoch: 270, Train Loss: 15.5745, Valid Loss: 1.8100, Test  Loss: 1.6168, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.66%, Test  F1: 74.58%, 
Epoch: 271, Train Loss: 15.5476, Valid Loss: 1.8018, Test  Loss: 1.6141, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.66%, Test  F1: 74.46%, 
Epoch: 272, Train Loss: 15.5401, Valid Loss: 1.7922, Test  Loss: 1.6111, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 70.66%, Test  F1: 74.69%, 
Epoch: 273, Train Loss: 15.5587, Valid Loss: 1.7860, Test  Loss: 1.6087, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.20%, Test  F1: 74.56%, 
Epoch: 274, Train Loss: 15.5746, Valid Loss: 1.7798, Test  Loss: 1.6064, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 69.80%, Test  F1: 74.92%, 
Epoch: 275, Train Loss: 15.6153, Valid Loss: 1.7740, Test  Loss: 1.6051, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 70.45%, Test  F1: 74.89%, 
Epoch: 276, Train Loss: 15.5612, Valid Loss: 1.7665, Test  Loss: 1.6039, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 70.85%, Test  F1: 74.76%, 
Epoch: 277, Train Loss: 15.5935, Valid Loss: 1.7522, Test  Loss: 1.6040, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.15%, Test  F1: 74.49%, 
Epoch: 278, Train Loss: 15.5631, Valid Loss: 1.7410, Test  Loss: 1.6029, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.15%, Test  F1: 74.25%, 
Epoch: 279, Train Loss: 15.5606, Valid Loss: 1.7282, Test  Loss: 1.5994, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.33%, Test  F1: 74.38%, 
Epoch: 280, Train Loss: 15.5562, Valid Loss: 1.7147, Test  Loss: 1.5914, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 69.92%, Test  F1: 74.45%, 
Epoch: 281, Train Loss: 15.5565, Valid Loss: 1.7052, Test  Loss: 1.5879, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 69.94%, Test  F1: 74.60%, 
Epoch: 282, Train Loss: 15.5931, Valid Loss: 1.6981, Test  Loss: 1.5900, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.73%, Test  F1: 74.32%, 
Epoch: 283, Train Loss: 15.5679, Valid Loss: 1.7008, Test  Loss: 1.5942, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.73%, Test  F1: 74.19%, 
Epoch: 284, Train Loss: 15.5302, Valid Loss: 1.7048, Test  Loss: 1.5989, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 71.02%, Test  F1: 74.08%, 
Epoch: 285, Train Loss: 15.5598, Valid Loss: 1.7070, Test  Loss: 1.6053, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 71.50%, Test  F1: 74.09%, 
Epoch: 286, Train Loss: 15.5671, Valid Loss: 1.7058, Test  Loss: 1.6122, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 71.37%, Test  F1: 74.07%, 
Epoch: 287, Train Loss: 15.5518, Valid Loss: 1.7055, Test  Loss: 1.6193, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.10%, Train F1: 100.00%, Valid F1: 71.37%, Test  F1: 73.92%, 
Epoch: 288, Train Loss: 15.5437, Valid Loss: 1.7037, Test  Loss: 1.6234, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.10%, Train F1: 100.00%, Valid F1: 71.35%, Test  F1: 73.88%, 
Epoch: 289, Train Loss: 15.5502, Valid Loss: 1.6983, Test  Loss: 1.6247, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.05%, Train F1: 100.00%, Valid F1: 71.03%, Test  F1: 73.80%, 
Epoch: 290, Train Loss: 15.5624, Valid Loss: 1.6945, Test  Loss: 1.6250, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 71.03%, Test  F1: 73.99%, 
Epoch: 291, Train Loss: 15.5714, Valid Loss: 1.6903, Test  Loss: 1.6274, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 70.85%, Test  F1: 74.03%, 
Epoch: 292, Train Loss: 15.5490, Valid Loss: 1.6882, Test  Loss: 1.6301, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.85%, Test  F1: 74.10%, 
Epoch: 293, Train Loss: 15.5396, Valid Loss: 1.6885, Test  Loss: 1.6310, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.80%, Test  F1: 74.19%, 
Epoch: 294, Train Loss: 15.5512, Valid Loss: 1.6894, Test  Loss: 1.6298, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.54%, Test  F1: 74.42%, 
Epoch: 295, Train Loss: 15.5203, Valid Loss: 1.6889, Test  Loss: 1.6267, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.79%, Test  F1: 74.32%, 
Epoch: 296, Train Loss: 15.5386, Valid Loss: 1.6877, Test  Loss: 1.6238, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 72.19%, Test  F1: 74.31%, 
Epoch: 297, Train Loss: 15.5208, Valid Loss: 1.6901, Test  Loss: 1.6228, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 74.21%, 
Epoch: 298, Train Loss: 15.5484, Valid Loss: 1.6926, Test  Loss: 1.6217, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 74.06%, 
Epoch: 299, Train Loss: 15.4900, Valid Loss: 1.6990, Test  Loss: 1.6234, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 74.19%, 
Epoch: 300, Train Loss: 15.5633, Valid Loss: 1.7150, Test  Loss: 1.6313, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 74.28%, 
Epoch: 301, Train Loss: 15.5349, Valid Loss: 1.7250, Test  Loss: 1.6334, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 72.10%, Test  F1: 74.40%, 
Epoch: 302, Train Loss: 15.5257, Valid Loss: 1.7382, Test  Loss: 1.6343, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 72.17%, Test  F1: 74.52%, 
Epoch: 303, Train Loss: 15.5214, Valid Loss: 1.7565, Test  Loss: 1.6393, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.77%, Test  F1: 74.31%, 
Epoch: 304, Train Loss: 15.5418, Valid Loss: 1.7700, Test  Loss: 1.6420, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.69%, Test  F1: 74.37%, 
Epoch: 305, Train Loss: 15.5392, Valid Loss: 1.7708, Test  Loss: 1.6408, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 71.29%, Test  F1: 74.59%, 
Epoch: 306, Train Loss: 15.5197, Valid Loss: 1.7706, Test  Loss: 1.6406, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.22%, Test  F1: 74.47%, 
Epoch: 307, Train Loss: 15.5399, Valid Loss: 1.7677, Test  Loss: 1.6404, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 74.62%, 
Epoch: 308, Train Loss: 15.5540, Valid Loss: 1.7635, Test  Loss: 1.6408, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 74.50%, 
Epoch: 309, Train Loss: 15.5370, Valid Loss: 1.7615, Test  Loss: 1.6405, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 70.66%, Test  F1: 74.25%, 
Epoch: 310, Train Loss: 15.5209, Valid Loss: 1.7547, Test  Loss: 1.6409, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 70.58%, Test  F1: 74.21%, 
Epoch: 311, Train Loss: 15.5491, Valid Loss: 1.7473, Test  Loss: 1.6411, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.16%, Test  F1: 74.30%, 
Epoch: 312, Train Loss: 15.5612, Valid Loss: 1.7405, Test  Loss: 1.6409, Train Acc: 100.00%, Valid Acc: 73.06%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.01%, Test  F1: 74.29%, 
Epoch: 313, Train Loss: 15.5231, Valid Loss: 1.7344, Test  Loss: 1.6414, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 74.34%, 
Epoch: 314, Train Loss: 15.5346, Valid Loss: 1.7372, Test  Loss: 1.6438, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 74.32%, 
Epoch: 315, Train Loss: 15.5141, Valid Loss: 1.7398, Test  Loss: 1.6466, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 70.68%, Test  F1: 74.46%, 
Epoch: 316, Train Loss: 15.5322, Valid Loss: 1.7385, Test  Loss: 1.6458, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.77%, Test  F1: 74.59%, 
Epoch: 317, Train Loss: 15.5218, Valid Loss: 1.7350, Test  Loss: 1.6459, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.01%, Test  F1: 74.60%, 
Epoch: 318, Train Loss: 15.5456, Valid Loss: 1.7268, Test  Loss: 1.6438, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.26%, Test  F1: 74.48%, 
Epoch: 319, Train Loss: 15.5316, Valid Loss: 1.7189, Test  Loss: 1.6437, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 70.93%, Test  F1: 74.57%, 
Epoch: 320, Train Loss: 15.5497, Valid Loss: 1.7128, Test  Loss: 1.6451, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.58%, Test  F1: 74.52%, 
Epoch: 321, Train Loss: 15.5371, Valid Loss: 1.7086, Test  Loss: 1.6472, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 71.58%, Test  F1: 74.37%, 
Epoch: 322, Train Loss: 15.4963, Valid Loss: 1.7048, Test  Loss: 1.6504, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.80%, Test  F1: 74.29%, 
Epoch: 323, Train Loss: 15.5361, Valid Loss: 1.7039, Test  Loss: 1.6521, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 72.17%, Test  F1: 74.22%, 
Epoch: 324, Train Loss: 15.5312, Valid Loss: 1.7031, Test  Loss: 1.6520, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 72.19%, Test  F1: 74.20%, 
Epoch: 325, Train Loss: 15.5214, Valid Loss: 1.6983, Test  Loss: 1.6516, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 72.99%, Test  F1: 74.37%, 
Epoch: 326, Train Loss: 15.4991, Valid Loss: 1.6935, Test  Loss: 1.6514, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 73.40%, Test  F1: 74.36%, 
Epoch: 327, Train Loss: 15.5399, Valid Loss: 1.6896, Test  Loss: 1.6512, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 73.15%, Test  F1: 74.43%, 
Epoch: 328, Train Loss: 15.5513, Valid Loss: 1.6776, Test  Loss: 1.6480, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 73.45%, Test  F1: 74.63%, 
Epoch: 329, Train Loss: 15.5315, Valid Loss: 1.6626, Test  Loss: 1.6419, Train Acc: 100.00%, Valid Acc: 76.75%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 73.72%, Test  F1: 74.73%, 
Epoch: 330, Train Loss: 15.5125, Valid Loss: 1.6504, Test  Loss: 1.6387, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 73.32%, Test  F1: 74.48%, 
Epoch: 331, Train Loss: 15.5296, Valid Loss: 1.6403, Test  Loss: 1.6361, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 72.82%, Test  F1: 74.44%, 
Epoch: 332, Train Loss: 15.5280, Valid Loss: 1.6381, Test  Loss: 1.6292, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 73.43%, Test  F1: 74.90%, 
Epoch: 333, Train Loss: 15.5393, Valid Loss: 1.6424, Test  Loss: 1.6267, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 73.02%, Test  F1: 75.00%, 
Epoch: 334, Train Loss: 15.5197, Valid Loss: 1.6449, Test  Loss: 1.6262, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 73.04%, Test  F1: 74.71%, 
Epoch: 335, Train Loss: 15.5100, Valid Loss: 1.6460, Test  Loss: 1.6260, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 73.25%, Test  F1: 74.73%, 
Epoch: 336, Train Loss: 15.5307, Valid Loss: 1.6502, Test  Loss: 1.6222, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 73.61%, Test  F1: 74.71%, 
Epoch: 337, Train Loss: 15.5061, Valid Loss: 1.6549, Test  Loss: 1.6208, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.69%, Test  F1: 74.67%, 
Epoch: 338, Train Loss: 15.5001, Valid Loss: 1.6585, Test  Loss: 1.6195, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.68%, Test  F1: 74.67%, 
Epoch: 339, Train Loss: 15.5268, Valid Loss: 1.6601, Test  Loss: 1.6183, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 72.67%, Test  F1: 74.51%, 
Epoch: 340, Train Loss: 15.5397, Valid Loss: 1.6612, Test  Loss: 1.6207, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 72.67%, Test  F1: 74.56%, 
Epoch: 341, Train Loss: 15.5205, Valid Loss: 1.6621, Test  Loss: 1.6240, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 72.73%, Test  F1: 74.55%, 
Epoch: 342, Train Loss: 15.5503, Valid Loss: 1.6552, Test  Loss: 1.6255, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.45%, Test  F1: 74.66%, 
Epoch: 343, Train Loss: 15.5165, Valid Loss: 1.6504, Test  Loss: 1.6288, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.45%, Test  F1: 74.68%, 
Epoch: 344, Train Loss: 15.5033, Valid Loss: 1.6485, Test  Loss: 1.6329, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.30%, Test  F1: 74.67%, 
Epoch: 345, Train Loss: 15.5346, Valid Loss: 1.6465, Test  Loss: 1.6370, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.74%, Test  F1: 74.87%, 
Epoch: 346, Train Loss: 15.5202, Valid Loss: 1.6440, Test  Loss: 1.6402, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.94%, Test  F1: 74.76%, 
Epoch: 347, Train Loss: 15.5338, Valid Loss: 1.6342, Test  Loss: 1.6441, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.94%, Test  F1: 74.65%, 
Epoch: 348, Train Loss: 15.5113, Valid Loss: 1.6260, Test  Loss: 1.6471, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 73.36%, Test  F1: 74.62%, 
Epoch: 349, Train Loss: 15.5298, Valid Loss: 1.6214, Test  Loss: 1.6517, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 73.31%, Test  F1: 74.63%, 
Epoch: 350, Train Loss: 15.5221, Valid Loss: 1.6127, Test  Loss: 1.6482, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 73.31%, Test  F1: 74.36%, 
Epoch: 351, Train Loss: 15.5250, Valid Loss: 1.6074, Test  Loss: 1.6446, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 72.97%, Test  F1: 74.32%, 
Epoch: 352, Train Loss: 15.5044, Valid Loss: 1.6033, Test  Loss: 1.6377, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.82%, Test  F1: 74.58%, 
Epoch: 353, Train Loss: 15.5099, Valid Loss: 1.6014, Test  Loss: 1.6286, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.07%, Test  F1: 74.65%, 
Epoch: 354, Train Loss: 15.4921, Valid Loss: 1.6112, Test  Loss: 1.6193, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 72.97%, Test  F1: 74.69%, 
Epoch: 355, Train Loss: 15.5403, Valid Loss: 1.6278, Test  Loss: 1.6151, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 72.09%, Test  F1: 75.05%, 
Epoch: 356, Train Loss: 15.5538, Valid Loss: 1.6502, Test  Loss: 1.6179, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 72.31%, Test  F1: 75.21%, 
Epoch: 357, Train Loss: 15.5303, Valid Loss: 1.6776, Test  Loss: 1.6260, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 72.91%, Test  F1: 75.16%, 
Epoch: 358, Train Loss: 15.5055, Valid Loss: 1.7048, Test  Loss: 1.6335, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.91%, Test  F1: 75.19%, 
Epoch: 359, Train Loss: 15.4894, Valid Loss: 1.7268, Test  Loss: 1.6400, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 75.21%, 
Epoch: 360, Train Loss: 15.4930, Valid Loss: 1.7419, Test  Loss: 1.6458, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.20%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 75.31%, 
Epoch: 361, Train Loss: 15.4885, Valid Loss: 1.7517, Test  Loss: 1.6504, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 75.15%, 
Epoch: 362, Train Loss: 15.5118, Valid Loss: 1.7525, Test  Loss: 1.6553, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 72.04%, Test  F1: 75.09%, 
Epoch: 363, Train Loss: 15.5401, Valid Loss: 1.7433, Test  Loss: 1.6575, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.21%, Test  F1: 75.01%, 
Epoch: 364, Train Loss: 15.5169, Valid Loss: 1.7297, Test  Loss: 1.6586, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.25%, Train F1: 100.00%, Valid F1: 72.12%, Test  F1: 75.48%, 
Epoch: 365, Train Loss: 15.5336, Valid Loss: 1.7240, Test  Loss: 1.6653, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.14%, Test  F1: 75.02%, 
Epoch: 366, Train Loss: 15.4888, Valid Loss: 1.7202, Test  Loss: 1.6706, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.01%, Test  F1: 74.66%, 
Epoch: 367, Train Loss: 15.4999, Valid Loss: 1.7210, Test  Loss: 1.6761, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.16%, Test  F1: 74.45%, 
Epoch: 368, Train Loss: 15.4906, Valid Loss: 1.7203, Test  Loss: 1.6789, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.09%, Test  F1: 74.47%, 
Epoch: 369, Train Loss: 15.5135, Valid Loss: 1.7210, Test  Loss: 1.6815, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.74%, Test  F1: 74.30%, 
Epoch: 370, Train Loss: 15.4925, Valid Loss: 1.7160, Test  Loss: 1.6830, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.74%, Test  F1: 74.29%, 
Epoch: 371, Train Loss: 15.5492, Valid Loss: 1.7025, Test  Loss: 1.6844, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 70.68%, Test  F1: 74.41%, 
Epoch: 372, Train Loss: 15.5110, Valid Loss: 1.6951, Test  Loss: 1.6910, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 70.68%, Test  F1: 74.49%, 
Epoch: 373, Train Loss: 15.5051, Valid Loss: 1.6889, Test  Loss: 1.6992, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 74.33%, 
Epoch: 374, Train Loss: 15.4858, Valid Loss: 1.6769, Test  Loss: 1.7037, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.48%, Test  F1: 74.30%, 
Epoch: 375, Train Loss: 15.4918, Valid Loss: 1.6721, Test  Loss: 1.7121, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.49%, Test  F1: 74.38%, 
Epoch: 376, Train Loss: 15.4921, Valid Loss: 1.6688, Test  Loss: 1.7212, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.49%, Test  F1: 74.33%, 
Epoch: 377, Train Loss: 15.5047, Valid Loss: 1.6607, Test  Loss: 1.7249, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 70.77%, Test  F1: 74.50%, 
Epoch: 378, Train Loss: 15.5293, Valid Loss: 1.6492, Test  Loss: 1.7202, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.09%, Test  F1: 74.69%, 
Epoch: 379, Train Loss: 15.5093, Valid Loss: 1.6403, Test  Loss: 1.7136, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.60%, Test  F1: 74.70%, 
Epoch: 380, Train Loss: 15.5284, Valid Loss: 1.6324, Test  Loss: 1.7074, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 71.05%, Test  F1: 74.50%, 
Epoch: 381, Train Loss: 15.5398, Valid Loss: 1.6291, Test  Loss: 1.7026, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.33%, Train F1: 100.00%, Valid F1: 71.45%, Test  F1: 74.36%, 
Epoch: 382, Train Loss: 15.4880, Valid Loss: 1.6286, Test  Loss: 1.6979, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 71.90%, Test  F1: 74.19%, 
Epoch: 383, Train Loss: 15.4890, Valid Loss: 1.6248, Test  Loss: 1.6920, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.19%, Train F1: 100.00%, Valid F1: 72.32%, Test  F1: 74.22%, 
Epoch: 384, Train Loss: 15.5144, Valid Loss: 1.6220, Test  Loss: 1.6839, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 72.26%, Test  F1: 74.43%, 
Epoch: 385, Train Loss: 15.5092, Valid Loss: 1.6210, Test  Loss: 1.6780, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.36%, Test  F1: 74.78%, 
Epoch: 386, Train Loss: 15.4835, Valid Loss: 1.6205, Test  Loss: 1.6744, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.69%, Test  F1: 74.92%, 
Epoch: 387, Train Loss: 15.5349, Valid Loss: 1.6249, Test  Loss: 1.6739, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 72.75%, Test  F1: 74.82%, 
Epoch: 388, Train Loss: 15.4838, Valid Loss: 1.6238, Test  Loss: 1.6723, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 73.03%, Test  F1: 74.67%, 
Epoch: 389, Train Loss: 15.4996, Valid Loss: 1.6239, Test  Loss: 1.6763, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.83%, Test  F1: 74.65%, 
Epoch: 390, Train Loss: 15.5226, Valid Loss: 1.6266, Test  Loss: 1.6801, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.36%, Test  F1: 74.65%, 
Epoch: 391, Train Loss: 15.4901, Valid Loss: 1.6340, Test  Loss: 1.6872, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.36%, Test  F1: 74.64%, 
Epoch: 392, Train Loss: 15.5107, Valid Loss: 1.6446, Test  Loss: 1.6970, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.35%, Test  F1: 74.65%, 
Epoch: 393, Train Loss: 15.4918, Valid Loss: 1.6567, Test  Loss: 1.7064, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.70%, Train F1: 100.00%, Valid F1: 71.61%, Test  F1: 74.60%, 
Epoch: 394, Train Loss: 15.4983, Valid Loss: 1.6691, Test  Loss: 1.7119, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 71.70%, Test  F1: 74.22%, 
Epoch: 395, Train Loss: 15.4893, Valid Loss: 1.6887, Test  Loss: 1.7185, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.30%, Test  F1: 74.36%, 
Epoch: 396, Train Loss: 15.5368, Valid Loss: 1.7060, Test  Loss: 1.7241, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 71.45%, Test  F1: 73.99%, 
Epoch: 397, Train Loss: 15.4833, Valid Loss: 1.7206, Test  Loss: 1.7273, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.00%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 73.75%, 
Epoch: 398, Train Loss: 15.5397, Valid Loss: 1.7298, Test  Loss: 1.7249, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 75.73%, Train F1: 100.00%, Valid F1: 70.42%, Test  F1: 73.53%, 
Epoch: 399, Train Loss: 15.4724, Valid Loss: 1.7327, Test  Loss: 1.7177, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 75.96%, Train F1: 100.00%, Valid F1: 70.42%, Test  F1: 73.74%, 
Epoch: 400, Train Loss: 15.4807, Valid Loss: 1.7246, Test  Loss: 1.7077, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.14%, Train F1: 100.00%, Valid F1: 70.32%, Test  F1: 74.00%, 
Epoch: 401, Train Loss: 15.5252, Valid Loss: 1.7121, Test  Loss: 1.6942, Train Acc: 100.00%, Valid Acc: 73.43%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.03%, Test  F1: 74.32%, 
Epoch: 402, Train Loss: 15.5156, Valid Loss: 1.7000, Test  Loss: 1.6816, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 70.47%, Test  F1: 74.46%, 
Epoch: 403, Train Loss: 15.4880, Valid Loss: 1.6928, Test  Loss: 1.6722, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.42%, Train F1: 100.00%, Valid F1: 70.93%, Test  F1: 74.30%, 
Epoch: 404, Train Loss: 15.5082, Valid Loss: 1.6883, Test  Loss: 1.6666, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 70.85%, Test  F1: 74.50%, 
Epoch: 405, Train Loss: 15.4686, Valid Loss: 1.6889, Test  Loss: 1.6619, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.17%, Test  F1: 74.64%, 
Epoch: 406, Train Loss: 15.4893, Valid Loss: 1.6916, Test  Loss: 1.6581, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 71.59%, Test  F1: 74.62%, 
Epoch: 407, Train Loss: 15.5110, Valid Loss: 1.6942, Test  Loss: 1.6579, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.00%, Test  F1: 74.77%, 
Epoch: 408, Train Loss: 15.5025, Valid Loss: 1.6982, Test  Loss: 1.6595, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.00%, Test  F1: 74.82%, 
Epoch: 409, Train Loss: 15.4786, Valid Loss: 1.7020, Test  Loss: 1.6657, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 72.23%, Test  F1: 74.99%, 
Epoch: 410, Train Loss: 15.4928, Valid Loss: 1.7057, Test  Loss: 1.6699, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 72.65%, Test  F1: 74.88%, 
Epoch: 411, Train Loss: 15.5177, Valid Loss: 1.7083, Test  Loss: 1.6723, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.62%, Test  F1: 74.75%, 
Epoch: 412, Train Loss: 15.4909, Valid Loss: 1.7068, Test  Loss: 1.6723, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 72.97%, Test  F1: 74.93%, 
Epoch: 413, Train Loss: 15.4787, Valid Loss: 1.7011, Test  Loss: 1.6719, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.08%, Test  F1: 74.73%, 
Epoch: 414, Train Loss: 15.4954, Valid Loss: 1.6947, Test  Loss: 1.6698, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.08%, Test  F1: 74.86%, 
Epoch: 415, Train Loss: 15.5353, Valid Loss: 1.6854, Test  Loss: 1.6659, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.08%, Test  F1: 74.80%, 
Epoch: 416, Train Loss: 15.5016, Valid Loss: 1.6773, Test  Loss: 1.6647, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 71.75%, Test  F1: 74.96%, 
Epoch: 417, Train Loss: 15.4973, Valid Loss: 1.6707, Test  Loss: 1.6648, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.69%, Test  F1: 74.93%, 
Epoch: 418, Train Loss: 15.5069, Valid Loss: 1.6694, Test  Loss: 1.6636, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.39%, Test  F1: 74.89%, 
Epoch: 419, Train Loss: 15.5436, Valid Loss: 1.6753, Test  Loss: 1.6628, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 74.92%, 
Epoch: 420, Train Loss: 15.5026, Valid Loss: 1.6873, Test  Loss: 1.6658, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.02%, Train F1: 100.00%, Valid F1: 71.38%, Test  F1: 74.99%, 
Epoch: 421, Train Loss: 15.4915, Valid Loss: 1.7015, Test  Loss: 1.6671, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.85%, Test  F1: 74.81%, 
Epoch: 422, Train Loss: 15.4727, Valid Loss: 1.7187, Test  Loss: 1.6705, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.51%, Test  F1: 74.80%, 
Epoch: 423, Train Loss: 15.5011, Valid Loss: 1.7368, Test  Loss: 1.6754, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 72.33%, Test  F1: 74.97%, 
Epoch: 424, Train Loss: 15.4941, Valid Loss: 1.7490, Test  Loss: 1.6777, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 72.79%, Test  F1: 74.97%, 
Epoch: 425, Train Loss: 15.4679, Valid Loss: 1.7645, Test  Loss: 1.6786, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.40%, Test  F1: 75.04%, 
Epoch: 426, Train Loss: 15.5033, Valid Loss: 1.7761, Test  Loss: 1.6788, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 71.72%, Test  F1: 75.36%, 
Epoch: 427, Train Loss: 15.4685, Valid Loss: 1.7794, Test  Loss: 1.6777, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 75.38%, 
Epoch: 428, Train Loss: 15.5034, Valid Loss: 1.7816, Test  Loss: 1.6736, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 75.42%, 
Epoch: 429, Train Loss: 15.4469, Valid Loss: 1.7815, Test  Loss: 1.6692, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.39%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 75.42%, 
Epoch: 430, Train Loss: 15.4758, Valid Loss: 1.7762, Test  Loss: 1.6659, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.34%, Train F1: 100.00%, Valid F1: 72.60%, Test  F1: 75.35%, 
Epoch: 431, Train Loss: 15.4786, Valid Loss: 1.7710, Test  Loss: 1.6659, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 75.37%, 
Epoch: 432, Train Loss: 15.4905, Valid Loss: 1.7683, Test  Loss: 1.6697, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 75.14%, 
Epoch: 433, Train Loss: 15.4571, Valid Loss: 1.7648, Test  Loss: 1.6754, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.57%, Test  F1: 74.98%, 
Epoch: 434, Train Loss: 15.4544, Valid Loss: 1.7643, Test  Loss: 1.6823, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.57%, Test  F1: 74.77%, 
Epoch: 435, Train Loss: 15.4897, Valid Loss: 1.7646, Test  Loss: 1.6891, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 71.41%, Test  F1: 74.48%, 
Epoch: 436, Train Loss: 15.4988, Valid Loss: 1.7662, Test  Loss: 1.6966, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.23%, Train F1: 100.00%, Valid F1: 70.83%, Test  F1: 74.22%, 
Epoch: 437, Train Loss: 15.4732, Valid Loss: 1.7681, Test  Loss: 1.7038, Train Acc: 100.00%, Valid Acc: 74.17%, Test  Acc: 76.37%, Train F1: 100.00%, Valid F1: 70.93%, Test  F1: 74.34%, 
Epoch: 438, Train Loss: 15.4541, Valid Loss: 1.7688, Test  Loss: 1.7089, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.28%, Train F1: 100.00%, Valid F1: 71.17%, Test  F1: 74.27%, 
Epoch: 439, Train Loss: 15.4764, Valid Loss: 1.7660, Test  Loss: 1.7108, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.17%, Test  F1: 74.47%, 
Epoch: 440, Train Loss: 15.4804, Valid Loss: 1.7621, Test  Loss: 1.7099, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.11%, Test  F1: 74.62%, 
Epoch: 441, Train Loss: 15.5220, Valid Loss: 1.7575, Test  Loss: 1.7112, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.01%, Test  F1: 74.61%, 
Epoch: 442, Train Loss: 15.4669, Valid Loss: 1.7567, Test  Loss: 1.7162, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.47%, Train F1: 100.00%, Valid F1: 71.89%, Test  F1: 74.44%, 
Epoch: 443, Train Loss: 15.4701, Valid Loss: 1.7567, Test  Loss: 1.7173, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 71.74%, Test  F1: 74.56%, 
Epoch: 444, Train Loss: 15.4832, Valid Loss: 1.7561, Test  Loss: 1.7178, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 72.09%, Test  F1: 74.51%, 
Epoch: 445, Train Loss: 15.4826, Valid Loss: 1.7527, Test  Loss: 1.7166, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 71.81%, Test  F1: 74.58%, 
Epoch: 446, Train Loss: 15.4604, Valid Loss: 1.7509, Test  Loss: 1.7170, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.56%, Train F1: 100.00%, Valid F1: 71.81%, Test  F1: 74.51%, 
Epoch: 447, Train Loss: 15.4747, Valid Loss: 1.7460, Test  Loss: 1.7155, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.24%, Test  F1: 74.72%, 
Epoch: 448, Train Loss: 15.4632, Valid Loss: 1.7303, Test  Loss: 1.7082, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 72.61%, Test  F1: 74.78%, 
Epoch: 449, Train Loss: 15.4903, Valid Loss: 1.7151, Test  Loss: 1.7028, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 74.79%, 
Epoch: 450, Train Loss: 15.4742, Valid Loss: 1.7101, Test  Loss: 1.7010, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 72.31%, Test  F1: 74.60%, 
Epoch: 451, Train Loss: 15.4810, Valid Loss: 1.7149, Test  Loss: 1.7022, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 72.31%, Test  F1: 74.65%, 
Epoch: 452, Train Loss: 15.4632, Valid Loss: 1.7183, Test  Loss: 1.7015, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.65%, Train F1: 100.00%, Valid F1: 72.15%, Test  F1: 74.44%, 
Epoch: 453, Train Loss: 15.4798, Valid Loss: 1.7220, Test  Loss: 1.7004, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.22%, Test  F1: 74.54%, 
Epoch: 454, Train Loss: 15.4960, Valid Loss: 1.7250, Test  Loss: 1.7048, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.60%, Train F1: 100.00%, Valid F1: 72.20%, Test  F1: 74.39%, 
Epoch: 455, Train Loss: 15.4702, Valid Loss: 1.7300, Test  Loss: 1.7089, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.51%, Train F1: 100.00%, Valid F1: 72.47%, Test  F1: 74.31%, 
Epoch: 456, Train Loss: 15.4880, Valid Loss: 1.7154, Test  Loss: 1.7092, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.74%, Train F1: 100.00%, Valid F1: 72.87%, Test  F1: 74.51%, 
Epoch: 457, Train Loss: 15.4879, Valid Loss: 1.7065, Test  Loss: 1.7105, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 73.42%, Test  F1: 74.44%, 
Epoch: 458, Train Loss: 15.4828, Valid Loss: 1.7035, Test  Loss: 1.7140, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.66%, Test  F1: 74.63%, 
Epoch: 459, Train Loss: 15.4654, Valid Loss: 1.7027, Test  Loss: 1.7153, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 73.07%, Test  F1: 74.67%, 
Epoch: 460, Train Loss: 15.5063, Valid Loss: 1.6957, Test  Loss: 1.7160, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 72.86%, Test  F1: 74.59%, 
Epoch: 461, Train Loss: 15.4748, Valid Loss: 1.6892, Test  Loss: 1.7169, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 72.99%, Test  F1: 74.41%, 
Epoch: 462, Train Loss: 15.4923, Valid Loss: 1.6796, Test  Loss: 1.7191, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.83%, Train F1: 100.00%, Valid F1: 72.49%, Test  F1: 74.55%, 
Epoch: 463, Train Loss: 15.4429, Valid Loss: 1.6715, Test  Loss: 1.7209, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.49%, Test  F1: 74.77%, 
Epoch: 464, Train Loss: 15.4613, Valid Loss: 1.6702, Test  Loss: 1.7228, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 72.88%, Test  F1: 74.86%, 
Epoch: 465, Train Loss: 15.4665, Valid Loss: 1.6698, Test  Loss: 1.7184, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 72.13%, Test  F1: 74.67%, 
Epoch: 466, Train Loss: 15.5144, Valid Loss: 1.6893, Test  Loss: 1.7157, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 72.34%, Test  F1: 74.86%, 
Epoch: 467, Train Loss: 15.4847, Valid Loss: 1.7161, Test  Loss: 1.7204, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.30%, Train F1: 100.00%, Valid F1: 72.36%, Test  F1: 75.01%, 
Epoch: 468, Train Loss: 15.4560, Valid Loss: 1.7556, Test  Loss: 1.7309, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.13%, Test  F1: 75.02%, 
Epoch: 469, Train Loss: 15.4532, Valid Loss: 1.8012, Test  Loss: 1.7466, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.42%, Test  F1: 75.01%, 
Epoch: 470, Train Loss: 15.4492, Valid Loss: 1.8439, Test  Loss: 1.7637, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 71.21%, Test  F1: 74.96%, 
Epoch: 471, Train Loss: 15.4509, Valid Loss: 1.8756, Test  Loss: 1.7772, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.93%, Train F1: 100.00%, Valid F1: 71.15%, Test  F1: 74.70%, 
Epoch: 472, Train Loss: 15.4717, Valid Loss: 1.9009, Test  Loss: 1.7891, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 71.00%, Test  F1: 74.65%, 
Epoch: 473, Train Loss: 15.4866, Valid Loss: 1.9185, Test  Loss: 1.7991, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.00%, Test  F1: 74.71%, 
Epoch: 474, Train Loss: 15.4607, Valid Loss: 1.9250, Test  Loss: 1.8045, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 76.88%, Train F1: 100.00%, Valid F1: 70.83%, Test  F1: 74.63%, 
Epoch: 475, Train Loss: 15.4894, Valid Loss: 1.9115, Test  Loss: 1.7966, Train Acc: 100.00%, Valid Acc: 73.80%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 70.15%, Test  F1: 74.92%, 
Epoch: 476, Train Loss: 15.4719, Valid Loss: 1.8742, Test  Loss: 1.7770, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.39%, Train F1: 100.00%, Valid F1: 70.89%, Test  F1: 75.18%, 
Epoch: 477, Train Loss: 15.4833, Valid Loss: 1.8380, Test  Loss: 1.7642, Train Acc: 100.00%, Valid Acc: 74.54%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 70.90%, Test  F1: 75.22%, 
Epoch: 478, Train Loss: 15.4519, Valid Loss: 1.8115, Test  Loss: 1.7595, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.27%, Test  F1: 74.95%, 
Epoch: 479, Train Loss: 15.4765, Valid Loss: 1.7902, Test  Loss: 1.7563, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.27%, Test  F1: 74.95%, 
Epoch: 480, Train Loss: 15.4819, Valid Loss: 1.7730, Test  Loss: 1.7515, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.24%, Test  F1: 74.78%, 
Epoch: 481, Train Loss: 15.4569, Valid Loss: 1.7592, Test  Loss: 1.7475, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.25%, Test  F1: 74.53%, 
Epoch: 482, Train Loss: 15.4942, Valid Loss: 1.7454, Test  Loss: 1.7447, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 76.79%, Train F1: 100.00%, Valid F1: 71.29%, Test  F1: 74.55%, 
Epoch: 483, Train Loss: 15.4815, Valid Loss: 1.7343, Test  Loss: 1.7382, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.70%, Test  F1: 74.76%, 
Epoch: 484, Train Loss: 15.4694, Valid Loss: 1.7288, Test  Loss: 1.7333, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.07%, Train F1: 100.00%, Valid F1: 71.43%, Test  F1: 74.82%, 
Epoch: 485, Train Loss: 15.4479, Valid Loss: 1.7126, Test  Loss: 1.7236, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 72.32%, Test  F1: 74.90%, 
Epoch: 486, Train Loss: 15.4578, Valid Loss: 1.7006, Test  Loss: 1.7147, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 77.20%, Train F1: 100.00%, Valid F1: 72.58%, Test  F1: 74.95%, 
Epoch: 487, Train Loss: 15.4591, Valid Loss: 1.6906, Test  Loss: 1.7095, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.09%, Test  F1: 74.82%, 
Epoch: 488, Train Loss: 15.4364, Valid Loss: 1.6828, Test  Loss: 1.7069, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.15%, Test  F1: 74.78%, 
Epoch: 489, Train Loss: 15.4897, Valid Loss: 1.6770, Test  Loss: 1.7046, Train Acc: 100.00%, Valid Acc: 74.91%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 71.13%, Test  F1: 75.16%, 
Epoch: 490, Train Loss: 15.4943, Valid Loss: 1.6730, Test  Loss: 1.7051, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.16%, Train F1: 100.00%, Valid F1: 71.79%, Test  F1: 74.94%, 
Epoch: 491, Train Loss: 15.4540, Valid Loss: 1.6745, Test  Loss: 1.7075, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.27%, Test  F1: 74.86%, 
Epoch: 492, Train Loss: 15.4391, Valid Loss: 1.6787, Test  Loss: 1.7111, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 76.97%, Train F1: 100.00%, Valid F1: 71.90%, Test  F1: 74.75%, 
Epoch: 493, Train Loss: 15.4885, Valid Loss: 1.6847, Test  Loss: 1.7135, Train Acc: 100.00%, Valid Acc: 75.65%, Test  Acc: 77.11%, Train F1: 100.00%, Valid F1: 72.17%, Test  F1: 74.90%, 
Epoch: 494, Train Loss: 15.4422, Valid Loss: 1.6912, Test  Loss: 1.7129, Train Acc: 100.00%, Valid Acc: 75.28%, Test  Acc: 77.20%, Train F1: 100.00%, Valid F1: 71.89%, Test  F1: 75.00%, 
Epoch: 495, Train Loss: 15.4540, Valid Loss: 1.7008, Test  Loss: 1.7150, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 77.39%, Train F1: 100.00%, Valid F1: 72.43%, Test  F1: 75.18%, 
Epoch: 496, Train Loss: 15.4422, Valid Loss: 1.7092, Test  Loss: 1.7171, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 77.39%, Train F1: 100.00%, Valid F1: 72.36%, Test  F1: 75.18%, 
Epoch: 497, Train Loss: 15.4714, Valid Loss: 1.7165, Test  Loss: 1.7190, Train Acc: 100.00%, Valid Acc: 76.38%, Test  Acc: 77.25%, Train F1: 100.00%, Valid F1: 72.86%, Test  F1: 75.05%, 
Epoch: 498, Train Loss: 15.4483, Valid Loss: 1.7225, Test  Loss: 1.7199, Train Acc: 100.00%, Valid Acc: 76.75%, Test  Acc: 77.30%, Train F1: 100.00%, Valid F1: 73.34%, Test  F1: 75.04%, 
Epoch: 499, Train Loss: 15.4604, Valid Loss: 1.7243, Test  Loss: 1.7199, Train Acc: 100.00%, Valid Acc: 76.01%, Test  Acc: 77.43%, Train F1: 100.00%, Valid F1: 72.82%, Test  F1: 75.17%, 
Run 01:
Highest ACC Train: 100.00
Highest ACC Valid: 76.75
Highest ACC Test: 77.43
Highest F1 Train: 100.00
Highest F1 Valid: 73.72
Highest F1 Test: 75.48
  Final Train ACC: 100.00
   Final Val ACC: 76.75
   Final Test ACC: 76.70
  Final Train F1: 100.00
   Final Val F1: 73.72
   Final Test F1: 74.73
The running time is 1.9476739048957825

```
