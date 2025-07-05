# Artificial Neural Networks and Deep Learning Challenges [Edition 2023],[Edition 2024]


<b>Authors:</b> 

[Edition 2024] : [Kamil Hanna](https://github.com/KamilHanna), [Enrico Tirri](https://github.com/EnricoTirri),[Giorgio Negro](https://github.com/giorgionegro), 

[Edition 2023] : [Kamil Hanna](https://github.com/KamilHanna), [Filippo Desantis](), [Livia Giacomin]()

Note that additional information about the models are present in the report pdf files ,in each directory.
Moreover, please note that some of the competition links might be down/unaccessible to users without Politecnico di Milano credentials.

### Challenge 1 [Edition 2024] :
Competition available on [Codabench](https://www.codabench.org/competitions/4430/).


### Challenge 2 [Edition 2024] :
Competition available on [Kaggle](https://www.kaggle.com/t/af80f36772144dbb8b6179fea6180574 ).

### Challenge 1 [Edition 2023] :

Competition available on [Codalab](https://codalab.lisn.upsaclay.fr/competitions/16245).

# Leaf Classification

This project addresses a binary classification task: identifying whether a leaf in an image is **healthy** or **unhealthy**. The dataset consists of **5200 RGB images** of size **96x96**, each labeled accordingly. 

After cleaning the data—removing 98 duplicate outliers—and balancing the classes using **SMOTE**, we trained a deep learning model using **transfer learning**. The final architecture is based on **ConvNeXt Large**, with the first 90 layers frozen, and incorporates **batch normalization**, **dropout (0.1)**, and a **cyclical learning rate**. The best model achieved a **validation accuracy of 0.81**.

All code is located in the `Challenge1_2023` folder, along with a detailed report outlining the full pipeline and additional architecture experiments.


### Challenge 2 [Edition 2023] :
Competition available on [Codalab](https://codalab.lisn.upsaclay.fr/competitions/16514).










