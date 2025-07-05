# Artificial Neural Networks and Deep Learning Challenges [Edition 2023],[Edition 2024]


<b>Authors:</b> 

[Edition 2024] : [Kamil Hanna](https://github.com/KamilHanna), [Enrico Tirri](https://github.com/EnricoTirri),[Giorgio Negro](https://github.com/giorgionegro), 

[Edition 2023] : [Kamil Hanna](https://github.com/KamilHanna), [Filippo Desantis](), [Livia Giacomin]()

Note that additional information about the models are present in the report pdf files ,in each directory.
Moreover, please note that some of the competition links might be down/unaccessible to users without Politecnico di Milano credentials.

## Challenge 1 [Edition 2024] :  Blood Cell Classification

Competition available on [Codabench](https://www.codabench.org/competitions/4430/).

This challenge focuses on building a Convolutional Neural Network (CNN) classifier to categorize **96x96 RGB images of blood cells** into **one of eight predefined classes**, each representing a specific cell state. The cleaned dataset comprises **11,959 labeled images** after removing significant outliers and handling class imbalance.

The final solution is based on **ConvNeXtBase**, pre-trained on ImageNet, combined with a custom classification head including **SELU-activated dense layers**, **dropout**, and **L1L2 regularization**. We applied an extensive **data augmentation pipeline**, including **MixUp**, **CutMix**, **RandAugment**, and **channel shuffling**, with augmentations pre-applied to avoid performance issues during training.

All code and assets are located in the `Challenge1_2024` folder, which also includes a detailed report outlining the full pipeline, augmentation techniques, architecture design, and tested alternatives. In addition to the final model, the folder contains our two best alternative models: one based on **EfficientNetV2B3** and another using a **simple custom convolutional network**. The final ConvNeXt model achieved a **Codabench score of 0.86**, with performance validated across training, validation, and local test sets.

## Challenge 2 [Edition 2024] : Mars Terrain Segmentation

Competition available on [Kaggle](https://www.kaggle.com/t/af80f36772144dbb8b6179fea6180574 ).

This challenge focuses on a semantic segmentation task, classifying **64x128 greyscale images of Mars terrain** into **five classes**, each corresponding to a specific type of terrain. The classifier must assign a class label to **every pixel** in the input image.

The dataset contains **2,615 images**, reduced to **2,505** after removing outliers detected via t-SNE analysis. To address **class imbalance**, especially for class 4, we applied targeted data augmentation, generating up to **36 variations per sample** for that class. The training-validation split was carefully designed to reflect the imbalance, while optimizing data efficiency.

Our final model is based on **U-Net 3+**, modified with architectural enhancements including **dropout layers**, **trainable skip connection filters**, **squeeze-and-excite blocks**, and **L2 regularization**. We used a **weighted sparse categorical loss function** and trained the model over **two 100-epoch sessions**, reaching a **mean IoU of 0.643** on the Kaggle leaderboard. All training used a batch size of **64**, with **AdamW** optimizer and learning rate scheduling.

All code and resources are located in the `Challenge2_2024` folder, including:
- A detailed report describing the preprocessing pipeline, architecture experiments (MS-UNet, MarsSeg, Deep Residual U-Net), and training results.
- Python scripts used for **dataset modification and augmentation**.
- A reference implementation of a **basic U-Net** model for comparison and reproducibility.


## Challenge 1 [Edition 2023] : Leaf Classification

Competition available on [Codalab](https://codalab.lisn.upsaclay.fr/competitions/16245).

This challenge addresses a binary classification task: identifying whether a leaf in an image is **healthy** or **unhealthy**. The dataset consists of **5200 RGB images** of size **96x96**, each labeled accordingly. 

After cleaning the data—removing 98 duplicate outliers—and balancing the classes using **SMOTE**, we trained a deep learning model using **transfer learning**. The final architecture is based on **ConvNeXt Large**, with the first 90 layers frozen, and incorporates **batch normalization**, **dropout (0.1)**, and a **cyclical learning rate**. The best model achieved a **validation accuracy of 0.81**.

All code is located in the `Challenge1_2023` folder, along with a detailed report outlining the full pipeline and additional architecture experiments.

## Challenge 2 [Edition 2023] : Time Series Prediction

Competition available on [Codalab](https://codalab.lisn.upsaclay.fr/competitions/16514).

This challenge addresses a time series forecasting task: predicting the next steps of given time series sequences. The dataset contains **48,000 multivariate time series**, each with **2,776 time steps**, and associated metadata specifying a start and end index for the valid range. Although category labels (A–F) were provided, they were not used, as they did not correspond to distinct series behaviors.

After extracting the valid segments from each series, we processed the data into fixed-length sequences of **200 steps**, paired with targets of **18 future steps**, using a custom splicing approach. We applied basic data augmentation, including duplication with slight rounding variations, which improved results.

The final model is a **simple LSTM-based neural network** without convolutional or attention layers, followed by three dense layers. It was trained using a batch size of **32** and **MSE loss**, and evaluated using validation MSE. The best submission achieved an **MSE of 0.009** on the CodaLab leaderboard.

All code for this project is located in the `Challenge2_2023` folder, along with a detailed report documenting the full pipeline, architecture selection, and experimental results.










