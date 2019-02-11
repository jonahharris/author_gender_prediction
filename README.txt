Text classification model using Multilayer perceptron model (mlp)
Tested on Ubuntu 16.04 LTS

***

Requirements:
python 3.5
pandas (pip install pandas)
numpy (pip install numpy)
scikit-learn (pip install scikit-learn)
tensorflow (pip install tensorflow)

***
Usage:
train - ./train.sh training_data models/post_mlp_model.h5

predict - ./classify.sh models/post_mlp_model.h5 "Your post to predict"
