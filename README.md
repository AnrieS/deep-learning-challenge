# deep-learning-challenge

The challenge was fine-tuning the tensorflow keras model to predict an accurate algorithm

for the original model, I had the cut-off for the classification at 1800 and I encoded the values of True and False to binary 1's and 0's
The results of the original model was pretty inaccurate and I struggled throughout the challenge to find an optimised answer for the predictive model. I couldn't find an answer to get a higher accuracy rating for the tensorflow model.

Results/Code: 
  input_features_total = len(X_train[0])
  hidden_nodes_layer1 = 8
  hidden_nodes_layer2 = 5
  fit_model = nn.fit(X_train_scaled,y_train,epochs=100)
  Loss: 0.5559459924697876, Accuracy: 0.7290962338447571


For the first optimisation model, I tried to add more nodes and hidden layers in the defining the model process, however this later showed to be unfruitful. With an accuracy score of 72.6 which is lower than my initial model I found maybe having too many nodes and hidden layers can prove to be less productive.
Results/code:
  input_features_total = len(X_train[0])
  hidden_nodes_layer1 = 100
  hidden_nodes_layer2 = 80
  hidden_nodes_layer3 = 60
  fit_model = nn.fit(X_train_scaled,y_train,epochs=80)
  Loss: 0.5784154534339905, Accuracy: 0.7261807322502136

The second optimisation was no better and I got an even worse accuracy score for my model with 72.5, lower than my original model, and even lower than my first optimisation model. This time I removed another hidden layer and kept the nodes the same 80 for the first layer and 60 for the second.
Results/Code:
  input_features_total = len(X_train[0])
  hidden_nodes_layer1 = 80
  hidden_nodes_layer2 = 60
  fit_model = nn.fit(X_train_scaled,y_train,epochs=80)
Loss: 0.5589141249656677, Accuracy: 0.7255976796150208

The third and final optimisation was the best, but the change was too small for it to be called an improvement. With two hidden layers, and lessening the nodes to 35 and 15 respectively. The smaller nodes in the hidden layers had a small impact on the model's accuracy. 
Results/Code:
  input_features_total = len(X_train[0])
  hidden_nodes_layer1 = 35
  hidden_nodes_layer2 = 15
  fit_model = nn.fit(X_train_scaled,y_train,epochs=80)
  Loss: 0.5520990490913391, Accuracy: 0.7300291657447815
