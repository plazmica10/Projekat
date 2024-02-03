import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import  mean_squared_error

class GradientBooster:
    
    def __init__(self,max_depth=8,lr=0.1,iter=1000,random_state=42):
        self.max_depth = max_depth
        self.lr = lr
        self.num_iter = iter
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.min_samples_split = 3
        self.min_samples_leaf = 3
        self.max_features = 3
        self.y_mean = 0
    
    def predict(self,models,X):
        pred_0 = np.array([self.y_mean] * len(X))
        pred = pred_0.reshape(len(pred_0),1)
        
        for i in range(len(models)):
            temp = (models[i].predict(X)).reshape(len(X),1)
            pred -= self.lr * temp
        
        return pred
    
    def train(self, X, y):
        models = []
        losses = []

        self.y_mean = np.mean(y)
        #first prediction is average of y
        pred_0 = np.array([np.mean(y)] * len(y))
        #reshape for creating base model because it needs 2D array
        pred = pred_0.reshape(len(pred_0),1)

        #iterative model creation
        for epoch in range(self.num_iter):
            #calculate loss
            loss = np.sqrt(mean_squared_error(y,pred))
            losses.append(loss)

            #calculate gradients
            grads = -(y-pred)
 
            #create decision tree
            base = DecisionTreeRegressor(criterion='squared_error',max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            max_features=self.max_features,random_state=self.random_state)
            base.fit(X,grads)
            
            #make prediction
            r = (base.predict(X)).reshape(len(X),1)
            #update prediction
            pred -= self.lr * r
            #store the model for future use
            models.append(base)
        return models, losses, pred