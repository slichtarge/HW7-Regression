"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import sys
import os

#adding the parent directory to the search path since apparently that was necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regression import logreg

# (you will probably need to import more things here)

def test_prediction():
	#going with a simple case: 2 samples, 1 feature + 1 bias
    model = logreg.LogisticRegressor(num_feats=1)
    model.W = np.array([1.0, 0.0]) #weight for feature is 1, bias is 0
    
    X = np.array([[0.0, 1.0], [2.0, 1.0]])
        
    preds = model.make_prediction(X)
    
    assert preds[0] == pytest.approx(0.5) #answer should be 0.5
    assert preds[1] == pytest.approx(0.880797, rel=1e-5) # answer should be approx 0.880797

def test_loss_function():
	model = logreg.LogisticRegressor(num_feats=1)
	y_true = np.array([1, 0])
	y_pred = np.array([0.9, 0.1])
      
	calculated_loss = model.loss_function(y_true, y_pred)
	assert calculated_loss == pytest.approx(0.10536, rel=1e-4) #should be approx 0.10536

def test_gradient():
	model = logreg.LogisticRegressor(num_feats=1)
	model.W = np.array([0.0, 0.0]) #weights are zero
	
	X = np.array([[1.0, 1.0], [2.0, 1.0]]) #features with bias
	y_true = np.array([1, 0])
    
    #y_pred will be [0.5, 0.5]
    #grad should be[0.25, 0.0]
	grad = model.calculate_gradient(y_true, X)
	assert np.allclose(grad, np.array([0.25, 0.0]))

def test_training():
	#do the weights actually change after one epoch?
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1, 0, 1])
    X_val = X_train
    y_val = y_train
    
    model = logreg.LogisticRegressor(num_feats=1, learning_rate=0.1, max_iter=2)
    initial_weights = model.W.copy()
    
    model.train_model(X_train, y_train, X_val, y_val)
    
    #weights should not be the same as the random initialization
    assert not np.array_equal(model.W, initial_weights)
    #loss history should have entries
    assert len(model.loss_hist_train) > 0