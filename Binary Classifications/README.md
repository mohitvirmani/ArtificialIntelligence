The program creates a prediction model on 
the Banknote Authentication dataset 
where the problem is to predict whether 
a banknote (think dollar bill or euro) 
is authentic or a forgery, based on four 
predictor variables. The demo loads a training 
subset into memory then creates a 4-(8-8)-1 deep neural network.

After training for 500 iterations, the resulting 
model scores 99.27 percent accuracy on
 a held-out test dataset. The demo concludes by 
 making a prediction for a hypothetical banknote 
 that has average input values. The probability 
 that the unknown item is a forgery is only 0.0009,
 therefore the conclusion is that the banknote is authentic.