# DeepLearning

## VAE

These scripts implements and trains a VAE for feature extraction and an SVM for classification using the Fashion-MNIST dataset.

### Prerequisites

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Google Colab (Optional: to run the script)

### VAE

The function `run-experiment()` trains and tests our VAE model and the SVM classifier. It will also print out the final accuracies for the four different numbers of labels (100, 600, 1000, 3000). 

Example log:

```
Epoch 1 complete! 	Average Loss:  407.2164127604167
Epoch 2 complete! 	Average Loss:  323.04082421875
Epoch 3 complete! 	Average Loss:  304.61422330729164
Epoch 4 complete! 	Average Loss:  295.2882578125
Epoch 5 complete! 	Average Loss:  288.2894140625
Epoch 6 complete! 	Average Loss:  282.1976510416667
...
{100: 65.09, 600: 73.22, 1000: 75.71, 3000: 80.54}
```

Our models are all saved in the local `model` folder. Both the VAE model and the SVM model are saved after training.

To test our model, we created a function `test_model(size=3000)` where we feed in a random latent vector and generate an image using the decoder from our trained VAE model, then we use our trained SVM to predict the class. Size can be `100, 600, 1000, 3000`.