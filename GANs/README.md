# DeepLearning

## GANs

These scripts implementats of GANS: DCGAN, WGAN (weight clipping), and WGAN-GP (gradient clipping) using the Fashion-MNIST dataset.

### Prerequisites

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Google Colab (Optional: to run the script)

### DCGANs, WGANs, and WGAN-GPs

To train our GANs, we created a function `run(model_name, filter_size)` that specifies which of three methods we want to use, and the base filter size to use. Our two architectures differ in the base filter size, where one uses 32 and the other uses 64.

Example command:
```
G_WGAN_GP64, D_WGAN_GP64, G_losses_WGAN_GP64, D_losses_WGAN_GP64, img_list_WGAN_GP64 = run("WGAN-GP", 64)

[1/5][0/469]	Loss_G: 0.0330	Loss_C: -0.0531
[1/5][25/469]	Loss_G: 0.6247	Loss_C: -1.3128
[1/5][50/469]	Loss_G: 0.6250	Loss_C: -1.4137
[1/5][75/469]	Loss_G: 0.6084	Loss_C: -1.3106
[1/5][100/469]	Loss_G: 0.6898	Loss_C: -1.4923
[1/5][125/469]	Loss_G: 0.7156	Loss_C: -1.5165
...
```

Our models are all saved in the `model` folder. 

To test our models, we created a function `generate_images_from_trained_model(path, name, latent_size, filter_size, nc=1)` that uses a specified model to generate images. You can generate 6 images from each of the ten classes. 

Options for parameters:
- path: `YOUR_PATH_TO_MODEL` (path you saved your model to during training)
- name: `DCGAN, WGAN, WGAN_GP` 
- latent_size: `100`
- filter_size: `32, 64`

Example command:
```
generate_images_from_trained_model("/content/drive/MyDrive/Colab Notebooks/model/G_WGAN_GP64.pth", "WGAN-GP", 100, 64)
```
