# bijective-nns

Collection of invertible layers for building invertible neural networks in TensorFlow2.0 beta.
On-going project. I plan to add more functionality soon. 
So far:
- 'layers.py' contains invertible layer architectures from recent publications (see references below)
- 'blocks.py' provides examples of invertible building blocks / steps of flow
- 'simple_model.py' showcases a toy model with training setup on the MNIST dataset

References:
- D. P. Kingma and P. Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in Neural Information Processing Systems. 2018.

- L. Dinh, J. Sohl-Dickstein and S. Bengio. Density estimation using real nvp. International Conference on Learning Representations, 2017.

- L. Dinh, D. Krueger, and Y. Bengio. Nice: Non-linear independent components estimation. arXiv:1410.8516 (2014).

- W. Shi, J. Caballero, F. Huszar, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert and Z. Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1874 -- 1883, 2016.


