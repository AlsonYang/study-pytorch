# study-pytorch


## 7_CIFAR10_classification
![The CNN architecture](https://user-images.githubusercontent.com/31328407/190283336-1d7b604b-c6e0-4338-9266-48dfa91a92ee.png)

- Feature learning: The CNN layer uses filters and pooling to extract + condense the input into a smaller but paramount features. 
- Classification: It flatten out the CNN features, and use fully connected layer to do the classification task

CNN specification
- `filters`:
    - purpose: filter is a weight metrix that learns to represent abstraction of the given matrix
    - It will reduce the size of the original matrix as side effect, but can be dealt with using `padding`
    - num_Kernels(filters) = num_input_channels * num_output_channels
    - params:
        - in_channels: For colour image, it's 3. For grey scale image, it's 1
        - out_channels: Anything. The more channels, the more feature space representation
        - kernel_size: the size of filter matrix
        - stride: The number of pixel it shifts each time, usually 1

- `pooling`:
    - purpose: condense the matrix representation into a smaller matrix. ie. Max pooling
    - benefit: This will force the neural network to represent information condensely (throw away the unimportant)
    - params:
        - kernel_size: the size of matrix
        - stride: The number of pixel it shifts each time


# ML Flow
## Tracking
```bash
# create local host for web UI to check experiment
mlflow ui
# 
```