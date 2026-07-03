# LexiCam ROI Autocrop

_Developed between 2021 and 2023; revisited in 2026._

This repository documents the region-of-interest (ROI) autocrop component used in LexiCam's reverse image search pipeline. The model predicts a bounding box around the artwork or object that should be cropped before similarity search.

![Illustration of algorithm](/readme_images/algo_illustration.png "Reverse image search")

The reverse image search flow has three main steps with two network round trips.

**1. ROI autocrop**

- A downsampled image is sent to the server
- The autocrop model predicts the region of interest and returns normalized bounding-box coordinates

**2. Vectorization of the cropped image (metric learning model)**

- The original image is cropped according to the bounding box, downsampled, and sent to the server
- The metric-learning model calculates an embedding from the cropped image

**3. Approximate nearest neighbor search in vector database**

- Images with the most similar embeddings are returned as results

If the full image search takes longer than 1 second, it starts to feel slow. The mobile app also needs time to resize and upload images (~65KB for 320x480 and ~25KB for 224x224), so the backend should stay well below 500ms in good network conditions. Steps 2 and 3 take around 200ms together. The target latency for the ROI autocrop microservice is therefore 100-200ms.

### Autocrop task

![Crop illustration](/readme_images/crop_illustration_small.png)  
The autocrop model needs to crop two types of subjects: paintings and physical art objects such as vases or jewelry.

A practical approach is to fine-tune a pretrained object detection model for this use case. The classification head is still evaluated, but its cost should be negligible compared with feature extraction. PyTorch provides several [pretrained object detection models](https://pytorch.org/vision/stable/models.html#object-detection), including mAP and computational cost (GFLOPS).

Faster R-CNN is a strong object detection baseline and supports different feature extractors ("backbones"). Benchmarking different backbones on my Linux laptop, which is not much faster than a typical low budget server instances, left two realistic contenders: MobileNet V3 at >500ms and MobileNetV3-320 at ~120ms. MobileNetV3-320 uses an input size where the shorter image side is 320px. That also reduces upload size from the app, making it the best choice for the current latency target. More computationally demanding models may become practical with quantization or GPU-backed serving.

### Training the model

I use [Optuna](https://optuna.org/) for hyperparameter search and stratified k-fold cross validation from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) to split the training and validation data.

1. Wide range hyperparameter optimization
2. Narrow range hyperparameter search
3. Training with fixed parameters and many epochs  
   At first it was possible to train the model locally. As the dataset and experiments grew, I moved training to DigitalOcean and Google Colab.

### Check if more data is useful

![graph to test if more data is needed](/readme_images/test_more_data_needed_1050images.png)  
Since obtaining and labeling data is time-consuming, it is useful to estimate how much additional data might improve the model. Training data usually has diminishing returns, so a learning curve can help extrapolate the likely benefit. There is still visible improvement from using 85% to 100% of the dataset, suggesting that additional data may help. Keep in mind that the y-axis starts at 0.22, which visually amplifies the actual difference.
![learning curve](/learning_curves/fastercnnmobile320_1050images_22epochs_3pred.png)  
Plotting training and validation loss over time helps inspect the learning process. The validation loss oscillates around the training loss, which suggests that the validation split (one third of the data in 3-fold cross validation) may not be fully representative. I added another 500 images to test whether more data would stabilize the result.
![new graph to test if more data is needed](/readme_images/test_more_data_needed_1500images.png)  
Running the same experiment with 1500 samples shows diminishing returns, as expected. The loss decrease from 1050 to 1500 samples was slightly smaller than the decrease from 850 to 1050 samples. Data volume does not seem to be the main bottleneck at this stage.

### Artificial training data (augmented images)

Image augmentation creates additional training examples by applying transformations to existing images. [Albumentations](https://albumentations.ai) provides a wide range of options. I selected transformations that should still produce realistic images: substantial changes to color, lighting, and sharpness, plus smaller changes to crop and perspective.
![Final training curve](/readme_images/augmentation.png)  
This way I increased the number of images for training from about 1500 original images to more than 4700 in total.

### Serving the model in production

[bbRegressionMicroservice](/bbRegressionMicroservice) contains a minimal FastAPI service for serving the ONNX model.

#### Model runtime

1. [TorchScript/JIT](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html) offers a way to optimize certain models for efficient multithreading during inference with little extra code. Fortunately this works for FasterRCNN and the inference latency decreases by about 40ms.
2. Using [ONNX Runtime](https://onnxruntime.ai) instead of TorchScript reduces average inference latency by another 20-30ms. To use ONNX Runtime, the model has to be exported to `.onnx`. [This script](./bbRegressionMicroservice/export_onnx.py) contains the export configuration.

#### Web framework

I chose [FastAPI](https://fastapi.tiangolo.com) because it is easy to use and offers reasonable resource utilization. Compared with web frameworks in C++, Rust, or Go, a Python web framework adds some latency and variance. In my tests, requests had around 20ms higher latency and occasional outliers of roughly 100ms. See the [TechEmpower benchmark](https://www.techempower.com/) for broader context. For now, switching the serving layer to a systems language does not seem worth the added complexity.

#### Model artifacts

Model weights (`*.pkl`) and exported ONNX models (`*.onnx`) are treated as generated artifacts and are not committed to this repository. To build the Docker image as-is, place the exported model at `bbRegressionMicroservice/out_dyn.onnx` first.

### Results

![Final training curve](/learning_curves/lr9_30_00004_1.17e-5_0.88.png)  
After trying many parameter combinations with Optuna and adjusting the learning rate, the final learning curve suggests that the model generalizes reasonably well.

### Testing

The model is tested on images that were not used during training:
![Test set illustration](/readme_images/testing.png)  
Because image augmentation can create near-duplicates across train and test splits, I keep a second application test set without augmented images.
As evaluation metric I use intersection-over-union (IOU) and % of misses.  
Test set: ~75.0% IOU, 3% misses  
Application test set: ~71.5% IOU, 0% misses

### Possible ways to further improve the model

- Use a more complex model while keeping latency low by
  - serving the model on a server with GPUs
  - reducing the computational complexity via [Quantization](https://pytorch.org/docs/stable/quantization.html) (FasterRCNN with Mobilenet was not supported for quantization in PyTorch at the time of writing)
- Search the current dataset for hard cases to identify which additional images would be most useful

### 2026 Update

By 2026, the model and deployment landscape has changed substantially. More efficient transformer-based detectors, such as RF-DETR, can achieve higher mAP than Faster R-CNN with only modest additional computational cost. At the same time, React Native integrations for runtimes such as ExecuTorch and ONNX Runtime make it much more practical to run inference directly on mobile devices. That could remove one network round trip from the search process while also allowing the app to use a stronger object detection model for autocrop.
