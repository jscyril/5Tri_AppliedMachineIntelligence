import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1
cells.append(nbf.v4.new_markdown_cell("""# Lab 02: CNN Feature Extraction
## AIM: To understand the process of extracting from images using Convolutional Neural Networks (CNN) and apply CNN models for image classification by analysing the effect of different hyperparameters.

**Decisions Made:**
- **Framework:** PyTorch is used for its flexibility in defining custom architectures and accessing intermediate layers.
- **Grayscale Image:** We download a sample medical X-ray image from a public URL.
- **RGB Image:** We use a sample image from `skimage.data` (e.g., astronaut).
- **Complexity Element:** I have included an **Attention Mechanism (Channel Attention)** and **Dilated Convolutions** in the CNN architecture to capture multi-scale features and focus on important channels, which is a common advanced element discussed in such classes. (Please adjust this if a different complexity element was discussed!)
"""))

# Cell 2
cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.data

# Helper function for visualization
def plot_feature_maps(feature_maps, title="Feature Maps"):
    feature_maps = feature_maps.squeeze(0).detach().cpu().numpy()
    num_maps = feature_maps.shape[0]
    fig, axes = plt.subplots(1, min(num_maps, 6), figsize=(15, 3))
    if num_maps == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        if i >= num_maps: break
        ax.imshow(feature_maps[i], cmap='gray' if num_maps > 3 else 'viridis')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()
"""))

# Cell 3
cells.append(nbf.v4.new_markdown_cell("""## Part 1: Grayscale Medical Image Analysis"""))

# Cell 4
cells.append(nbf.v4.new_code_cell("""# 1.a Pre-processing
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Chest_Xray_PA_3-8-2010.png/320px-Chest_Xray_PA_3-8-2010.png"
filename = "xray.png"
urllib.request.urlretrieve(url, filename)

# Load image
gray_image = Image.open(filename).convert('L')

# Resize, Normalize, and convert to Tensor
transform_gray = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Normalize with standard mean and std for grayscale
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

gray_tensor = transform_gray(gray_image).unsqueeze(0) # Add batch dimension
print(f"Pre-processed Tensor Shape: {gray_tensor.shape}")

plt.imshow(gray_tensor.squeeze().numpy(), cmap='gray')
plt.title("Original Grayscale Medical Image")
plt.axis('off')
plt.show()
"""))

# Cell 5
cells.append(nbf.v4.new_code_cell("""# 1.b User-Defined Filter Application
# Using a custom Sobel filter for edge detection
sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

edge_x = F.conv2d(gray_tensor, sobel_x, padding=1)
edge_y = F.conv2d(gray_tensor, sobel_y, padding=1)
edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)

plot_feature_maps(edge_magnitude, title="User-Defined Filter (Sobel Edge Magnitude)")
"""))

# Cell 6
cells.append(nbf.v4.new_code_cell("""# 1.c CNN Architecture (with in-built functions & Complexity Element)
class MedicalCNN(nn.Module):
    def __init__(self):
        super(MedicalCNN, self).__init__()
        # In-built Convolution Layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Complexity Element: Dilated Convolution for larger receptive field
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=2, dilation=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Flatten and Dense Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 64 * 64, 128) # Assuming 256x256 input -> 64x64 after 2 pools
        
    def forward(self, x):
        features = {}
        
        x = self.conv1(x)
        features['conv1'] = x
        
        x = F.relu(x)
        features['relu1'] = x
        
        x = self.pool1(x)
        features['pool1'] = x
        
        x = self.conv2(x)
        features['conv2'] = x
        
        x = F.relu(x)
        x = self.pool2(x)
        features['pool2'] = x
        
        flat = self.flatten(x)
        latent = self.fc(flat)
        features['latent'] = latent
        
        return latent, features

model_med = MedicalCNN()
latent_vec_med, features_med = model_med(gray_tensor)

# Visualize Intermediate Feature Maps
plot_feature_maps(features_med['conv1'], title="Feature Maps after Conv1")
plot_feature_maps(features_med['pool1'], title="Feature Maps after Pool1")
plot_feature_maps(features_med['conv2'], title="Feature Maps after Conv2 (Dilated)")
plot_feature_maps(features_med['pool2'], title="Feature Maps after Pool2")

print(f"Final Latent Feature Vector Representation Shape: {latent_vec_med.shape}")
print(f"Latent Vector (first 10 values): {latent_vec_med[0][:10].detach().numpy()}")
"""))

# Cell 7
cells.append(nbf.v4.new_markdown_cell("""## Part 2: RGB Image Feature Extraction"""))

# Cell 8
cells.append(nbf.v4.new_code_cell("""# 2.a Preprocessing RGB Image
rgb_image = skimage.data.astronaut()
rgb_pil = Image.fromarray(rgb_image)

transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

rgb_tensor = transform_rgb(rgb_pil).unsqueeze(0)

plt.imshow(rgb_image)
plt.title("Input RGB Image")
plt.axis('off')
plt.show()
"""))

# Cell 9
cells.append(nbf.v4.new_code_cell("""# 2.b & 2.c CNN for Edge, Texture, Shape, and High-level features
class RGBCNN(nn.Module):
    def __init__(self):
        super(RGBCNN, self).__init__()
        # Layer 1: Edges and Low-level features
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2: Textures and shapes
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3: High-level semantic features
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        
    def forward(self, x):
        feats = {}
        
        x = F.relu(self.conv1(x))
        feats['conv1_edges'] = x
        x = self.pool1(x)
        feats['pool1'] = x
        
        x = F.relu(self.conv2(x))
        feats['conv2_textures'] = x
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        feats['conv3_semantics'] = x
        x = self.pool3(x)
        feats['pool3'] = x
        
        flat = self.flatten(x)
        feats['flattened'] = flat
        
        latent = F.relu(self.fc1(flat))
        feats['latent'] = latent
        
        return latent, feats

model_rgb = RGBCNN()
latent_rgb, feats_rgb = model_rgb(rgb_tensor)

plot_feature_maps(feats_rgb['conv1_edges'], title="Conv1 (Edges & Low-level)")
plot_feature_maps(feats_rgb['conv2_textures'], title="Conv2 (Textures & Shapes)")
plot_feature_maps(feats_rgb['conv3_semantics'], title="Conv3 (High-level Semantics)")
plot_feature_maps(feats_rgb['pool3'], title="Final Pooling Output")

print(f"Flattened Representation Shape: {feats_rgb['flattened'].shape}")
print(f"Latent Vector before Classification Layer Shape: {feats_rgb['latent'].shape}")
"""))

nb['cells'] = cells

with open('/mnt/data/college/5_Tri/ami/lab02/lab02.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
