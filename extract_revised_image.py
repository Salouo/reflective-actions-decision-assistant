import pickle
import json
import glob
from PIL import Image


with open('data/revised_scenario_idxs.pkl', 'rb') as f:
    revised_idx = pickle.load(f)['revised']

# Load all 400 raw image paths
file_paths = glob.glob('raw_images/*.jpg')
# Sort in-placed based on file name.
file_paths.sort()

# Load all 400 raw image
images = []
for file_path in file_paths:
    img = Image.open(file_path)
    images.append(img)

# Extract revised images from raw images
revised_images = [images[i] for i in range(len(images)) if i in revised_idx]

# Save 355 revised images as a list at .pkl format
with open('data/revised_images.pkl', 'wb') as f:
    pickle.dump(revised_images, f)
