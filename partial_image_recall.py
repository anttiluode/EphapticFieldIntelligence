
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from holographic import EphapticInfoStorage

def load_image(path, size=(128, 128)):
    img = Image.open(path).convert('L').resize(size)
    return np.array(img) / 255.0

def show_image(img_array, title="Image"):
    plt.imshow(img_array, cmap='gray', origin='lower')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load full image and store it in the field
storage = EphapticInfoStorage(field_shape=(128, 128), num_instantons=8)
full_image = load_image('full_image_example.png')
config_id = storage.store_data(full_image, label='original_image')

print(f"✅ Stored full image as configuration ID: {config_id}")

# Now load partial image
partial_image = load_image('partial_image_patch.png')

# Try using partial as a retrieval cue
retrieved = storage.retrieve_data_by_similarity(partial_image)

# Show all
show_image(full_image, title='Original Full Image (Stored)')
show_image(partial_image, title='Partial Input (Recall Trigger)')
show_image(retrieved, title='Recalled Image from Field')
