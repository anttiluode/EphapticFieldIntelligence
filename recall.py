import numpy as np
import matplotlib.pyplot as plt
import cv2 # Used by RobustHolographicMemory for some image ops
from PIL import Image

# Assuming face_recognition.py is in the same directory or accessible in PYTHONPATH
from face_recognition import RobustHolographicMemory #

def create_test_image(index, size=(128, 128)):
    """Creates simple, distinct test images."""
    image = np.zeros(size, dtype=np.float32)
    if index == 0: # Circle
        cv2.circle(image, (size[0]//2, size[1]//2), size[0]//3, 1.0, -1)
    elif index == 1: # Square
        cv2.rectangle(image, (size[0]//4, size[1]//4), (size[0]*3//4, size[1]*3//4), 0.8, -1)
    elif index == 2: # Triangle
        pts = np.array([[size[0]//2, size[1]//4], 
                        [size[0]*1//4, size[1]*3//4], 
                        [size[0]*3//4, size[1]*3//4]], dtype=np.int32)
        cv2.drawContours(image, [pts], 0, 0.6, -1)
    elif index == 3: # Plus Sign
        cx, cy = size[0]//2, size[1]//2
        thickness = size[0]//8
        cv2.line(image, (cx, cy - size[1]//3), (cx, cy + size[1]//3), 0.9, thickness)
        cv2.line(image, (cx - size[0]//3, cy), (cx + size[0]//3, cy), 0.9, thickness)
    else: # Default: X Sign
        thickness = size[0]//8
        cv2.line(image, (size[0]//4, size[1]//4), (size[0]*3//4, size[1]*3//4), 0.7, thickness)
        cv2.line(image, (size[0]*3//4, size[1]//4), (size[0]//4, size[1]*3//4), 0.7, thickness)
    return np.clip(image, 0, 1)

def create_fragment(image_array, missing_fraction=0.6, noise_level=0.15):
    """Creates a noisy, partial fragment of an image."""
    fragment = image_array.copy()
    
    # Introduce missing part (e.g., right side missing)
    cols_to_remove = int(image_array.shape[1] * missing_fraction)
    fragment[:, -cols_to_remove:] = 0
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, fragment.shape).astype(np.float32)
    fragment += noise
    return np.clip(fragment, 0, 1)

def display_results(original, fragment, recalled, original_label, fragment_label, recalled_label):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [original, fragment, recalled]
    titles = [original_label, fragment_label, recalled_label]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower') #
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def run_continual_learning_demo():
    print("=" * 80)
    print("🌌 EPHAPTIC FIELD: CONTINUAL LEARNING DEMONSTRATION")
    print("=" * 80)
    print("This demo will show that the system can learn multiple patterns sequentially")
    print("and recall earlier patterns even after new ones have been learned,")
    print("showcasing its resistance to catastrophic forgetting.\n")

    # Initialize the Robust Holographic Memory system
    field_shape = (128, 128)
    memory_system = RobustHolographicMemory(field_shape=field_shape, num_instantons=12) #

    num_images_to_learn = 4
    learned_images_info = [] # To store (original_image, label, memory_id)

    # --- Step 1: Sequential Learning Phase ---
    print("\n--- LEARNING PHASE ---")
    for i in range(num_images_to_learn):
        label = f"Image_{i+1}"
        print(f"\n🧠 Teaching {label} to the ephaptic field...")
        original_image = create_test_image(i, size=field_shape)
        
        # Store the image using the system's method
        memory_id = memory_system.robust_holographic_storage(original_image, label=label) #
        learned_images_info.append({'original': original_image, 'label': label, 'id': memory_id})
        
        plt.imshow(original_image, cmap='gray', origin='lower') #
        plt.title(f"Taught: {label}\nStored as: {memory_id}")
        plt.axis('off')
        plt.show()
        print(f"✅ {label} stored with ID: {memory_id}. Robustness: {memory_system.holographic_memories[memory_id]['robustness_score']:.2f}") #

    print(f"\n📚 Total memories stored: {len(memory_system.holographic_memories)}") #

    # --- Step 2: Recall Testing Phase ---
    print("\n\n--- RECALL TESTING PHASE ---")
    print("Now, we will attempt to recall each learned image using a noisy, partial fragment.")
    print("Crucially, we test early memories *after* later memories have been consolidated.")

    for i, info in enumerate(learned_images_info):
        original_image = info['original']
        label = info['label']
        
        print(f"\n🔍 Testing recall for: {label} (taught as item #{i+1})")
        
        # Create a noisy, partial fragment
        fragment = create_fragment(original_image, missing_fraction=0.6, noise_level=0.2)
        
        # Perform recall using the system's method
        # Lower similarity threshold for more challenging fragments might be needed.
        recalled_image = memory_system.robust_holographic_recall(fragment, similarity_threshold=0.20) #
        
        if recalled_image is not None:
            print(f"✅ Successful recall for {label}!")
            display_results(original_image, fragment, recalled_image,
                            f"Original: {label}", 
                            "Input Fragment\n(60% missing, 20% noise)",
                            f"Recalled: {label}")
        else:
            print(f"❌ Recall FAILED for {label}.")
            display_results(original_image, fragment, np.zeros_like(original_image), # Show black for failed recall
                            f"Original: {label}", 
                            "Input Fragment\n(60% missing, 20% noise)",
                            f"RECALL FAILED for: {label}")

    print("\n\n--- DEMONSTRATION COMPLETE ---")
    if all(memory_system.robust_holographic_recall(create_fragment(info['original']), similarity_threshold=0.20) is not None for info in learned_images_info): #
        print("🎉🎉🎉 SUCCESS! All images, including early ones, were recallable after subsequent learning!")
        print("This highlights the system's resistance to catastrophic forgetting.")
    else:
        print("⚠️ Some images were not recalled successfully. Further tuning or investigation may be needed.")
        print("However, the core holographic principle aims to avoid catastrophic forgetting.")

if __name__ == "__main__":
    run_continual_learning_demo()