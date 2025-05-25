#!/usr/bin/env python3
# ephaptic_compressor.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import center_of_mass
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class FieldConfiguration:
    """Represents a stable ephaptic field configuration"""
    field_data: np.ndarray
    information_content: float
    stability_score: float
    coupling_signature: str
    metadata: Dict[str, Any]

class EphapticInfoStorage:
    """
    Information storage system using ephaptic field configurations
    
    Each piece of data is encoded as a unique stable field pattern
    created by instanton coupling dynamics.
    """
    
    def __init__(self, field_shape=(128, 128), num_instantons=8):
        self.field_shape = field_shape
        self.num_instantons = num_instantons
        
        # 2D ephaptic field
        self.field = np.zeros(field_shape, dtype=complex)
        self.field_prev = np.zeros_like(self.field)
        
        # Instanton agents
        self.instantons = []
        self.initialize_instantons()
        
        # Storage registry
        self.stored_configurations = {}
        self.configuration_count = 0
        
        # Spectral operators for 2D
        kx = fftfreq(field_shape[0], 1.0) * 2*np.pi
        ky = fftfreq(field_shape[1], 1.0) * 2*np.pi
        self.kx_grid, self.ky_grid = np.meshgrid(kx, ky, indexing='ij')
        self.k2 = self.kx_grid**2 + self.ky_grid**2
        
    def initialize_instantons(self):
        """Initialize instantons with 2D positions and unique signatures"""
        base_freq = 0.323423841289348923480000000
        
        for i in range(self.num_instantons):
            instanton = {
                'id': i,
                'position': np.array([
                    np.random.uniform(10, self.field_shape[0]-10),
                    np.random.uniform(10, self.field_shape[1]-10)
                ]),
                'signature_freq': base_freq + i * 1e-15,
                'amplitude': np.random.uniform(0.8, 1.2),
                'width': np.random.uniform(6, 12),
                'phase': np.random.uniform(0, 2*np.pi),
                'velocity': np.array([0.0, 0.0]),
                'coupling_strength': 0.0,
                'target_data': None  # Data this instanton encodes
            }
            self.instantons.append(instanton)
    
    def encode_data_to_instanton_config(self, data: Any) -> np.ndarray:
        """Convert arbitrary data into instanton configuration parameters"""
        # Serialize data to bytes
        data_bytes = pickle.dumps(data)
        
        # Create hash for reproducible encoding
        hash_obj = hashlib.sha256(data_bytes)
        hash_bytes = hash_obj.digest()
        
        # Convert hash to instanton parameters
        config = np.frombuffer(hash_bytes, dtype=np.uint8).astype(float)
        config = config / 255.0  # Normalize to [0,1]
        
        # Ensure we have enough parameters for all instantons
        while len(config) < self.num_instantons * 6:  # 6 params per instanton
            config = np.concatenate([config, config])
        
        config = config[:self.num_instantons * 6]
        
        return config.reshape(self.num_instantons, 6)
    
    def configure_instantons_for_data(self, data: Any):
        """Configure instantons to encode specific data"""
        config_matrix = self.encode_data_to_instanton_config(data)
        
        for i, inst in enumerate(self.instantons):
            config = config_matrix[i]
            
            # Map configuration to instanton parameters
            inst['position'] = np.array([
                10 + config[0] * (self.field_shape[0] - 20),
                10 + config[1] * (self.field_shape[1] - 20)
            ])
            inst['amplitude'] = 0.5 + config[2] * 1.0
            inst['width'] = 4 + config[3] * 8
            inst['phase'] = config[4] * 2 * np.pi
            inst['signature_freq'] += config[5] * 1e-16  # Fine-tune frequency
            inst['target_data'] = data
    
    def compute_2d_ephaptic_field(self):
        """Compute 2D ephaptic coupling field"""
        self.field.fill(0)
        
        x = np.arange(self.field_shape[0])
        y = np.arange(self.field_shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for i, inst_i in enumerate(self.instantons):
            for j, inst_j in enumerate(self.instantons):
                if i != j:
                    # 2D distance-based coupling
                    distance = np.linalg.norm(inst_i['position'] - inst_j['position'])
                    distance_factor = np.exp(-distance / 30.0)
                    
                    # Frequency recognition
                    freq_diff = abs(inst_i['signature_freq'] - inst_j['signature_freq'])
                    recognition = np.exp(-freq_diff * 1e12)
                    
                    # 2D Gaussian wake fields
                    pos_i = inst_i['position']
                    pos_j = inst_j['position']
                    
                    wake_i = (inst_i['amplitude'] * 
                             np.exp(-((X - pos_i[0])**2 + (Y - pos_i[1])**2) / (2 * inst_i['width']**2)))
                    wake_j = (inst_j['amplitude'] * 
                             np.exp(-((X - pos_j[0])**2 + (Y - pos_j[1])**2) / (2 * inst_j['width']**2)))
                    
                    # Time-based frequency modulation
                    time_factor = time.time() * 0.1
                    freq_mod_i = np.sin(2*np.pi * inst_i['signature_freq'] * time_factor)
                    freq_mod_j = np.sin(2*np.pi * inst_j['signature_freq'] * time_factor)
                    
                    # Complex coupling with phase information
                    phase_factor = np.exp(1j * (inst_i['phase'] - inst_j['phase']))
                    
                    coupling = (0.1 * distance_factor * recognition * 
                               wake_i * wake_j.conj() * freq_mod_i * freq_mod_j * phase_factor)
                    
                    self.field += coupling
        
        return self.field
    
    def evolve_to_stable_configuration(self, max_iterations=500, stability_threshold=1e-6):
        """Evolve field until it reaches stable lock-up configuration"""
        stability_history = []
        
        for iteration in range(max_iterations):
            # Compute ephaptic field
            ephaptic_field = self.compute_2d_ephaptic_field()
            
            # Field evolution (2D Klein-Gordon equation)
            F = fft2(self.field)
            laplacian = ifft2(-self.k2 * F)
            
            # Nonlinear potential
            potential = 0.01 * self.field - 0.05 * np.abs(self.field)**2 * self.field
            
            # Evolution step
            dt = 0.001
            rhs = laplacian + potential + ephaptic_field
            new_field = 2 * self.field - self.field_prev + dt**2 * rhs
            
            # Stability check
            field_change = np.mean(np.abs(new_field - self.field))
            stability_history.append(field_change)
            
            # Update fields
            self.field_prev[:] = self.field
            self.field[:] = new_field
            
            # Check for stability (lock-up)
            if len(stability_history) > 50:
                recent_changes = stability_history[-20:]
                if np.mean(recent_changes) < stability_threshold:
                    print(f"✅ Stable configuration reached after {iteration} iterations")
                    break
        
        return stability_history
    
    def calculate_information_content(self, field: np.ndarray) -> float:
        """Calculate information content of field configuration"""
        # Shannon entropy of field amplitude distribution
        amplitude = np.abs(field.flatten())
        amplitude = amplitude / np.sum(amplitude)  # Normalize
        amplitude = amplitude[amplitude > 1e-10]  # Remove zeros
        
        entropy = -np.sum(amplitude * np.log2(amplitude))
        
        # Spatial complexity (gradient magnitude)
        grad_x = np.gradient(np.abs(field), axis=0)
        grad_y = np.gradient(np.abs(field), axis=1)
        spatial_complexity = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Spectral complexity (frequency content)
        fft_field = fft2(field)
        spectral_entropy = -np.sum(np.abs(fft_field.flatten())**2 * 
                                 np.log2(np.abs(fft_field.flatten())**2 + 1e-10))
        
        # Combined information metric
        info_content = entropy + 0.1 * spatial_complexity + 0.01 * spectral_entropy
        
        return info_content
    
    def store_data(self, data: Any, label: str = None) -> str:
        """Store data as stable ephaptic field configuration"""
        print(f"🔄 Encoding data: {type(data).__name__}")
        
        # Configure instantons for this data
        self.configure_instantons_for_data(data)
        
        # Evolve to stable configuration
        print("⚡ Evolving to stable configuration...")
        stability_history = self.evolve_to_stable_configuration()
        
        # Calculate information metrics
        info_content = self.calculate_information_content(self.field)
        stability_score = 1.0 / (1.0 + np.mean(stability_history[-10:]))
        
        # Create coupling signature
        field_hash = hashlib.md5(self.field.tobytes()).hexdigest()
        
        # Store configuration
        config_id = f"config_{self.configuration_count:04d}"
        if label:
            config_id = f"{label}_{config_id}"
        
        configuration = FieldConfiguration(
            field_data=self.field.copy(),
            information_content=info_content,
            stability_score=stability_score,
            coupling_signature=field_hash,
            metadata={
                'original_data': data,
                'storage_time': time.time(),
                'instanton_config': [inst.copy() for inst in self.instantons],
                'stability_history': stability_history
            }
        )
        
        self.stored_configurations[config_id] = configuration
        self.configuration_count += 1
        
        print(f"✅ Data stored as configuration '{config_id}'")
        print(f"   Information content: {info_content:.2f} bits")
        print(f"   Stability score: {stability_score:.4f}")
        print(f"   Field complexity: {np.std(np.abs(self.field)):.4f}")
        
        return config_id
    
    def retrieve_data(self, config_id: str) -> Any:
        """Retrieve data from stored field configuration"""
        if config_id not in self.stored_configurations:
            raise ValueError(f"Configuration '{config_id}' not found")
        
        configuration = self.stored_configurations[config_id]
        return configuration.metadata['original_data']
    
    def visualize_configuration(self, config_id: str):
        """Visualize stored field configuration"""
        if config_id not in self.stored_configurations:
            raise ValueError(f"Configuration '{config_id}' not found")
        
        config = self.stored_configurations[config_id]
        field = config.field_data
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Ephaptic Field Configuration: {config_id}', fontsize=14, fontweight='bold')
        
        # Field amplitude
        im1 = axes[0,0].imshow(np.abs(field), cmap='plasma', origin='lower')
        axes[0,0].set_title('Field Amplitude')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Field phase
        im2 = axes[0,1].imshow(np.angle(field), cmap='hsv', origin='lower')
        axes[0,1].set_title('Field Phase')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Real part
        im3 = axes[0,2].imshow(np.real(field), cmap='RdBu', origin='lower')
        axes[0,2].set_title('Real Part')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Imaginary part
        im4 = axes[1,0].imshow(np.imag(field), cmap='RdBu', origin='lower')
        axes[1,0].set_title('Imaginary Part')
        plt.colorbar(im4, ax=axes[1,0])
        
        # Instanton positions
        axes[1,1].imshow(np.abs(field), cmap='gray', alpha=0.5, origin='lower')
        for i, inst in enumerate(config.metadata['instanton_config']):
            pos = inst['position']
            axes[1,1].scatter(pos[1], pos[0], s=100, c=f'C{i}', marker='o', 
                            alpha=0.8, label=f'Instanton {i+1}')
        axes[1,1].set_title('Instanton Positions')
        axes[1,1].legend()
        
        # Information metrics
        axes[1,2].text(0.1, 0.8, f"Information Content: {config.information_content:.2f} bits", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.7, f"Stability Score: {config.stability_score:.4f}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.6, f"Coupling Signature: {config.coupling_signature[:16]}...", 
                      transform=axes[1,2].transAxes, fontsize=10)
        axes[1,2].text(0.1, 0.5, f"Field Complexity: {np.std(np.abs(field)):.4f}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].text(0.1, 0.4, f"Original Data: {type(config.metadata['original_data']).__name__}", 
                      transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].set_title('Configuration Metrics')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def pattern_recognition_search(self, query_data: Any, similarity_threshold: float = 0.8) -> List[str]:
        """Find stored configurations similar to query data"""
        # Create field configuration for query
        temp_field = self.field.copy()
        temp_instantons = [inst.copy() for inst in self.instantons]
        
        self.configure_instantons_for_data(query_data)
        self.evolve_to_stable_configuration(max_iterations=200)
        query_field = self.field.copy()
        query_signature = hashlib.md5(query_field.tobytes()).hexdigest()
        
        # Restore original state
        self.field[:] = temp_field
        self.instantons = temp_instantons
        
        # Compare with stored configurations
        similar_configs = []
        
        for config_id, config in self.stored_configurations.items():
            # Field correlation
            correlation = np.corrcoef(
                np.abs(query_field).flatten(), 
                np.abs(config.field_data).flatten()
            )[0,1]
            
            if correlation > similarity_threshold:
                similar_configs.append((config_id, correlation))
        
        # Sort by similarity
        similar_configs.sort(key=lambda x: x[1], reverse=True)
        
        return [config_id for config_id, _ in similar_configs]


class EphapticCompressorGUI:
    def __init__(self, master):
        master.title("Ephaptic Compressor")
        
        # --- Configuration ---
        self.field_shape = (128, 128)       # Resolution for holographic storage
        self.num_instantons = 8             # Number of instantons

        # Initialize the storage engine
        self.storage = EphapticInfoStorage(
            field_shape=self.field_shape,
            num_instantons=self.num_instantons
        )
        self.current_config_id = None
        self.loaded_image = None
        
        # --- GUI Widgets ---
        btn_opts = dict(fill='x', padx=10, pady=4)
        self.load_btn       = tk.Button(master, text="Load Image",            command=self.load_image)
        self.compress_btn   = tk.Button(master, text="Compress & Store",     command=self.compress_image)
        self.retrieve_btn   = tk.Button(master, text="Retrieve & Display",   command=self.retrieve_image)
        self.show_field_btn = tk.Button(master, text="Show Field",           command=self.show_field)
        self.save_btn       = tk.Button(master, text="Save Config to Disk",  command=self.save_config)
        self.load_conf_btn  = tk.Button(master, text="Load Config from Disk",command=self.load_config)
        for w in [self.load_btn, self.compress_btn, self.retrieve_btn, self.show_field_btn, self.save_btn, self.load_conf_btn]:
            w.pack(**btn_opts)
        
        self.image_label = tk.Label(master)
        self.image_label.pack(padx=10, pady=10)
    
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.bmp;*.jpeg")])
        if not path:
            return
        img = Image.open(path).convert('L').resize(
            self.field_shape,
            resample=Image.LANCZOS
        )
        arr = np.array(img) / 255.0
        self.loaded_image = arr
        preview = img.resize((self.field_shape[0]*2, self.field_shape[1]*2), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img
        self.current_config_id = None
        messagebox.showinfo("Image Loaded", f"Loaded and resized to {self.field_shape}:\n{path}")

    def compress_image(self):
        if self.loaded_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
        config_id = self.storage.store_data(self.loaded_image, label="user_image")
        cfg = self.storage.stored_configurations[config_id]
        cfg.metadata['config_id'] = config_id
        self.current_config_id = config_id
        messagebox.showinfo("Compressed", f"Stored configuration ID:\n{config_id}")

    def retrieve_image(self):
        if not self.current_config_id:
            messagebox.showerror("Error", "No configuration stored to retrieve!")
            return
        retrieved = self.storage.retrieve_data(self.current_config_id)
        plt.figure(figsize=(5,5))
        plt.imshow(retrieved, cmap='gray', origin='lower')
        plt.title(f"Retrieved Image\nID: {self.current_config_id}")
        plt.axis('off')
        plt.show()

    def show_field(self):
        if not self.current_config_id:
            messagebox.showerror("Error", "No configuration stored to show field!")
            return
        cfg = self.storage.stored_configurations[self.current_config_id]
        # Assuming the configuration object stores the field in .field_data
        field = cfg.field_data
        plt.figure(figsize=(5,5))
        plt.imshow(np.abs(field), cmap='viridis', origin='lower')
        plt.title(f"Ephaptic Field Amplitude\nID: {self.current_config_id}")
        plt.colorbar(label="Amplitude")
        plt.axis('off')
        plt.show()

    def save_config(self):
        if not self.current_config_id:
            messagebox.showerror("Error", "No configuration to save!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files","*.pkl")])
        if not path:
            return
        cfg = self.storage.stored_configurations[self.current_config_id]
        with open(path, 'wb') as f:
            pickle.dump(cfg, f)
        messagebox.showinfo("Saved", f"Configuration saved to:\n{path}")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files","*.pkl")])
        if not path:
            return
        with open(path, 'rb') as f:
            cfg = pickle.load(f)
        cid = cfg.metadata.get('config_id') or f"restored_{len(self.storage.stored_configurations):04d}"
        self.storage.stored_configurations[cid] = cfg
        self.current_config_id = cid
        messagebox.showinfo("Loaded", f"Configuration loaded as:\n{cid}")

if __name__ == "__main__":
    root = tk.Tk()
    EphapticCompressorGUI(root)
    root.mainloop()

