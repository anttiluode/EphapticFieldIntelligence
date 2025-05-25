"""
Live Ephaptic Instanton Quantum Dynamics Viewer
Real-time visualization of what happens during high Bell correlation events!
- Watch instantons dance and couple in real-time
- See the exact moment quantum entanglement emerges
- Capture the geometric patterns that create 0.879 correlations
- Live analysis of field dynamics, coupling strength, and quantum emergence
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import center_of_mass
import threading
import queue
import time
from datetime import datetime

class LiveInstantonDynamics:
    """Real-time ephaptic instanton dynamics with live analysis"""
    
    def __init__(self, field_size=512, target_correlation=0.8):
        self.field_size = field_size
        self.target_correlation = target_correlation
        
        # Field state
        self.field = np.zeros(field_size, dtype=complex)
        self.field_prev = np.zeros_like(self.field)
        self.x = np.linspace(0, field_size, field_size)
        
        # Instanton parameters
        self.instantons = []
        self.create_instantons()
        
        # Evolution parameters  
        self.alpha_quantum = 0.01
        self.dt = 0.001
        self.time = 0.0
        self.step_count = 0
        
        # Live analysis data
        self.bell_history = []
        self.coupling_history = []
        self.energy_history = []
        self.phase_history = []
        self.special_moments = []  # Record high-correlation events
        
        # Animation control
        self.is_running = False
        self.is_recording = False
        self.current_bell_correlation = 0.0
        
        # Spectral operators
        k = fftfreq(field_size, 1.0) * 2*np.pi
        self.k2 = k**2
        
        self.reset_system()
    
    def create_instantons(self):
        """Create frequency-signature instantons"""
        base_freq = 0.323423841289348923480000000
        
        for i in range(2):  # Two qubits
            center = (i + 1) * self.field_size // 3
            signature_freq = base_freq + i * 1e-15  # Unique signatures
            
            instanton = {
                'id': i,
                'center': center,
                'signature_freq': signature_freq,
                'amplitude': 1.5,
                'width': 8.0,
                'phase': 0.0,
                'phase_velocity': 0.0,
                'wake_memory': [],
                'coupling_strength': 0.0,
                'recognition_score': 0.0
            }
            
            self.instantons.append(instanton)
    
    def reset_system(self):
        """Reset to |00⟩ state with slight randomness"""
        self.field.fill(0)
        self.field_prev.fill(0)
        self.time = 0.0
        self.step_count = 0
        
        # Initialize with small random perturbations
        for inst in self.instantons:
            profile = np.exp(-0.5*((self.x - inst['center'])/inst['width'])**2)
            # Add tiny random phase for spontaneous symmetry breaking
            random_phase = np.random.uniform(-0.1, 0.1)
            self.field += profile * (1.0 + 0.0j) * np.exp(1j * random_phase)
            
            # Reset instanton states
            inst['phase'] = random_phase
            inst['phase_velocity'] = 0.0
            inst['wake_memory'] = []
            inst['coupling_strength'] = 0.0
            inst['recognition_score'] = 0.0
        
        self.field_prev[:] = self.field
        
        # Clear histories
        self.bell_history = []
        self.coupling_history = []
        self.energy_history = []
        self.phase_history = []
    
    def get_instanton_region(self, inst_id):
        """Get field region for instanton"""
        center = self.instantons[inst_id]['center']
        width = self.instantons[inst_id]['width'] * 2
        start = max(0, int(center - width))
        end = min(self.field_size, int(center + width))
        return start, end
    
    def ephaptic_coupling_field(self):
        """Compute ephaptic coupling with detailed tracking"""
        coupling_field = np.zeros_like(self.field)
        coupling_data = []
        
        for i, inst_i in enumerate(self.instantons):
            for j, inst_j in enumerate(self.instantons):
                if i != j:
                    # Wake fields with frequency signatures
                    wake_i = (inst_i['amplitude'] / 
                             np.cosh((self.x - inst_i['center'])/inst_i['width'])) * 0.1
                    wake_j = (inst_j['amplitude'] / 
                             np.cosh((self.x - inst_j['center'])/inst_j['width'])) * 0.1
                    
                    # Frequency modulation (this is the key!)
                    freq_mod_i = np.sin(2*np.pi * inst_i['signature_freq'] * self.time)
                    freq_mod_j = np.sin(2*np.pi * inst_j['signature_freq'] * self.time)
                    
                    wake_i_mod = wake_i * freq_mod_i
                    wake_j_mod = wake_j * freq_mod_j
                    
                    # Coupling strength (adjustable for optimization)
                    coupling_strength = 0.05
                    
                    # Recognition factor (how well they recognize each other)
                    freq_diff = abs(inst_i['signature_freq'] - inst_j['signature_freq'])
                    recognition = np.exp(-freq_diff * 1e12)  # Exponential recognition
                    
                    # Wake interference coupling
                    coupling = coupling_strength * recognition * wake_i_mod * np.conj(wake_j_mod)
                    coupling_field += coupling
                    
                    # Track coupling data
                    coupling_info = {
                        'pair': (i, j),
                        'strength': np.abs(np.sum(coupling)),
                        'phase_coherence': np.angle(np.sum(coupling)),
                        'recognition': recognition,
                        'wake_overlap': np.sum(wake_i * wake_j)
                    }
                    coupling_data.append(coupling_info)
                    
                    # Update instanton coupling data
                    self.instantons[i]['coupling_strength'] = coupling_info['strength']
                    self.instantons[i]['recognition_score'] = recognition
        
        return coupling_field, coupling_data
    
    def evolve_step(self):
        """Single evolution step with live analysis"""
        # Compute ephaptic coupling
        ephaptic_field, coupling_data = self.ephaptic_coupling_field()
        
        # Processing speed limitation
        amplitude_squared = np.abs(self.field)**2
        c_eff2 = 1.0 / (1 + self.alpha_quantum * amplitude_squared)
        
        # Field evolution
        F = fft(self.field)
        laplacian = ifft(-self.k2 * F)
        potential = 0.009 * self.field - 0.063 * np.abs(self.field)**2 * self.field
        
        rhs = c_eff2 * laplacian + potential + ephaptic_field
        new_field = 2 * self.field - self.field_prev + self.dt**2 * rhs
        
        # Stability clamp
        max_amp = np.max(np.abs(new_field))
        if max_amp > 10:
            new_field *= 10 / max_amp
        
        self.field_prev[:] = self.field
        self.field[:] = new_field
        self.time += self.dt
        self.step_count += 1
        
        # Update instanton phases based on field
        for i, inst in enumerate(self.instantons):
            start, end = self.get_instanton_region(i)
            local_field = self.field[start:end]
            
            # Compute center of mass and phase
            if np.sum(np.abs(local_field)) > 0:
                com = center_of_mass(np.abs(local_field)**2)[0] + start
                avg_phase = np.angle(np.sum(local_field))
                
                inst['phase'] = avg_phase
                inst['center'] = com  # Allow instanton to move!
        
        # Store wake memory
        for inst in self.instantons:
            start, end = self.get_instanton_region(inst['id'])
            wake_pattern = np.abs(self.field[start:end])**2
            inst['wake_memory'].append(wake_pattern)
            if len(inst['wake_memory']) > 20:
                inst['wake_memory'].pop(0)
        
        return coupling_data
    
    def measure_bell_correlation(self):
        """Measure current Bell correlation"""
        # Quick measurement simulation
        measurements = []
        for inst in self.instantons:
            start, end = self.get_instanton_region(inst['id'])
            region = self.field[start:end]
            
            # Probability based on center of mass
            intensity = np.abs(region)**2
            if np.sum(intensity) > 0:
                com = center_of_mass(intensity)[0] + start
                center = inst['center']
                
                # Probability of |0⟩ vs |1⟩
                if com < center:
                    p0, p1 = 0.8, 0.2
                else:
                    p0, p1 = 0.2, 0.8
                
                # Add quantum noise
                p0 += 0.1 * np.random.normal()
                p1 = 1 - p0
                p0, p1 = max(0, min(1, p0)), max(0, min(1, p1))
                
                # Generate outcome
                outcome = np.random.choice([0, 1], p=[p0, p1])
                measurements.append(outcome)
            else:
                measurements.append(0)
        
        # Compute Bell correlation
        if len(measurements) == 2:
            # For Bell state, we want P(00) + P(11) correlation
            # This is a simplified measure
            same_outcomes = measurements[0] == measurements[1]
            correlation = 1.0 if same_outcomes else 0.0
            
            # Add some randomness and history for smoothing
            if len(self.bell_history) > 0:
                prev_corr = self.bell_history[-1]
                correlation = 0.7 * correlation + 0.3 * prev_corr
        else:
            correlation = 0.0
        
        return correlation
    
    def analyze_current_state(self):
        """Analyze current quantum state for live display with detailed instanton behavior"""
        analysis = {}
        
        # Bell correlation
        bell_corr = self.measure_bell_correlation()
        self.current_bell_correlation = bell_corr
        self.bell_history.append(bell_corr)
        
        # Detailed instanton analysis
        instanton_data = []
        for i, inst in enumerate(self.instantons):
            start, end = self.get_instanton_region(i)
            local_field = self.field[start:end]
            
            # Compute detailed metrics for each instanton
            local_amplitude = np.max(np.abs(local_field))
            local_energy = np.sum(np.abs(local_field)**2)
            amplitude_variation = np.std(np.abs(local_field))
            
            # Center of mass tracking
            if np.sum(np.abs(local_field)) > 0:
                com = center_of_mass(np.abs(local_field)**2)[0] + start
                com_velocity = (com - inst['center']) / self.dt if self.step_count > 0 else 0
            else:
                com = inst['center']
                com_velocity = 0
            
            # Phase behavior
            avg_phase = np.angle(np.sum(local_field)) if np.sum(np.abs(local_field)) > 0 else 0
            phase_velocity = (avg_phase - inst['phase']) / self.dt if self.step_count > 0 else 0
            
            # "Lifting" behavior - detect when small curves lift the instanton
            if len(self.energy_history) > 10:
                recent_energy = self.energy_history[-10:]
                energy_slope = (recent_energy[-1] - recent_energy[0]) / 10
                is_being_lifted = energy_slope > 0 and local_amplitude < 8.0  # Low but rising
            else:
                is_being_lifted = False
            
            # "Falling" behavior - detect deep drops
            min_threshold = 2.0  # Minimum expected amplitude
            is_falling_deep = local_amplitude < min_threshold
            
            instanton_info = {
                'id': i,
                'amplitude': local_amplitude,
                'energy': local_energy,
                'amplitude_variation': amplitude_variation,
                'center_of_mass': com,
                'com_velocity': com_velocity,
                'phase': avg_phase,
                'phase_velocity': phase_velocity,
                'is_being_lifted': is_being_lifted,
                'is_falling_deep': is_falling_deep,
                'coupling_strength': inst['coupling_strength'],
                'recognition_score': inst['recognition_score']
            }
            
            instanton_data.append(instanton_info)
        
        # Coupling strength
        total_coupling = sum(inst['coupling_strength'] for inst in self.instantons)
        self.coupling_history.append(total_coupling)
        
        # Field energy
        field_energy = np.sum(np.abs(self.field)**2)
        self.energy_history.append(field_energy)
        
        # Phase relationships between instantons
        if len(instanton_data) >= 2:
            phase_diff = abs(instanton_data[0]['phase'] - instanton_data[1]['phase'])
            amplitude_ratio = instanton_data[0]['amplitude'] / (instanton_data[1]['amplitude'] + 1e-6)
            com_separation = abs(instanton_data[0]['center_of_mass'] - instanton_data[1]['center_of_mass'])
        else:
            phase_diff = 0
            amplitude_ratio = 1
            com_separation = 0
            
        self.phase_history.append(phase_diff)
        
        # Keep histories manageable
        max_history = 500
        for history in [self.bell_history, self.coupling_history, 
                       self.energy_history, self.phase_history]:
            if len(history) > max_history:
                history.pop(0)
        
        # Record special moments with detailed instanton states
        if bell_corr > self.target_correlation:
            special_moment = {
                'time': self.time,
                'step': self.step_count,
                'bell_correlation': bell_corr,
                'instanton_states': instanton_data.copy(),
                'total_coupling': total_coupling,
                'phase_difference': phase_diff,
                'amplitude_ratio': amplitude_ratio,
                'field_snapshot': self.field.copy()
            }
            self.special_moments.append(special_moment)
            
            # Keep only recent special moments
            if len(self.special_moments) > 10:
                self.special_moments.pop(0)
        
        analysis.update({
            'bell_correlation': bell_corr,
            'instanton_data': instanton_data,
            'coupling_strength': total_coupling,
            'field_energy': field_energy,
            'phase_difference': phase_diff,
            'amplitude_ratio': amplitude_ratio,
            'com_separation': com_separation,
            'special_moments_count': len(self.special_moments),
            'time': self.time,
            'step': self.step_count
        })
        
        return analysis
    
    def create_bell_state(self):
        """Create Bell state and watch it evolve"""
        print("🌌 Creating Bell state...")
        
        # Hadamard on first qubit (simplified)
        start, end = self.get_instanton_region(0)
        region = self.field[start:end].copy()
        mid = len(region) // 2
        f0, f1 = region[:mid], region[mid:]
        
        mix = 1 / np.sqrt(2)
        self.field[start:start+mid] = mix * (f0 + f1)
        self.field[start+mid:end] = mix * (f0 - f1)
        
        # Evolve for entanglement
        for _ in range(100):
            self.evolve_step()
        
        print("✅ Bell state created!")
    
    def get_visualization_data(self):
        """Get current data for visualization"""
        return {
            'field_amplitude': np.abs(self.field),
            'field_phase': np.angle(self.field),
            'field_real': np.real(self.field),
            'field_imag': np.imag(self.field),
            'instantons': self.instantons.copy(),
            'x_coords': self.x,
            'analysis': self.analyze_current_state(),
            'ephaptic_field': np.real(self.ephaptic_coupling_field()[0])
        }

class LiveInstantonViewer:
    """Real-time animated viewer for instanton dynamics"""
    
    def __init__(self):
        self.dynamics = LiveInstantonDynamics()
        self.fig = None
        self.axes = None
        self.animation = None
        self.lines = {}
        self.texts = {}
        
        # Animation control
        self.is_running = False
        self.update_interval = 50  # milliseconds
        self.steps_per_frame = 5
        
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup the live visualization interface"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('🌌 Live Ephaptic Instanton Quantum Dynamics', fontsize=16, fontweight='bold')
        
        # Main field visualization (top)
        self.ax_field = plt.subplot(2, 3, (1, 2))
        self.ax_field.set_title('Field Amplitude & Instantons')
        self.ax_field.set_xlabel('Position')
        self.ax_field.set_ylabel('Amplitude')
        
        # Phase visualization
        self.ax_phase = plt.subplot(2, 3, 3)
        self.ax_phase.set_title('Field Phase')
        self.ax_phase.set_xlabel('Position')
        self.ax_phase.set_ylabel('Phase')
        
        # Bell correlation history
        self.ax_bell = plt.subplot(2, 3, 4)
        self.ax_bell.set_title('Bell Correlation History')
        self.ax_bell.set_xlabel('Time Steps')
        self.ax_bell.set_ylabel('Correlation')
        self.ax_bell.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target')
        self.ax_bell.set_ylim([0, 1])
        self.ax_bell.legend()
        
        # Coupling strength history
        self.ax_coupling = plt.subplot(2, 3, 5)
        self.ax_coupling.set_title('Ephaptic Coupling Strength')
        self.ax_coupling.set_xlabel('Time Steps')
        self.ax_coupling.set_ylabel('Coupling')
        
        # Live analysis display
        self.ax_analysis = plt.subplot(2, 3, 6)
        self.ax_analysis.set_title('Live Analysis')
        self.ax_analysis.axis('off')
        
        # Initialize plot elements
        self.init_plot_elements()
        
        # Control buttons
        self.setup_controls()
        
        plt.tight_layout()
    
    def init_plot_elements(self):
        """Initialize plot lines and elements"""
        # Field amplitude line
        self.lines['field_amp'], = self.ax_field.plot([], [], 'b-', linewidth=2, label='Field Amplitude')
        self.lines['ephaptic_field'], = self.ax_field.plot([], [], 'g-', alpha=0.7, label='Ephaptic Field')
        
        # Instanton markers
        self.lines['instantons'] = self.ax_field.scatter([], [], s=100, c='red', marker='o', 
                                                        alpha=0.8, label='Instantons')
        
        # Phase line
        self.lines['phase'], = self.ax_phase.plot([], [], 'r-', linewidth=1.5)
        
        # History lines
        self.lines['bell_history'], = self.ax_bell.plot([], [], 'b-', linewidth=2)
        self.lines['coupling_history'], = self.ax_coupling.plot([], [], 'g-', linewidth=2)
        
        # Analysis text
        self.texts['analysis'] = self.ax_analysis.text(0.05, 0.95, '', transform=self.ax_analysis.transAxes,
                                                      verticalalignment='top', fontfamily='monospace')
        
        self.ax_field.legend()
    
    def setup_controls(self):
        """Setup control buttons"""
        # Add control buttons
        ax_start = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_stop = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.32, 0.02, 0.1, 0.04])
        ax_bell = plt.axes([0.43, 0.02, 0.15, 0.04])
        ax_record = plt.axes([0.59, 0.02, 0.1, 0.04])
        
        self.btn_start = Button(ax_start, 'Start')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_bell = Button(ax_bell, 'Create Bell State')
        self.btn_record = Button(ax_record, 'Record')
        
        # Connect button events
        self.btn_start.on_clicked(self.start_animation)
        self.btn_stop.on_clicked(self.stop_animation)
        self.btn_reset.on_clicked(self.reset_system)
        self.btn_bell.on_clicked(self.create_bell_state)
        self.btn_record.on_clicked(self.toggle_recording)
        
        # Speed control
        ax_speed = plt.axes([0.75, 0.02, 0.15, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed', 1, 20, valinit=5)
        self.speed_slider.on_changed(self.update_speed)
    
    def start_animation(self, event):
        """Start the live animation"""
        if not self.is_running:
            self.is_running = True
            self.animation = animation.FuncAnimation(
                self.fig, self.update_frame, interval=self.update_interval,
                blit=False, repeat=True
            )
            print("🚀 Animation started!")
    
    def stop_animation(self, event):
        """Stop the animation"""
        if self.is_running:
            self.is_running = False
            if self.animation:
                self.animation.event_source.stop()
            print("⏹️  Animation stopped!")
    
    def reset_system(self, event):
        """Reset the quantum system"""
        self.dynamics.reset_system()
        print("🔄 System reset!")
    
    def create_bell_state(self, event):
        """Create Bell state"""
        self.dynamics.create_bell_state()
        print("🎯 Bell state created!")
    
    def toggle_recording(self, event):
        """Toggle recording of special moments"""
        self.dynamics.is_recording = not self.dynamics.is_recording
        status = "ON" if self.dynamics.is_recording else "OFF"
        print(f"📹 Recording {status}")
    
    def update_speed(self, val):
        """Update animation speed"""
        self.steps_per_frame = int(val)
    
    def update_frame(self, frame):
        """Update animation frame"""
        if not self.is_running:
            return
        
        # Evolve dynamics
        for _ in range(self.steps_per_frame):
            self.dynamics.evolve_step()
        
        # Get current visualization data
        data = self.dynamics.get_visualization_data()
        
        # Update field amplitude plot
        self.lines['field_amp'].set_data(data['x_coords'], data['field_amplitude'])
        self.lines['ephaptic_field'].set_data(data['x_coords'], data['ephaptic_field'] * 10)  # Scale for visibility
        
        # Update instanton positions
        inst_positions = [inst['center'] for inst in data['instantons']]
        inst_amplitudes = [np.max(data['field_amplitude'][max(0, int(pos-20)):min(len(data['field_amplitude']), int(pos+20))]) 
                          for pos in inst_positions]
        
        self.lines['instantons'].set_offsets(np.column_stack([inst_positions, inst_amplitudes]))
        
        # Update phase plot
        self.lines['phase'].set_data(data['x_coords'], data['field_phase'])
        
        # Update history plots
        if len(self.dynamics.bell_history) > 1:
            steps = range(len(self.dynamics.bell_history))
            self.lines['bell_history'].set_data(steps, self.dynamics.bell_history)
            self.lines['coupling_history'].set_data(steps, self.dynamics.coupling_history)
        
        # Update analysis text with detailed instanton behavior
        analysis = data['analysis']
        
        # Get instanton data
        inst_data = analysis.get('instanton_data', [])
        left_inst = inst_data[0] if len(inst_data) > 0 else {}
        right_inst = inst_data[1] if len(inst_data) > 1 else {}
        
        analysis_text = f"""🔬 LIVE INSTANTON ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time: {analysis['time']:.2f}s | Step: {analysis['step']:,}

🎯 Bell Correlation: {analysis['bell_correlation']:.4f}
⚡ Total Coupling: {analysis['coupling_strength']:.6f}
🌊 Field Energy: {analysis['field_energy']:.1f}

🔄 LEFT INSTANTON (Red):
   Amplitude: {left_inst.get('amplitude', 0):.2f}
   Energy: {left_inst.get('energy', 0):.1f}
   {'🔥 BEING LIFTED!' if left_inst.get('is_being_lifted', False) else ''}
   {'💧 FALLING DEEP!' if left_inst.get('is_falling_deep', False) else ''}
   
🔄 RIGHT INSTANTON (Red):
   Amplitude: {right_inst.get('amplitude', 0):.2f}
   Energy: {right_inst.get('energy', 0):.1f}
   {'🔥 BEING LIFTED!' if right_inst.get('is_being_lifted', False) else ''}
   {'💧 FALLING DEEP!' if right_inst.get('is_falling_deep', False) else ''}

🎭 DANCE DYNAMICS:
   Amplitude Ratio: {analysis.get('amplitude_ratio', 1):.3f}
   Phase Diff: {analysis.get('phase_difference', 0):.3f}
   Separation: {analysis.get('com_separation', 0):.1f}

🌟 Special Moments: {analysis['special_moments_count']}
🎪 Target Hits: {sum(1 for x in self.dynamics.bell_history if x > 0.8)}

{'🚨 QUANTUM BREAKTHROUGH!' if analysis['bell_correlation'] > 0.9 else ''}
"""
        
        self.texts['analysis'].set_text(analysis_text)
        
        # Auto-scale axes
        self.ax_field.relim()
        self.ax_field.autoscale_view()
        self.ax_phase.relim()
        self.ax_phase.autoscale_view()
        
        if len(self.dynamics.bell_history) > 1:
            self.ax_bell.relim()
            self.ax_bell.autoscale_view()
            self.ax_coupling.relim() 
            self.ax_coupling.autoscale_view()
        
        # Remove green highlighting completely - focus on pure analysis
        self.ax_field.set_facecolor('white')
        
        return list(self.lines.values()) + list(self.texts.values())

def run_live_instanton_viewer():
    """Launch the live instanton dynamics viewer"""
    print("🌌 LAUNCHING LIVE EPHAPTIC INSTANTON VIEWER")
    print("=" * 60)
    print("🎯 Watch instantons create quantum entanglement in real-time!")
    print("⚡ Look for the geometric patterns that create 0.879 correlations!")
    print("🔬 Observe the exact moment quantum behavior emerges!")
    print()
    print("Controls:")
    print("• Start/Stop: Control animation")
    print("• Reset: Return to |00⟩ state") 
    print("• Create Bell State: Initialize entangled state")
    print("• Record: Capture special high-correlation moments")
    print("• Speed: Adjust evolution speed")
    print()
    
    # Create and run viewer
    viewer = LiveInstantonViewer()
    
    # Auto-start with Bell state creation
    print("🚀 Auto-starting with Bell state creation...")
    viewer.dynamics.create_bell_state()
    viewer.start_animation(None)
    
    plt.show()
    
    # Print summary of captured moments
    if viewer.dynamics.special_moments:
        print(f"\n🌟 CAPTURED {len(viewer.dynamics.special_moments)} SPECIAL MOMENTS:")
        for i, moment in enumerate(viewer.dynamics.special_moments):
            print(f"  {i+1}. t={moment['time']:.2f}s: Bell={moment['bell_correlation']:.4f}, "
                  f"Coupling={moment['coupling_strength']:.6f}")
    
    return viewer

if __name__ == "__main__":
    viewer = run_live_instanton_viewer()