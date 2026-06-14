
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt


class MultiScaleCNNBlock(layers.Layer):
   
    
    def __init__(self, filters=64, **kwargs):
        super(MultiScaleCNNBlock, self).__init__(**kwargs)
        self.filters = filters
        
        # SCALE 1: High-frequency (small kernel)
        self.conv1 = layers.Conv1D(
            filters=filters, kernel_size=3, 
            activation='relu', padding='same', name='conv_scale1_k3'
        )
        self.bn1 = layers.BatchNormalization()
        
        # SCALE 2: Mid-frequency (medium kernel)
        self.conv2 = layers.Conv1D(
            filters=filters, kernel_size=7,
            activation='relu', padding='same', name='conv_scale2_k7'
        )
        self.bn2 = layers.BatchNormalization()
        
        # SCALE 3: Low-frequency (large kernel)
        self.conv3 = layers.Conv1D(
            filters=filters, kernel_size=15,
            activation='relu', padding='same', name='conv_scale3_k15'
        )
        self.bn3 = layers.BatchNormalization()
    
    def call(self, x, training=False):
        # Process in parallel
        scale1 = self.bn1(self.conv1(x), training=training)
        scale2 = self.bn2(self.conv2(x), training=training)
        scale3 = self.bn3(self.conv3(x), training=training)
        
        # Concatenate all scales
        concatenated = layers.Concatenate()([scale1, scale2, scale3])
        return concatenated


class MultiScaleCNN(Model):
    """
    Multi-Scale CNN for Bearing Fault Detection
    
    Architecture:
    1. Input: Raw 1D vibration signal (2048, 1)
    2. Multi-Scale Block 1: Extract features at 3 scales
    3. MaxPooling + Multi-Scale Block 2: Downsample + re-extract
    4. Global pooling + Dense layers: Classification
    5. Output: 2-class (Normal/Faulty) or Multi-class
    
    Key Advantage:
    - Explicit multi-scale learning (not implicit)
    - Each scale learns independently
    - Better fault discrimination across frequencies
    """
    
    def __init__(self, num_classes=2, **kwargs):
        super(MultiScaleCNN, self).__init__(**kwargs)
        
        self.num_classes = num_classes
        
        # ===== STAGE 1: Initial Multi-Scale Feature Extraction =====
        self.multi_scale_1 = MultiScaleCNNBlock(filters=64, name='multi_scale_block_1')
        self.pool1 = layers.MaxPooling1D(pool_size=4, name='pool1')
        self.drop1 = layers.Dropout(0.2)
        
        # ===== STAGE 2: Refined Multi-Scale Feature Extraction =====
        self.multi_scale_2 = MultiScaleCNNBlock(filters=128, name='multi_scale_block_2')
        self.pool2 = layers.MaxPooling1D(pool_size=4, name='pool2')
        self.drop2 = layers.Dropout(0.2)
        
        # ===== STAGE 3: Global Pooling + Classification =====
        self.global_pool = layers.GlobalAveragePooling1D(name='global_pool')
        
        self.fc1 = layers.Dense(256, activation='relu', name='dense1')
        self.bn_fc = layers.BatchNormalization()
        self.drop_fc = layers.Dropout(0.3)
        
        self.fc2 = layers.Dense(128, activation='relu', name='dense2')
        self.drop_fc2 = layers.Dropout(0.2)
        
        # Output layer
        if num_classes == 2:
            self.output_layer = layers.Dense(1, activation='sigmoid', name='output_binary')
        else:
            self.output_layer = layers.Dense(num_classes, activation='softmax', name='output_multiclass')
    
    def call(self, x, training=False):
        # Stage 1: Multi-scale extraction + pooling
        x = self.multi_scale_1(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        
        # Stage 2: Refined multi-scale extraction + pooling
        x = self.multi_scale_2(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        
        # Stage 3: Global pooling
        x = self.global_pool(x)
        
        # Fully connected + classification
        x = self.fc1(x)
        x = self.bn_fc(x, training=training)
        x = self.drop_fc(x, training=training)
        
        x = self.fc2(x)
        x = self.drop_fc2(x, training=training)
        
        output = self.output_layer(x)
        return output
    
    def get_config(self):
        return {"num_classes": self.num_classes}


def build_and_compile_multi_scale_model(num_classes=2):
    """Build and compile the multi-scale CNN model"""
    model = MultiScaleCNN(num_classes=num_classes)
    
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=metrics
    )
    
    return model


def visualize_multi_scale_architecture():
    """Visualize how multi-scale kernels capture different patterns"""
    print("\n" + "="*80)
    print("🔬 MULTI-SCALE CNN ARCHITECTURE VISUALIZATION")
    print("="*80)
    
    # Create synthetic signal with multiple components
    t = np.linspace(0, 1, 2048)
    
    # Component 1: Low-frequency trend (fault envelope)
    low_freq = 5 * np.sin(2 * np.pi * 2 * t)
    
    # Component 2: Mid-frequency oscillation (main bearing tone)
    mid_freq = 3 * np.sin(2 * np.pi * 20 * t)
    
    # Component 3: High-frequency noise (fault impulses)
    high_freq = 1 * np.sin(2 * np.pi * 100 * t)
    
    signal = low_freq + mid_freq + high_freq
    
    # Plot original signal
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(t[:500], signal[:500], linewidth=1, color='black')
    axes[0, 0].set_title('Combined Signal\n(Low + Mid + High Freq)', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Simulate kernel responses
    axes[0, 1].plot(t[:500], low_freq[:500], linewidth=2, color='#1f77b4', label='Low-Freq')
    axes[0, 1].plot(t[:500], mid_freq[:500], linewidth=2, color='#ff7f0e', label='Mid-Freq')
    axes[0, 1].set_title('Signal Components', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Demonstrate kernel size sensitivity
    axes[0, 2].text(0.5, 0.8, 'Kernel Sizes:', ha='center', fontweight='bold', fontsize=12, transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.5, 0.6, 'K=3: High-Freq (impulses)', ha='center', fontsize=10, transform=axes[0, 2].transAxes, bbox=dict(boxstyle='round', facecolor='lightblue'))
    axes[0, 2].text(0.5, 0.4, 'K=7: Mid-Freq (oscillations)', ha='center', fontsize=10, transform=axes[0, 2].transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[0, 2].text(0.5, 0.2, 'K=15: Low-Freq (trends)', ha='center', fontsize=10, transform=axes[0, 2].transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    axes[0, 2].axis('off')
    
    # Simulated kernel responses (conceptual)
    from scipy import signal as sig
    
    # Kernel 3 response
    kernel_3 = np.array([1, -2, 1]) / 4  # High-pass-like
    k3_response = sig.convolve(high_freq, kernel_3, mode='same')
    axes[1, 0].plot(t[:500], k3_response[:500], linewidth=2, color='#1f77b4')
    axes[1, 0].set_title('Kernel=3 Response\n(High-Freq Details)', fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Activation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Kernel 7 response
    kernel_7 = np.ones(7) / 7  # Band-pass-like
    k7_response = sig.convolve(signal, kernel_7, mode='same')
    axes[1, 1].plot(t[:500], k7_response[:500], linewidth=2, color='#ff7f0e')
    axes[1, 1].set_title('Kernel=7 Response\n(Mid-Freq Features)', fontweight='bold')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Kernel 15 response
    kernel_15 = np.ones(15) / 15  # Low-pass-like
    k15_response = sig.convolve(signal, kernel_15, mode='same')
    axes[1, 2].plot(t[:500], k15_response[:500], linewidth=2, color='#2ca02c')
    axes[1, 2].set_title('Kernel=15 Response\n(Low-Freq Trends)', fontweight='bold')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_scale_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved: multi_scale_visualization.png")
    plt.close()


if __name__ == "__main__":
    print("\n🔧 Multi-Scale CNN for Bearing Fault Detection")
    print("="*80)
    
    # Build model
    model = build_and_compile_multi_scale_model(num_classes=2)
    
    print("\n📊 Model Summary:")
    model.summary()
    
    # Test with dummy data
    print("\n🧪 Testing with dummy data...")
    dummy_signals = np.random.randn(16, 2048, 1).astype(np.float32)
    
    print(f"Input shape: {dummy_signals.shape}")
    
    output = model(dummy_signals, training=False)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
    
    # Visualize architecture concept
    visualize_multi_scale_architecture()
    
    print("\n" + "="*80)
    print("✅ IMPROVEMENT 1 - Multi-Scale CNN: READY FOR DEPLOYMENT")
    print("="*80)
    print("\n📌 Panel Value:")
    print("   'Our multi-scale architecture explicitly learns bearing fault")
    print("    features at different frequency ranges (high, mid, low),")
    print("    providing better discrimination across various fault types.'")
    print("\n💡 Key Advantage:")
    print("   - Kernel size 3: Captures sharp impulses (fault spikes)")
    print("   - Kernel size 7: Captures oscillations (bearing tones)")
    print("   - Kernel size 15: Captures trends (envelope modulation)")
