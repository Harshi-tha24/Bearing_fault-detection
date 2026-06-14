import numpy as np
import soundfile as sf
import os

# Create directory if it doesn't exist
os.makedirs('./test_audio_samples_extended', exist_ok=True)

sample_rate = 16000
duration = 5  # seconds
samples = duration * sample_rate

print("Generating diverse bearing audio samples...")

# ============ NORMAL BEARING SAMPLES (6 variations) ============
print("\n✓ Generating NORMAL bearing samples (6 variations)...")

# Normal 1: Clean low-frequency vibration
normal_1 = 0.03 * np.sin(2 * np.pi * 50 * np.linspace(0, duration, samples))
normal_1 += 0.02 * np.sin(2 * np.pi * 120 * np.linspace(0, duration, samples))
normal_1 += 0.005 * np.random.normal(0, 1, samples)
sf.write('./test_audio_samples_extended/normal_bearing_1_clean.wav', normal_1, sample_rate)

# Normal 2: Moderate frequency mix
normal_2 = 0.025 * np.sin(2 * np.pi * 60 * np.linspace(0, duration, samples))
normal_2 += 0.015 * np.sin(2 * np.pi * 150 * np.linspace(0, duration, samples))
normal_2 += 0.008 * np.random.normal(0, 1, samples)
sf.write('./test_audio_samples_extended/normal_bearing_2_moderate.wav', normal_2, sample_rate)

# Normal 3: High-frequency clean
normal_3 = 0.02 * np.sin(2 * np.pi * 100 * np.linspace(0, duration, samples))
normal_3 += 0.01 * np.sin(2 * np.pi * 200 * np.linspace(0, duration, samples))
normal_3 += 0.003 * np.random.normal(0, 1, samples)
sf.write('./test_audio_samples_extended/normal_bearing_3_quiet.wav', normal_3, sample_rate)

# Normal 4: Variable frequency sweep
freq_sweep = np.linspace(40, 100, samples)
phase = 2 * np.pi * np.cumsum(freq_sweep) / sample_rate
normal_4 = 0.03 * np.sin(phase)
normal_4 += 0.004 * np.random.normal(0, 1, samples)
sf.write('./test_audio_samples_extended/normal_bearing_4_sweep.wav', normal_4, sample_rate)

# Normal 5: Multi-frequency stable
t = np.linspace(0, duration, samples)
normal_5 = 0.02 * np.sin(2 * np.pi * 45 * t)
normal_5 += 0.015 * np.sin(2 * np.pi * 90 * t)
normal_5 += 0.01 * np.sin(2 * np.pi * 180 * t)
normal_5 += 0.005 * np.random.normal(0, 1, samples)
sf.write('./test_audio_samples_extended/normal_bearing_5_stable.wav', normal_5, sample_rate)

# Normal 6: Steady baseline
normal_6 = 0.035 * np.sin(2 * np.pi * 75 * np.linspace(0, duration, samples))
normal_6 += 0.006 * np.random.normal(0, 1, samples)
sf.write('./test_audio_samples_extended/normal_bearing_6_baseline.wav', normal_6, sample_rate)

# ============ FAULTY BEARING SAMPLES (6 variations) ============
print("✓ Generating FAULTY bearing samples (6 variations)...")

# Faulty 1: High-impact impulses with noise
faulty_1 = np.zeros(samples)
impulse_positions = np.linspace(0, samples, 50, dtype=int)[1:-1]
for pos in impulse_positions:
    impulse_width = int(0.01 * sample_rate)  # 10ms impulse
    window_start = max(0, pos - impulse_width // 2)
    window_end = min(samples, pos + impulse_width // 2)
    faulty_1[window_start:window_end] += 0.5 * np.hamming(window_end - window_start)
faulty_1 += 0.1 * np.random.normal(0, 1, samples)
faulty_1 = np.clip(faulty_1, -1, 1)
sf.write('./test_audio_samples_extended/faulty_bearing_1_impacts.wav', faulty_1, sample_rate)

# Faulty 2: Oscillating fault signature
t = np.linspace(0, duration, samples)
faulty_2 = 0.5 * np.sin(2 * np.pi * 10 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))  # Modulated
faulty_2 += 0.15 * np.random.normal(0, 1, samples)
faulty_2 = np.clip(faulty_2, -1, 1)
sf.write('./test_audio_samples_extended/faulty_bearing_2_oscillating.wav', faulty_2, sample_rate)

# Faulty 3: Spiky random impulses
faulty_3 = 0.08 * np.random.normal(0, 1, samples)
spike_positions = np.random.choice(samples, int(samples/1000), replace=False)
spike_positions.sort()
for pos in spike_positions:
    spike_width = int(0.005 * sample_rate)  # 5ms spike
    window_start = max(0, pos - spike_width // 2)
    window_end = min(samples, pos + spike_width // 2)
    faulty_3[window_start:window_end] += np.random.choice([-1, 1]) * 0.8
faulty_3 = np.clip(faulty_3, -1, 1)
sf.write('./test_audio_samples_extended/faulty_bearing_3_spiky.wav', faulty_3, sample_rate)

# Faulty 4: Low-frequency rumble with high-frequency noise
faulty_4 = 0.4 * np.sin(2 * np.pi * 8 * np.linspace(0, duration, samples))  # Deep rumble
faulty_4 += 0.25 * np.sin(2 * np.pi * 4000 * np.linspace(0, duration, samples))  # High-freq noise
faulty_4 += 0.1 * np.random.normal(0, 1, samples)
faulty_4 = np.clip(faulty_4, -1, 1)
sf.write('./test_audio_samples_extended/faulty_bearing_4_rumble.wav', faulty_4, sample_rate)

# Faulty 5: Decaying oscillation (bearing degradation pattern)
t = np.linspace(0, duration, samples)
decay_envelope = np.exp(-0.5 * t)
faulty_5 = 0.6 * decay_envelope * np.sin(2 * np.pi * 500 * t)
faulty_5 += 0.3 * (1 - decay_envelope) * np.random.normal(0, 1, samples)
faulty_5 = np.clip(faulty_5, -1, 1)
sf.write('./test_audio_samples_extended/faulty_bearing_5_decay.wav', faulty_5, sample_rate)

# Faulty 6: Chirp with high-amplitude noise
t = np.linspace(0, duration, samples)
f0, f1 = 100, 2000
phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
faulty_6 = 0.5 * np.sin(phase)
faulty_6 += 0.2 * np.random.normal(0, 1, samples)
faulty_6 = np.clip(faulty_6, -1, 1)
sf.write('./test_audio_samples_extended/faulty_bearing_6_chirp.wav', faulty_6, sample_rate)

print("\n" + "="*60)
print("✅ Audio generation complete!")
print("="*60)
print("\n📁 Generated samples in './test_audio_samples_extended/':")
print("\n✅ NORMAL BEARINGS (6 samples):")
print("  1. normal_bearing_1_clean.wav - Clean low-frequency vibration")
print("  2. normal_bearing_2_moderate.wav - Moderate frequency mix")
print("  3. normal_bearing_3_quiet.wav - High-frequency clean")
print("  4. normal_bearing_4_sweep.wav - Frequency sweep pattern")
print("  5. normal_bearing_5_stable.wav - Multi-frequency stable")
print("  6. normal_bearing_6_baseline.wav - Steady baseline")

print("\n⚠️  FAULTY BEARINGS (6 samples):")
print("  1. faulty_bearing_1_impacts.wav - High-impact impulses")
print("  2. faulty_bearing_2_oscillating.wav - Oscillating fault signature")
print("  3. faulty_bearing_3_spiky.wav - Spiky random impulses")
print("  4. faulty_bearing_4_rumble.wav - Low-frequency rumble")
print("  5. faulty_bearing_5_decay.wav - Decaying oscillation")
print("  6. faulty_bearing_6_chirp.wav - Chirp with noise")
print("\nTotal: 12 diverse samples for testing bearing fault detection!")
