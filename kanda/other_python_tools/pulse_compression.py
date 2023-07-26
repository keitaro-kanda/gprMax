import re

import matplotlib.pyplot as plt
import numpy as np

sample_rate = 1e3 # 1kHz
tmax = 15
t_simulation = np.arange(0, tmax, 1/sample_rate)
obserbed_data = np.zeros_like(t_simulation)

# ===入力電波===
freq = 10 # 10 Hz
wave_duration = 1 # 1s
t_transmission = np.arange(0, wave_duration, 1/sample_rate)
transmission = np.cos(2 * np.pi * freq * t_transmission)
# transmissionをobserbed_dataに入れる
obserbed_data[:len(transmission)] = transmission


# ===反射波(2s間隔)===
t_start = 5 # 5s
# 反射波の間隔
t_interval_2 = 2*wave_duration # 2s
t_reflection_2 = np.arange(t_start, tmax, t_interval_2)

t_interval_15 = 1.5*wave_duration # 1.5s
t_reflection_15 = np.arange(t_start, tmax, t_interval_15)

attenuation = 0.8 # 伝搬中の減衰率
reflection_rate = 0.2 # 反射率
reflected_wave_2 = np.zeros_like(t_simulation)
reflected_wave_15 = np.zeros_like(t_simulation)

for t in t_reflection_2:
    amp_rate = reflection_rate * attenuation**(t - t_start) # 振幅の減衰率を計算
    reflected_wave_2[int(t*sample_rate):int((t+wave_duration)*sample_rate)] = transmission * amp_rate

for t in t_reflection_15:
    amp_rate = reflection_rate * attenuation**(t - t_start) # 振幅の減衰率を計算
    reflected_wave_15[int(t*sample_rate):int((t+wave_duration)*sample_rate)] = transmission * amp_rate

# 白色ノイズを加える
SNR = 10 # [dB]
white_noise = np.random.randn(len(t_simulation)) * np.std(transmission) / np.power(10, SNR/20)
reflected_wave_noise_2 = reflected_wave_2 + white_noise
reflected_wave_noise_15 = reflected_wave_15 + white_noise

# ===相互相関処理===
correlation_2 = np.correlate(reflected_wave_2, transmission, mode='same')
correlation_2 /= np.amax(correlation_2) # 正規化
correlation_15 = np.correlate(reflected_wave_15, transmission, mode='same')
correlation_15 /= np.amax(correlation_15) # 正規化


# ===プロット===
fig = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='w')

# 1つ目のグラフ
plt.subplot(2, 2, 1)
plt.plot(t_simulation, reflected_wave_noise_2, label='reflected wave+' + str(SNR) + 'dB white noisse')
plt.plot(t_simulation, obserbed_data, label='transmission')
plt.plot(t_simulation, reflected_wave_2, label='reflected wave')

plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# 2つ目のグラフ
plt.subplot(2, 2, 3)
plt.plot(t_simulation, correlation_2, label='correlation')

plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# 3つ目のグラフ
plt.subplot(2, 2, 2)
plt.plot(t_simulation, reflected_wave_noise_15, label='reflected wave+' + str(SNR) + 'dB white noisse')
plt.plot(t_simulation, obserbed_data, label='transmission')
plt.plot(t_simulation, reflected_wave_15, label='reflected wave')

plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# 4つ目のグラフ
plt.subplot(2, 2, 4)
plt.plot(t_simulation, correlation_15, label='correlation')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.show()

