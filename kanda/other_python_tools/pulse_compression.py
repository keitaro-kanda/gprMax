import re

import matplotlib.pyplot as plt
import numpy as np

sample_rate = 1e3 # 1kHz
tmax = 15
t_simulation = np.arange(0, tmax, 1/sample_rate)


# ===入射波の作成===
def input_wave():

    transmission_inpulse = np.zeros_like(t_simulation)
    transmission_chirp = np.zeros_like(t_simulation)

    freq = 10 # 10 Hz
    wave_duration = 1 # 1s
    t_transmission = np.arange(0, wave_duration, 1/sample_rate) # puse duration

    # インパルス
    inpulse = np.cos(2 * np.pi * freq * t_transmission)
    transmission_inpulse[:len(inpulse)] = inpulse

    # チャープ
    chirp = np.cos(10 *np.pi * t_transmission**2)
    transmission_chirp[:len(chirp)] = chirp

    return wave_duration, transmission_inpulse, transmission_chirp, inpulse, chirp

wave_duration, transmission_inpulse, transmission_chirp, inpulse, chirp = input_wave()



# ===反射波の作成===
def make_reflected_wave(interval, input_type):

    interval = 2 # 反射波の到来間隔
    input_type = transmission_inpulse # 入射波の種類
    reflected_wave = np.zeros_like(t_simulation)

    t_start = 5 # 最初の反射波の到来時刻
    reflection_interval = interval*wave_duration # 反射波の到来間隔
    reflection_time = np.arange(t_start, tmax, reflection_interval) # 反射波の到来時刻

    attenuation = 0.8 # 伝搬中の減衰率
    reflection_rate = 0.2 # 反射率

    for t in reflection_time:
        amp_rate = reflection_rate * attenuation**(t - reflection_time) # 振幅の減衰率を計算
        if input_type == 'transmission_inpulse':
            reflected_wave[int(t*sample_rate): int((t+wave_duration)*sample_rate)] = inpulse * amp_rate
        elif input_type == 'transmission_chirp':
            reflected_wave[int(t*sample_rate): int((t+wave_duration)*sample_rate)] = chirp * amp_rate


    # 白色ノイズを加える
    SNR = 15 # [dB]

    white_noise = np.random.randn(len(t_simulation)) * np.std(input_type) / np.power(10, SNR/20)
    reflected_wave_noise = reflected_wave + white_noise

    return reflected_wave, reflected_wave_noise, SNR, reflection_time

reflected_wave_1 = make_reflected_wave(2, transmission_inpulse)[0]


# 2種類のinterval、2種類のinput_typeについてreflected_waveを計算しplotする
interval_list = [2, 1.5]
input_type_list = [transmission_inpulse, transmission_chirp]

def calc_correlation(input_type, reflect_type):
    correlation = np.correlate(reflect_type, input_type, mode='same')
    correlation /= np.amax(correlation) # 正規化
    return correlation

correlation_1 = calc_correlation(transmission_inpulse, make_reflected_wave(2, transmission_inpulse)[1])
correlation_2 = calc_correlation(transmission_inpulse, make_reflected_wave(1.5, transmission_inpulse)[1])
correlation_3 = calc_correlation(transmission_inpulse, make_reflected_wave(2, transmission_chirp)[1])
correlation_4 = calc_correlation(transmission_inpulse, make_reflected_wave(1.5, transmission_chirp)[1])


# ===plot===
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(t_simulation, make_reflected_wave(2, transmission_inpulse)[1], label='reflected wave with noise')
ax1.plot(t_simulation, transmission_inpulse, label='transmission')
ax1.plot(t_simulation, make_reflected_wave(2, transmission_inpulse)[0], label='reflected wave')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_simulation, make_reflected_wave(1.5, transmission_inpulse)[1], label='reflected wave with noise')
ax2.plot(t_simulation, transmission_inpulse, label='transmission')
ax2.plot(t_simulation, make_reflected_wave(1.5, transmission_inpulse)[0], label='reflected wave')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_simulation, correlation_1, label='correlation')

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t_simulation, correlation_2, label='correlation')

plt.show()
""""
# 整合フィルタ処理
def matched_filter():
    # transmissionのフーリエ変換
    transmission_fft = np.fft.fft(transmission)
    # transmissionのパワースペクトル
    transmission_power = np.abs(transmission_fft) ** 2
    
    # reflected_wave_noise_15のフーリエ変換
    reflected_wave_noise_15_fft = np.fft.fft(reflected_wave_noise_15)

    return plt


matched_filter()
"""