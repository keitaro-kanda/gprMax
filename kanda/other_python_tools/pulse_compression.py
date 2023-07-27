import re
from weakref import ref

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft

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

    reflected_wave = np.zeros_like(t_simulation)

    t_start = 5 # 最初の反射波の到来時刻
    reflection_interval = interval*wave_duration # 反射波の到来間隔
    reflection_time = np.arange(t_start, tmax, reflection_interval) # 反射波の到来時刻のリスト

    attenuation = 0.8 # 伝搬中の減衰率
    reflection_rate = 0.2 # 反射率


    for t in reflection_time:
        amp_rate = reflection_rate * attenuation**(t - t_start) # 振幅の減衰率を計算
        reflect_start = int(t*sample_rate) # 反射波の到来時刻
        reflect_end = int((t+wave_duration)*sample_rate) # 反射波の終了時刻

        if input_type == 'inpulse':
            reflected_wave[reflect_start: reflect_end] = inpulse * amp_rate
        elif input_type == 'chirp':
            reflected_wave[reflect_start: reflect_end] = chirp * amp_rate

    # 白色ノイズを加える
    SNR = 15 # [dB]

    white_noise = np.zeros_like(t_simulation)
    if input_type == 'inpulse':
        white_noise = np.random.randn(len(t_simulation)) * np.std(inpulse) / np.power(10, SNR/20)
    elif input_type == 'chirp':
        white_noise = np.random.randn(len(t_simulation)) * np.std(chirp) / np.power(10, SNR/20)
    else:
        print('input_type is inpulse or chirp')

    reflected_wave_noise = reflected_wave + white_noise

    return reflected_wave, reflected_wave_noise, SNR, reflection_time



# ===相互相関の計算===
def calc_correlation(input_array, reflect_array):
    reflect_wave = make_reflected_wave

    correlation = np.correlate(reflect_array, input_array, mode='same')
    correlation /= np.amax(correlation) # 正規化
    return correlation

correlation_1 = calc_correlation(inpulse, make_reflected_wave(2, 'inpulse')[1])
correlation_2 = calc_correlation(inpulse, make_reflected_wave(1.5, 'inpulse')[1])
correlation_3 = calc_correlation(chirp, make_reflected_wave(2, 'chirp')[1])
correlation_4 = calc_correlation(chirp, make_reflected_wave(1.5, 'chirp')[1])

def plot():
    # ===plot===
    fig, ax = plt.subplots(4, 2, figsize=(24, 12), tight_layout=True)

    ax[0, 0].plot(t_simulation, make_reflected_wave(2, 'inpulse')[1], label='reflected wave with noise')
    ax[0, 0].plot(t_simulation, transmission_inpulse, label='input wave')
    ax[0, 0].plot(t_simulation, make_reflected_wave(2, 'inpulse')[0], label='reflected wave')
    ax[0, 0].set_title('inpulse input & wide interval', size=18)
    ax[0,0].legend(fontsize=12)

    ax[1, 0].plot(t_simulation, correlation_1, label='correlation')
    ax[1, 0].legend(fontsize=12)

    ax[2, 0].plot(t_simulation, make_reflected_wave(1.5, 'inpulse')[1], label='reflected wave with noise')
    ax[2, 0].plot(t_simulation, transmission_inpulse, label='input wave')
    ax[2, 0].plot(t_simulation, make_reflected_wave(1.5, 'inpulse')[0], label='reflected wave')
    ax[2, 0].set_title('inpulse input & narrow interval', size=18)
    ax[2, 0].legend(fontsize=12)


    ax[3, 0].plot(t_simulation, correlation_2, label='correlation')
    ax[3, 0].legend(fontsize=12)


    ax[0, 1].plot(t_simulation, make_reflected_wave(2, 'chirp')[1], label='reflected wave with noise')
    ax[0, 1].plot(t_simulation, transmission_chirp, label='input wave')
    ax[0, 1].plot(t_simulation, make_reflected_wave(2,  'chirp')[0], label='reflected wave')
    ax[0, 1].set_title('chirp input & wide interval', size=18)
    ax[0, 1].legend(fontsize=12)


    ax[1, 1].plot(t_simulation, correlation_3, label='correlation')
    ax[1, 1].legend(fontsize=12)

    ax[2, 1].plot(t_simulation, make_reflected_wave(1.5,  'chirp')[1], label='reflected wave with noise')
    ax[2, 1].plot(t_simulation, transmission_chirp, label='input wave')
    ax[2, 1].plot(t_simulation, make_reflected_wave(1.5,  'chirp')[0], label='reflected wave')
    ax[2, 1].set_title('chirp input & narrow interval', size=18)
    ax[2, 1].legend(fontsize=12)

    ax[3, 1].plot(t_simulation, correlation_4, label='correlation')
    ax[3, 1].legend(fontsize=12)


    fig.supxlabel('Time [s]', size=20)
    fig.supylabel('Amplitude', size=20)
    plt.show()

    return plt
#plot()



# ===FFTの計算===
def calc_fft(data_array):
    # フーリエ変換
    fft_data = fft.fft(data_array)
    fft_data = fft.fftshift(fft_data)
    # パワースペクトル密度の計算
    #power_spectrum_density = np.abs(fft_data)**2
    #power = 10 * np.log10(power_spectrum_density)

    # 周波数軸の作成
    freq = fft.fftfreq(len(data_array), d=1/sample_rate)
    freq = fft.fftshift(freq)

    return fft_data, freq
fft_data, freq = calc_fft(transmission_chirp)
#plt.plot(freq, np.abs(fft_data))
#plt.xlim(1, 50)
#plt.show()

# ===整合フィルタ処理===
def calc_matched_filter(input_array, reflect_array):

    # パディング
    input_array = np.hstack((np.flipud(input_array), np.zeros(len(reflect_array) - len(input_array))))
    input_fft = fft.fft(input_array)
    reflect_fft = fft.fft(reflect_array)

    # 整合フィルタ処理
    fft_conv = input_fft * reflect_fft
    # フーリエ逆変換
    Ifft = fft.ifft(fft_conv)
    Ifft = np.abs(Ifft)/np.amax(np.abs(Ifft)) # 正規化
    power = 10 * np.log10(np.abs(Ifft))
    power -= np.amax(power) # 最大値を0dBにする

    return power

plt.plot(t_simulation, calc_matched_filter(chirp, make_reflected_wave(1.5, 'chirp')[1]))
plt.show()