import numpy as np
import matplotlib.pyplot as plt

def gaussian_Ez_input(dt, t_max, f, amplitude):
    time = np.arange(0, t_max, dt)
    # a = 2 * (np.pi * f)**2
    # Ez_array = amplitude * np.exp(-a * (time - 1/f)**2)
    mean = 1 / f
    sigma2 = (1/(2*np.pi*f))**2 / 2
    E_array = amplitude * 1/np.sqrt(2 * np.pi * sigma2) * np.exp(-(time - mean)**2 / (2 * sigma2))

    return time, E_array

def gaussian_sin(dt, t_max, f, amplitude, FWHM):
    time = np.arange(0, t_max, dt)
    sigma = FWHM / 2.35
    Ez_array = np.exp(-((time - 4e-9)**2 / 2 / sigma**2)) * np.sin(2 * np.pi * f * (time - 4e-9))
    return time, Ez_array

def gaussian_cos(dt, t_max, f, amplitude, FWHM):
    time = np.arange(0, t_max, dt)
    sigma = FWHM / 2.35
    Ez_array = np.exp(-((time - 5e-9)**2 / 2 / sigma**2)) * np.cos(2 * np.pi * f * (time - 5e-9))
    return time, Ez_array

def plot_source_file(time, Ez_array, output_dir):
    # Calculate FFT
    Ez_fft = np.fft.fft(Ez_array)
    freq = np.fft.fftfreq(len(Ez_array), d=time[1] - time[0])
    magnitude = np.log(np.abs(Ez_fft/np.max(np.abs(Ez_fft)) + 1e-12))

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(time * 1e9, Ez_array)
    ax[0].set_title('Time Domain Signal')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('E-fileld')
    ax[0].grid()

    ax[1].plot(freq / 1e6, magnitude)
    ax[1].set_title('Frequency Domain Signal')
    ax[1].set_xlabel('Frequency (MHz)')
    ax[1].set_ylabel('Magnitude')
    ax[1].set_xlim(0, 2000)  # Limit x-axis to 2000 MHz
    ax[1].grid()

    plt.savefig(f"{output_dir}/my_pulse.png", dpi=300)
    plt.show()



def main():
    output_dir = input("Enter output directory path: ").strip()

    dt = 4.71731e-12 # dx=dy=dz=0.002m
    t_max = 20e-9
    f = 500e6
    amplitude = 1.0
    FWHM = 1.450e-9

    time, Ez_array = gaussian_cos(dt, t_max, f, amplitude, FWHM)
    plot_source_file(time, Ez_array, output_dir)

    # Save to file
    np.savetxt(f"{output_dir}/my_pulse.txt", Ez_array, header='my_pulse', comments='')

if __name__ == "__main__":
    main()

