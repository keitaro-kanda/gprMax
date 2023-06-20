import h5py

from tools.outputfiles_merge import get_output_data
from tools.plot_Bscan import mpl_plot

# 読み込むファイルの指定
filename_original = 'kanda/inner_tube/ver8/B-scan/v8_A_35_x4_02_original/inner_v8_merged.out'
filename_onlysurface = 'kanda/inner_tube/ver8/B-scan/v8_A_35_x4_02_singlelayer/inner_v8_merged.out'
filename_array = [filename_original, filename_onlysurface]


# .outファイルの読み込み
for filename in filename_array:
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    f.close()

    for rx in range(1, nrx + 1):
        if filename == filename_array[0]:
            outputdata_1, dt = get_output_data(filename, rx, 'Ez')
        elif filename == filename_array[1]:
            outputdata_2, dt = get_output_data(filename, rx, 'Ez')


# 地下エコー要素の抽出
outputdata_extract_subsurface = outputdata_1 - outputdata_2 
#f = h5py.File('outputdata_extract_subsurface.out', 'w')

def PrintOnlyDataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(name)
        # print('\t',obj)
outputdata_extract_subsurface.visititems(PrintOnlyDataset) 


# 地下エコー要素のプロット
#for rx in range(1, nrx + 1):
#    plthandle = mpl_plot(filename, outputdata_extract_subsurface, dt, rx, 'Ez')

#plthandle.show()

