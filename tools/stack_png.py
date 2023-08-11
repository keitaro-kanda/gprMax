import numpy as np

# pngファイルの読み込み
png = np.loadtxt('kanda/domain_10x10/test/B-scan/smooth_2_biarray/migration_results/migration_result1.png', delimiter=',')
print(png.shape)
print(png)