"""
1本のA-scan (.out) をN本複製して、均一なB-scanのmergedファイルを作成する。

B-scan計算の結果ではなく、A-scanを複製して作ったファイルであることが分かるよう、
出力ファイル名は <base>_Ascan_xN_merged.out とする。
出力先は入力ファイルがあるディレクトリの親（= resultディレクトリと同階層）で、
outputfiles_merge.py が作る merged.out と同じ階層に揃えている。

Usage:
    cd gprMax; python -m tools.core.k_duplicate_Ascan
"""

import os

import h5py
import numpy as np
from tqdm import tqdm

from gprMax._version import __version__
from gprMax.exceptions import CmdInputError


def duplicate_Ascan(ascanfile, numtraces):
    """Duplicates a single A-scan into N identical traces and writes a uniform B-scan file.

    Args:
        ascanfile (string): Filename (including path) of the A-scan output file.
        numtraces (int): Number of identical traces in the output B-scan.

    Returns:
        outputfile (string): Filename (including path) of the written B-scan file.
    """

    ascanfile = os.path.abspath(ascanfile)

    if not os.path.isfile(ascanfile):
        raise CmdInputError('File {} does not exist'.format(ascanfile))
    if not ascanfile.endswith('.out'):
        raise CmdInputError('{} is not a .out file'.format(ascanfile))
    if numtraces < 1:
        raise CmdInputError('Number of traces must be 1 or more, but {} was given'.format(numtraces))

    # 出力先は入力ファイルのあるディレクトリの親（= resultディレクトリと同階層）
    parent_dir = os.path.dirname(os.path.dirname(ascanfile))
    base = os.path.basename(ascanfile)[:-len('.out')]
    outputfile = os.path.join(parent_dir, '{}_Ascan_x{}_merged.out'.format(base, numtraces))

    print('Input A-scan file: {}'.format(ascanfile))
    print('Number of traces:  {}'.format(numtraces))
    print('Output file:       {}'.format(outputfile))

    fin = h5py.File(ascanfile, 'r')
    nrx = fin.attrs['nrx']

    # 受信機が無いファイルは複製しても中身が空になるので弾く
    if nrx == 0:
        fin.close()
        raise CmdInputError('No receivers found in {}'.format(ascanfile))

    fout = h5py.File(outputfile, 'w')

    fout.attrs['Title'] = fin.attrs['Title']
    fout.attrs['gprMax'] = __version__
    fout.attrs['Iterations'] = fin.attrs['Iterations']
    fout.attrs['dt'] = fin.attrs['dt']
    fout.attrs['nrx'] = nrx

    # 全受信機・全出力成分について、同一トレースを numtraces 列ぶん並べる
    for rx in tqdm(range(1, nrx + 1)):
        path = '/rxs/rx' + str(rx) + '/'
        grp = fout.create_group('/rxs/rx' + str(rx))
        availableoutputs = list(fin[path].keys())
        # For all receiver outputs
        for output in availableoutputs:
            trace = np.array(fin[path + output])

            # A-scan単体の出力は (Iterations,) だが、1トレースだけのmerged形式 (Iterations, 1) も受け付ける
            if trace.ndim == 2:
                if trace.shape[1] != 1:
                    fin.close()
                    fout.close()
                    raise CmdInputError('{} in {} contains {} traces; this tool only accepts a single A-scan'.format(output, ascanfile, trace.shape[1]))
                trace = trace[:, 0]

            dset = grp.create_dataset(output, (fout.attrs['Iterations'], numtraces), dtype=fin[path + output].dtype)
            dset[:, :] = trace[:, np.newaxis]

    fin.close()
    fout.close()

    print('Duplicated 1 A-scan into {} identical traces.'.format(numtraces))

    return outputfile


if __name__ == "__main__":

    print("Duplicates a single A-scan into N identical traces, then writes a uniform B-scan merged file.")

    ascanfile = input("Enter the path to the A-scan .out file: ").strip()

    numtraces = input("Enter the number of traces (N): ").strip()
    try:
        numtraces = int(numtraces)
    except ValueError:
        raise CmdInputError('Number of traces must be an integer, but {} was given'.format(numtraces))

    duplicate_Ascan(ascanfile, numtraces)
