# ディレクトリ構成について
- 基本：'kanda'の中に作成
- 構成：kanda > domein_OOxOO > geometryの違い > レーダー設定の違い
- jsonファイルはそれぞれの'out_files', '_merged.out'と同階層に配置
    - 例外的に，geometryフォルダがある場合には'geometry.h5'等と同階層に'geometry.json'を置いてもよい
- jsonファイルの命名は，jsonファイルが置いてある直上ディレクトリと同名にする

# プロットのフォントサイズ
- fontsize_large: 20 -> プロットのタイトル
- fontsize_medium: 18 -> 軸ラベル，カラーバーのラベル
- fontsize_small: 16 -> メモリの数値等