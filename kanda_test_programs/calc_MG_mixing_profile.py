"""レゴリス＋水氷の誘電プロファイル：Maxwell-Garnett 版

calc_LLL_mixing_profile.py（LLL 混合則版）の対照実験。
**混合則だけを差し替えて、それ以外は完全に同一**にすることで、
結果の混合則依存性を切り分ける。

--- なぜこれを作るのか -------------------------------------------------------
LLL 版で採用した「氷は空隙を埋める」という描像は、混合則の選択とは独立の
物理的判断である。しかし得られた結論（氷 1 vol% で eps' が +1.15%、界面反射
-51 dB）が **混合則の選び方に強く依存する**なら、その結論は信用できない。
そこで同じ体積分率の考え方を Maxwell-Garnett (MG) 則に適用し、傾向が一致
することを確認する。

--- MG が適用範囲外であることは承知のうえ ------------------------------------
MG は「連続な母材の中に孤立した介在物が薄く分散している」ことを前提とする。
レゴリスは空隙率 30-45% で、粒子も空隙も連続相と呼べる状態なので、本来は
適用範囲外である。また MG は母材と介在物を区別する非対称な式なので、
「どちらを母材と呼ぶか」という恣意性が入る（LLL や対数混合は対称なので
この問題がない）。

したがって本スクリプトの目的は「MG のほうが正しい」ことを示すことではなく、
**適用範囲外の規則を使っても傾向が変わらないことを確認する**ことにある。

--- 体積分率の考え方（LLL 版と共通）-----------------------------------------
    v_grain = rho(z) / rho_grain          Carrier の密度プロファイルから
    v_ice   = 指定値
    v_void  = 1 - v_grain - v_ice         残りを空隙が埋める
氷は空隙（真空）を置き換える。粒子の量は変わらない。

--- MG の適用形 -------------------------------------------------------------
多成分 MG は母材 eps_h に対して

    (eps_eff - eps_h)/(eps_eff + 2 eps_h) = sum_i v_i (eps_i - eps_h)/(eps_i + 2 eps_h)

と書ける。S = sum_i v_i beta_i と置けば eps_eff = eps_h (1 + 2S)/(1 - S)。

母材の選択（MG_HOST）:
  'void'  … 真空を母材、粒子と氷を介在物とする（既定）
  'grain' … 粒子を母材、空隙と氷を介在物とする
どちらも「適用範囲外だがやってみる」という位置づけ。両方試して差を見る。

--- 粒子誘電率の決め方（MG_GRAIN_MODE）--------------------------------------
MG では乾燥時の eps' が Carrier の 1.871^rho と一致しないので、粒子誘電率
eps_grain をどう決めるかで結果が変わる。2 通りを用意した。

  'per_depth'（既定）
      各深さで「乾燥時の MG が Carrier の eps' に一致する」ように eps_grain
      を逆算する。乾燥プロファイルが LLL 版と厳密に一致するので、
      **2 つの出力の差は氷の項だけになる**。混合則依存性を見るのが目的
      なので、これが本命。
  'fixed'
      基準密度（既定は eps'=3.0 に対応する rho）で一度だけ較正し、以後は
      固定する。素直な MG モデルだが、乾燥プロファイル自体が Carrier から
      ずれるので、そのずれも一緒に見えてしまう。

--- 虚部の扱い（LLL 版と共通）-----------------------------------------------
損失は鉱物粒子で発生し、空隙に氷が入っても粒子の量は変わらないので eps'' は
保存する（氷自身の微小な損失だけを加える）。これは混合則とは独立の物理的
判断であり、MG 版でも同じ扱いにする。虚部に混合則を適用すると「損失源が
増えていないのに減衰が増える」という非物理的な結果になる。
比較のため MG_EPS_IMAG_MODE = 'complex_mg' で複素 MG も試せるようにしてある。

--- 分散の扱い（LLL 版と共通）-----------------------------------------------
eps'' は帯域内で一定。Level_3.in と同じ最大平坦 2 極 Debye の解析解で実現する。
LLL 版の dry_eps_complex() をそのまま使うので自動的に揃う。

--- 実装方針 ---------------------------------------------------------------
物理定数・作図・検算はすべて calc_LLL_mixing_profile.py から import して使い、
**混合則の部分だけを差し替える**。定数を 2 か所に書かない（修正項目 A-1 と
同じ事故を防ぐ）。差し替えるのは medium_eps() ただ 1 つ。
"""

import os
import numpy as np
from scipy.optimize import brentq

import calc_LLL_mixing_profile as base
from calc_LLL_mixing_profile import (
    z, ice_contents, EPS_ICE, TAND_ICE, RHO_GRAIN,
    density_profile, carrier_eps_real, porosity, dry_eps_complex,
)

# 氷の描像（'pore' / 'excess'）は LLL 版の ICE_MODEL をそのまま参照する。
# main() が切り替えるので、ここでは import せずに base.ICE_MODEL を都度読む。

# =============================================================================
# 0. 設定  [EDIT HERE]
# =============================================================================
# 出力先。LLL 版と並べて比較できるよう、同じ親の下に別ディレクトリを作る。
OUTPUT_BASE = ('/Volumes/SSD_Kanda_BUFFALO/test_programs_output/'
               'MG_mixing_profile')

MG_HOST = 'grain'              # 'void'（真空が母材、既定）/ 'grain'（粒子が母材）

MG_GRAIN_MODE = 'per_depth'   # 'per_depth'（既定。各深さで Carrier に一致させる）
                              # 'fixed'（基準密度で一度だけ較正）
MG_GRAIN_REF_DEPTH_M = 0.5    # 'fixed' のときの較正深さ [m]。
                              # 0.5 m は eps' がほぼ 3.0 になる深さ
                              # （Level 3 の均質モデルが対応する深さ）

MG_EPS_IMAG_MODE = 'conserved'  # 'conserved'（既定。LLL 版と同じ）
                                # 'complex_mg'（複素 MG。比較用）

# 粒子誘電率の探索範囲（'per_depth' / 'fixed' の逆算に使う）
GRAIN_EPS_BOUNDS = (1.001, 200.0)


# =============================================================================
# 1. Maxwell-Garnett
# =============================================================================
def _beta(eps_i, eps_h):
    """MG の分極率因子 beta = (eps_i - eps_h)/(eps_i + 2 eps_h)。"""
    return (eps_i - eps_h) / (eps_i + 2.0 * eps_h)


def mg_effective(eps_host, inclusions):
    """多成分 Maxwell-Garnett。

    inclusions は [(体積分率, 誘電率), ...]。母材の体積分率は
    1 - sum(v_i) だが、式には現れない（母材は残り全部という扱い）。

        (eps_eff - eps_h)/(eps_eff + 2 eps_h) = sum_i v_i beta_i
        -> eps_eff = eps_h (1 + 2S)/(1 - S)
    """
    s = sum(v * _beta(e, eps_host) for v, e in inclusions)
    return eps_host * (1.0 + 2.0 * s) / (1.0 - s)


def mg_dry(eps_grain, v_grain):
    """乾燥レゴリス（粒子＋真空）の MG。氷なし。"""
    if MG_HOST == 'void':
        return mg_effective(1.0, [(v_grain, eps_grain)])
    if MG_HOST == 'grain':
        return mg_effective(eps_grain, [(1.0 - v_grain, 1.0)])
    raise ValueError("MG_HOST は 'void' か 'grain'")


def mg_wet(eps_grain, v_grain_dry, v_ice, eps_ice=None):
    """粒子・真空・氷の 3 相 MG。氷の描像で体積分率の作り方が変わる。

    v_grain_dry は乾燥時の粒子体積分率 rho/rho_grain。

    'pore'（吸着水描像）
        氷は空隙だけを埋める。粒子は減らない。
            v_grain = v_grain_dry
            v_void  = 1 - v_grain_dry - v_ice
    'excess'（過剰氷描像）
        氷がレゴリスごと押しのける。粒子も空隙も (1-v_ice) 倍になる。
            v_grain = (1 - v_ice) * v_grain_dry
            v_void  = (1 - v_ice) * (1 - v_grain_dry)
    """
    ei = EPS_ICE if eps_ice is None else eps_ice
    if base.ICE_MODEL == 'excess':
        v_g = (1.0 - v_ice) * v_grain_dry
        v_v = (1.0 - v_ice) * (1.0 - v_grain_dry)
    else:
        v_g = v_grain_dry
        v_v = 1.0 - v_grain_dry - v_ice
    if MG_HOST == 'void':
        return mg_effective(1.0, [(v_g, eps_grain), (v_ice, ei)])
    if MG_HOST == 'grain':
        return mg_effective(eps_grain, [(v_v, 1.0), (v_ice, ei)])
    raise ValueError("MG_HOST は 'void' か 'grain'")


def solve_eps_grain(eps_dry_target, v_grain):
    """乾燥時の MG が eps_dry_target に一致する eps_grain を求める。

    MG_HOST='void' なら閉形式で解ける:
        S = (eps_eff - 1)/(eps_eff + 2),  beta_g = S / v_grain
        eps_grain = (1 + 2 beta_g)/(1 - beta_g)
    MG_HOST='grain' は eps_grain が式のあちこちに現れるので数値的に解く。
    """
    if MG_HOST == 'void':
        s = (eps_dry_target - 1.0) / (eps_dry_target + 2.0)
        b = s / v_grain
        if b >= 1.0:
            raise ValueError(
                'MG（真空が母材）では eps_dry = {:.4f} を v_grain = {:.4f} で'
                '再現できません（beta >= 1）。'
                'MG の適用範囲外です。'.format(eps_dry_target, v_grain))
        return (1.0 + 2.0 * b) / (1.0 - b)

    def resid(eg):
        return mg_dry(eg, v_grain) - eps_dry_target
    lo, hi = GRAIN_EPS_BOUNDS
    if resid(lo) * resid(hi) > 0:
        raise ValueError(
            'MG（粒子が母材）で eps_dry = {:.4f} を再現する eps_grain が '
            '{} の範囲に見つかりません。'.format(eps_dry_target, GRAIN_EPS_BOUNDS))
    return brentq(resid, lo, hi, xtol=1e-12, rtol=1e-14)


# 深さごとの粒子誘電率（'per_depth'）または固定値（'fixed'）を先に作る。
_RHO_Z = density_profile(z)
_V_GRAIN_Z = _RHO_Z / RHO_GRAIN


def _build_grain_table():
    """深さ配列 z に対応する eps_grain を返す（形は (Nz,)）。"""
    eps_dry = carrier_eps_real(_RHO_Z)
    if MG_GRAIN_MODE == 'per_depth':
        return np.array([solve_eps_grain(ed, vg)
                         for ed, vg in zip(eps_dry, _V_GRAIN_Z)])
    if MG_GRAIN_MODE == 'fixed':
        rho_ref = float(np.atleast_1d(density_profile(MG_GRAIN_REF_DEPTH_M))[0])
        eg = solve_eps_grain(carrier_eps_real(rho_ref), rho_ref / RHO_GRAIN)
        return np.full_like(_RHO_Z, eg)
    raise ValueError("MG_GRAIN_MODE は 'per_depth' か 'fixed'")


EPS_GRAIN_Z = _build_grain_table()


def eps_grain_at(depth_m):
    """任意の深さの eps_grain（z 格子から線形内挿）。"""
    d = np.atleast_1d(np.asarray(depth_m, dtype=float))
    return np.interp(d, z, EPS_GRAIN_Z)


# =============================================================================
# 2. LLL 版の medium_eps を MG 版に差し替える
# =============================================================================
def medium_eps_mg(depth_m, freq_hz, ice_volpct, feotio2_wt=None):
    """深さ・周波数・氷量に対する (eps', eps'')。形は (Nz, Nf)。

    LLL 版 medium_eps() と同じ signature を持ち、混合則だけが違う。

    実部:
        乾燥時の eps' は Debye 実現込みで dry_eps_complex() から得る
        （周波数依存を LLL 版と厳密に揃えるため）。氷を入れたときの変化は
        MG の乾燥／湿潤の比として掛ける。
            eps'_wet(f) = eps'_dry(f) * [ mg_wet / mg_dry ]
        比で入れるのは、Debye 実現による帯域内のわずかな周波数依存を
        壊さないようにするため。MG_GRAIN_MODE='per_depth' なら
        mg_dry は Carrier の eps' に一致するので、この比は素直に
        「氷による増分」だけを表す。

    虚部:
        既定は LLL 版と同じ扱い。
          'pore'   … 粒子が減らないので eps'' は保存（氷の微小な損失だけ加算）
          'excess' … 粒子が (1-v_ice) 倍に減るので eps'' も同じ割合で希釈
        'complex_mg' なら複素 MG で計算する（比較用）。
    """
    er_dry, ei_dry = dry_eps_complex(depth_m, freq_hz, feotio2_wt)
    v = float(ice_volpct) / 100.0
    if v == 0.0:
        return er_dry, ei_dry

    d = np.atleast_1d(np.asarray(depth_m, dtype=float))
    eg = eps_grain_at(d)[:, None]                 # (Nz,1)
    vg = (density_profile(d) / RHO_GRAIN)[:, None]

    if MG_EPS_IMAG_MODE == 'complex_mg':
        # 複素誘電率のまま MG を適用する（比較用）。
        # 損失源が増えていないのに減衰が増えることがあるので既定にはしない。
        eps_c_dry = er_dry - 1j * ei_dry
        # 粒子の複素誘電率は「乾燥時の複素 MG が eps_c_dry に一致する」よう
        # 実部と同じ比で虚部を割り当てる（近似）。
        eg_c = eg * (1.0 + 1j * 0.0)
        eg_c = eg_c - 1j * (eg * (ei_dry / er_dry))
        wet = mg_wet(eg_c, vg, v, eps_ice=EPS_ICE - 1j * EPS_ICE * TAND_ICE)
        dry = mg_dry(eg_c, vg)
        ratio = wet / dry
        eps_c = eps_c_dry * ratio
        return np.real(eps_c), -np.imag(eps_c)

    ratio = mg_wet(eg, vg, v) / mg_dry(eg, vg)
    eps_re = er_dry * ratio
    if base.ICE_MODEL == 'excess':
        eps_im = (1.0 - v) * ei_dry + v * EPS_ICE * TAND_ICE
    else:
        eps_im = ei_dry + v * EPS_ICE * TAND_ICE
    return eps_re, eps_im


# --- 差し替えとキャッシュのクリア -------------------------------------------
# build_profile_set / alpha_velocity / propagation_table はすべて
# モジュール globals 経由で medium_eps を引くので、ここで置き換えれば
# 以降の計算はすべて MG 版になる。
base.medium_eps = medium_eps_mg
base.OUTPUT_BASE = OUTPUT_BASE
base.MIXING_LABEL = 'MG'
base.MIXING_DESC = ('Maxwell-Garnett  host={}, eps_grain={}, eps_imag={}'
                    .format(MG_HOST, MG_GRAIN_MODE, MG_EPS_IMAG_MODE))
base._prop_cache.clear()
base._moments_cache.clear()

# 差し替え後にプロファイル配列を作り直す（import 時に LLL で作られているため）
base.SET_FREQ = base.build_profile_set(base.PAIRS_FREQ)
base.SET_COMP = base.build_profile_set(base.PAIRS_COMP)


# =============================================================================
# 3. MG 固有の検算
# =============================================================================
def run_mg_checks():
    """混合則の妥当性と、LLL 版との差を数値で出す。"""
    lines = []
    add = lines.append
    i15 = int(1.5 / base.DZ)

    add('=' * 74)
    add('Maxwell-Garnett 版の設定')
    add('=' * 74)
    add(f'  母材            : {MG_HOST}')
    add(f'  eps_grain の決め方: {MG_GRAIN_MODE}'
        + (f'（較正深さ {MG_GRAIN_REF_DEPTH_M} m）'
           if MG_GRAIN_MODE == 'fixed' else ''))
    add(f'  虚部の扱い      : {MG_EPS_IMAG_MODE}')
    add(f'  氷の描像        : {base.ICE_MODEL} '
        f'（{base.ICE_MODEL_LABELS[base.ICE_MODEL]}）')
    add(f'  粒子密度        : {RHO_GRAIN} g/cm^3（空隙率の計算にのみ使う '
        'LLL 版と共通）')
    add('')
    add('  深さ [m]   rho     v_grain   v_void(氷なし)   eps_grain   '
        'eps_dry(MG)  eps_dry(Carrier)')
    for d in (0.0, 0.5, 1.5, 3.0):
        i = int(d / base.DZ) if d < z[-1] else len(z) - 1
        eg = EPS_GRAIN_Z[i]
        add('  {:7.2f}  {:.4f}  {:.4f}   {:.4f}          {:8.4f}   '
            '{:10.6f}  {:10.6f}'.format(
                z[i], _RHO_Z[i], _V_GRAIN_Z[i], 1 - _V_GRAIN_Z[i], eg,
                mg_dry(eg, _V_GRAIN_Z[i]), carrier_eps_real(_RHO_Z[i])))
    add('')
    if MG_GRAIN_MODE == 'per_depth':
        err = np.max(np.abs(
            np.array([mg_dry(eg, vg) for eg, vg in zip(EPS_GRAIN_Z, _V_GRAIN_Z)])
            / carrier_eps_real(_RHO_Z) - 1.0))
        add(f'  乾燥時の MG と Carrier の最大相対差: {100 * err:.3e} %')
        add('    -> per_depth なので厳密に一致する。LLL 版と乾燥プロファイルが')
        add('       同一になるため、2 つの出力の差は氷の項だけになる。')
    else:
        err = np.max(np.abs(
            np.array([mg_dry(eg, vg) for eg, vg in zip(EPS_GRAIN_Z, _V_GRAIN_Z)])
            / carrier_eps_real(_RHO_Z) - 1.0))
        add(f'  乾燥時の MG と Carrier の最大相対差: {100 * err:.3f} %')
        add('    -> fixed では MG の関数形が Carrier の指数形と違うため、')
        add('       基準深さから離れるほどずれる。')
    add('')

    add('=' * 74)
    add('LLL 版との比較（深さ 1.5 m, {:.2f} GHz, {} wt%, 描像 {}）'
        .format(base.PROFILE_FIXED_FREQ / 1e9, base.FEOTIO2_WT,
                base.ICE_MODEL))
    add('=' * 74)
    fa = np.array([base.PROFILE_FIXED_FREQ])
    add('  氷[vol%]   eps_prime            d(eps_prime)/vol%      界面R [dB]')
    add('              MG        LLL        MG      LLL        MG      LLL')
    er0_mg = medium_eps_mg(z, fa, 0)[0][i15, 0]
    er0_lll = _lll_medium_eps(z, fa, 0)[0][i15, 0]
    for c in ice_contents:
        if c == 0:
            continue
        er_mg = medium_eps_mg(z, fa, c)[0][i15, 0]
        er_lll = _lll_medium_eps(z, fa, c)[0][i15, 0]
        d_mg = 100 * (er_mg / er0_mg - 1) / c
        d_lll = 100 * (er_lll / er0_lll - 1) / c
        r_mg = (np.sqrt(er0_mg) - np.sqrt(er_mg)) / (np.sqrt(er0_mg) + np.sqrt(er_mg))
        r_lll = (np.sqrt(er0_lll) - np.sqrt(er_lll)) / (np.sqrt(er0_lll) + np.sqrt(er_lll))
        add('  {:6}   {:.5f}  {:.5f}   {:+6.3f}  {:+6.3f}   {:6.1f}  {:6.1f}'
            .format(c, er_mg, er_lll, d_mg, d_lll,
                    20 * np.log10(abs(r_mg)), 20 * np.log10(abs(r_lll))))
    add('')
    add('  -> 2 つの規則で d(eps\')/vol% と界面反射がどれだけ違うかが')
    add('     「結果の混合則依存性」そのもの。')
    add('')

    add('=' * 74)
    add('空隙率の確認')
    add('=' * 74)
    por = porosity(_RHO_Z)
    add(f'  空隙率: {100 * por.min():.1f} - {100 * por.max():.1f} %')
    bad = [c for c in ice_contents if c / 100.0 > por.min()]
    if bad:
        add(f'  ** 空隙率を超える氷量: {bad} vol% **')
    else:
        add('  すべての氷量が全深さで空隙率以内 -> OK')

    text = '\n'.join(lines)
    print(text)
    os.makedirs(base.out_dir(), exist_ok=True)
    with open(base.out_dir('mg_summary.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write(text + '\n')


def _lll_medium_eps(depth_m, freq_hz, ice_volpct, feotio2_wt=None):
    """比較用に LLL 版の混合をその場で計算する。

    base.mix_ice() をそのまま呼ぶので、描像（pore / excess）の分岐を
    2 か所に書かずに済む。
    """
    er, ei = dry_eps_complex(depth_m, freq_hz, feotio2_wt)
    return base.mix_ice(er, ei, ice_volpct)


# =============================================================================
# 4. 実行
# =============================================================================
def main():
    print('=' * 74)
    print('Maxwell-Garnett 版プロファイル計算')
    print('  （LLL 版 calc_LLL_mixing_profile.py の対照実験。'
          '混合則だけが違う）')
    print('=' * 74)

    # 描像ごとに「MG 固有の検算 -> 図と CSV」を回す。
    # 作図・CSV・共通の検算は LLL 版のものをそのまま使う。
    for model in base.ICE_MODELS:
        base.ICE_MODEL = model
        base.rebuild_profiles()
        print()
        run_mg_checks()
        print()
        base.run_for_model(model)
        print()
    base.compare_models(base.ICE_MODELS)


if __name__ == '__main__':
    main()