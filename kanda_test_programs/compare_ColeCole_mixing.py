"""
Mixing-rule validation: MG(bytownite + pure ilmenite) vs Boivin+2022 measurements
=================================================================================
Boivin et al. (2022) JGR Planets は、バイトウナイトに純イルメナイトを
10/15/20 wt% 混合した試料と、純イルメナイト単体の誘電特性を報告している。

  Table 4 : 各試料の 1極 Cole-Cole フィットパラメータ
  Table 3 : 各試料のバルク密度（体積分率の決定に必須）
  Table 7 : 各帯域での eps', eps'', tan_delta の実測値（純バイトウナイト含む）

このスクリプトは次を検証する:

  「純バイトウナイト + 純イルメナイト」を混合則で混ぜて、
   Boivin が実測した 10/15/20 wt% 混合物を再現できるか？

再現できれば、Boivin が測っていない低イルメナイト域（月南極の <1 wt%）へ
構成的に外挿できる。再現できなければ、混合則による構成は使えず、
実測値の内挿・外挿に頼るしかない。

比較は 2 段階で行う:
  (1) 複素誘電率スペクトルそのものの比較（混合則ごと、全帯域）
  (2) 双方を多極 Debye でフィットし、gprMax に渡すパラメータの比較


プロットの各線の意味
--------------------

  黒の実線 "Boivin measured"
      Boivin の生データそのものではなく、**Table 4 の 1極 Cole-Cole
      フィットパラメータを連続曲線として評価したもの**。すなわち
      Boivin が自分のデータに当てはめたモデル曲線である。

  黒の四角マーカー
      Table 7 に載っている**帯域ごとの実測値**（P/L/S/X の 4 点）。
      黒実線がこの 4 点をよく通っていれば、Cole-Cole フィットが
      実測を正しく代表していることの確認になる。

  破線 4 本 = 混合則で構成したスペクトル
      いずれも「純バイトウナイト（Table 7、非分散）に純イルメナイト
      （Table 4 の Cole-Cole）を体積分率 fv で混ぜた」結果。
      違いは混合則だけで、入力する 2 成分は共通。

      MG            : Maxwell-Garnett（球形介在物、脱分極係数 N = 1/3）。
                      希薄な介在物が連続的な母材に埋まっている状況を仮定する。
                      母材と介在物を対等に扱わない「非対称」な混合則。

      MG (N=0.462)  : 同じ Maxwell-Garnett だが、脱分極係数 N を
                      実測 eps'' に最も合うよう最小二乗で決めた版。
                      N は介在物の形状を表し、N=1/3 が球、N>1/3 が扁平（板状）、
                      N<1/3 が伸長（針状）に対応する。N=0.462 は軸比
                      c/a = 0.63（厚み:直径 = 1:1.6）の扁平粒子に相当する。
                      ただしランダム配向では (N_z + 2N_x)/3 = 1/3 が常に成立する
                      ため、スカラー N != 1/3 は本来「整列した粒子」を意味する。
                      粉体には物理的に不自然なので、これは形状の推定ではなく
                      **経験的な調整パラメータ**（球形 MG からどれだけ外れているか
                      の指標）として読むべきである。

      Bruggeman     : Bruggeman 対称有効媒質近似。MG と違い母材と介在物を
                      **対等**に扱う。各粒子が「有効媒質そのもの」に埋まって
                      いるとして自己無撞着に解くため、どちらが母材か決めにくい
                      高濃度混合や、両相が連結しうる系に適する。
                      浸透（パーコレーション）しきい値を持つのが特徴。

      CRIM          : Complex Refractive Index Method。
                      sqrt(eps_eff) = (1-fv) sqrt(eps_host) + fv sqrt(eps_incl)
                      という「屈折率の体積平均」。べき乗則 k=1/2 の場合に相当する。
                      GPR や土壌水分の分野で広く使われる経験的混合則で、
                      物理的導出よりも実用上の当てはまりの良さで選ばれる。

      （Lichtenecker : 上のべき乗則の k -> 0 極限。誘電率の対数を体積平均する。
                      表出力には含まれるがプロットには描いていない。）


結果の要点（実行すると確認できる）
----------------------------------

  * eps' はどの混合則でも 2-4% で一致する
  * eps'' はどの混合則でも一致しない（最良の MG(N 最適化) でも RMS 20%）
  * 原因は緩和時間 tau の希釈依存性:
        純イルメナイト tau = 374 ps  -> 緩和ピーク 0.43 GHz
        混合物        tau =  44-52 ps -> 緩和ピーク 3.1-3.6 GHz
    希釈すると緩和ピークが約 8 倍高周波側へ動くが、混合則は介在物の tau を
    保存するためこれを再現できない。脱分極係数や体積分率の調整では解消しない
    （形状ではなく時定数の問題であるため）。

  -> 「純イルメナイトを混合則で薄める」方法で低イルメナイト域を構成することは
     できない。低イルメナイト域は実測（Boivin が 1, 5 wt% で有意な分散を
     検出できなかったという報告）に従い、非分散として扱うのが妥当。

Requirements: numpy, scipy, matplotlib
"""

import os
import io
import datetime

import numpy as np
from scipy.optimize import least_squares, brentq, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 0. Boivin+2022 のデータ
# ---------------------------------------------------------------------------

# --- Table 4: 1極 Cole-Cole フィットパラメータ ---
#   eps*(w) = eps_inf + delta_eps / (1 + (i w tau)^alpha) - i sigma_dc/(w eps0)
COLE_COLE = {
    10:     dict(eps_inf=3.554, delta_eps=0.194, tau=4.370e-11, alpha=0.717, sigma_dc=2e-5),
    15:     dict(eps_inf=3.659, delta_eps=0.291, tau=5.201e-11, alpha=0.746, sigma_dc=1e-5),
    20:     dict(eps_inf=3.792, delta_eps=0.420, tau=5.036e-11, alpha=0.756, sigma_dc=1e-5),
    'pure': dict(eps_inf=7.048, delta_eps=17.437, tau=3.742e-10, alpha=0.553, sigma_dc=2e-5),
}

# --- Table 3: バルク密度 [g/cm^3] ---
#   体積分率の決定に使う。粉体試料なので固体密度ではなくバルク密度が必要。
BULK_DENSITY = {'byt': 1.61, 1: 1.63, 5: 1.65, 10: 1.69, 15: 1.73, 20: 1.76, 'pure': 2.70}

# --- Table 7: 各帯域の実測 (eps', eps'') ---
#   純バイトウナイトは全帯域で同一 = 非分散であることの直接的証拠。
MEASURED = {
    0.430e9: {'byt': (3.29, 0.006), 10: (3.73, 0.031), 15: (3.92, 0.050), 20: (4.17, 0.071)},
    1.25e9:  {'byt': (3.29, 0.006), 10: (3.70, 0.051), 15: (3.86, 0.083), 20: (4.10, 0.119)},
    2.38e9:  {'byt': (3.29, 0.006), 10: (3.67, 0.060), 15: (3.82, 0.096), 20: (4.03, 0.139)},
    7.14e9:  {'byt': (3.29, 0.006), 10: (3.62, 0.057), 15: (3.74, 0.084), 20: (3.91, 0.124)},
}
BAND_NAME = {0.430e9: 'P', 1.25e9: 'L', 2.38e9: 'S', 7.14e9: 'X'}

# --- 母材（ホスト）の複素誘電率 ---
#   Boivin の純バイトウナイトは全帯域で eps'=3.29, eps''=0.006 の非分散。
EPS_BYTOWNITE = 3.29 - 1j * 0.006

# ---------------------------------------------------------------------------
# [EDIT HERE] 計算設定
# ---------------------------------------------------------------------------
VOLUME_FRACTION_MODE = 'bulk_density'   # 'bulk_density' : Table 3 のバルク密度から
                                        # 'solid_density': 固体密度から（空隙を無視）
SOLID_RHO_ILM, SOLID_RHO_BYT = 4.72, 2.72   # [g/cm^3] solid_density モード用

MIXING_RULES = ['MG', 'MG_fitN', 'Bruggeman', 'Lichtenecker', 'CRIM']
VALIDATION_WT = [10, 15, 20]            # 検証に使う実測試料
EXTRAPOLATE_WT = [0.5, 1.0, 2.0, 5.0]   # 混合則で外挿してみる低イルメナイト域

F_MIN, F_MAX, N_F = 1e8, 1e10, 500
GPR_LO, GPR_HI, FC_GPR = 0.5e9, 2.0e9, 1.25e9
EPS0 = 8.854187817e-12

# gprMax の時間刻み制約（dx=0.0025 m, 2D クーラン条件）
DX, C0 = 0.0025, 2.99792458e8
DT_GPRMAX = DX / (np.sqrt(2) * C0)
TAU_MIN = 2 * DT_GPRMAX

OUTPUT_DIR = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/mixing_rule_validation'
RESULTS_FILENAME = 'mixing_rule_validation.txt'

freqs = np.geomspace(F_MIN, F_MAX, N_F)


# ---------------------------------------------------------------------------
# 1. Cole-Cole モデルと体積分率
# ---------------------------------------------------------------------------

def cole_cole(f, eps_inf, delta_eps, tau, alpha, sigma_dc=0.0):
    """1極 Cole-Cole の複素比誘電率。eps'' > 0 の符号で返す。

    Boivin Eq.1 の規約 (i w tau)^alpha を使う。alpha=1 が Debye。
    """
    w = 2 * np.pi * np.asarray(f, dtype=float)
    eps_star = eps_inf + delta_eps / (1.0 + (1j * w * tau) ** alpha) \
        - 1j * sigma_dc / (w * EPS0)
    return eps_star.real - 1j * (-eps_star.imag)   # eps' - i eps''


def volume_fraction(wt_pct, mode=VOLUME_FRACTION_MODE):
    """イルメナイト wt% -> 体積分率。

    'bulk_density' : Table 3 のバルク密度を使い、混合物の体積を
                     「バイトウナイト粉体の体積 + イルメナイト粉体の体積」
                     とみなす。粉体混合の実態に即しており、Table 3 の実測
                     バルク密度を 1% 以内で再現する（下の check_volume_model 参照）。
    'solid_density': 固体密度から求める。空隙を無視するので粉体には不適だが
                     比較のため用意。
    """
    if mode == 'bulk_density':
        v_ilm = wt_pct / BULK_DENSITY['pure']
        v_byt = (100.0 - wt_pct) / BULK_DENSITY['byt']
    else:
        v_ilm = wt_pct / SOLID_RHO_ILM
        v_byt = (100.0 - wt_pct) / SOLID_RHO_BYT
    return v_ilm / (v_ilm + v_byt)


def check_volume_model(_p):
    """加算体積モデルがバルク密度の実測を再現するか検証する。"""
    _p("\n--- 体積分率モデルの検証（Table 3 のバルク密度） ---")
    _p(f"{'wt%':>6}{'計算 rho_b':>12}{'実測 rho_b':>12}{'誤差':>9}{'ilm vol%':>11}")
    for w in [1, 5, 10, 15, 20]:
        v = w / BULK_DENSITY['pure'] + (100.0 - w) / BULK_DENSITY['byt']
        rho_calc = 100.0 / v
        _p(f"{w:>6}{rho_calc:>12.3f}{BULK_DENSITY[w]:>12.2f}"
           f"{100 * (rho_calc / BULK_DENSITY[w] - 1):>8.2f}%{volume_fraction(w) * 100:>11.2f}")


# ---------------------------------------------------------------------------
# 2. 混合則
# ---------------------------------------------------------------------------

def mix_maxwell_garnett(eps_host, eps_incl, fv, N=1.0 / 3.0):
    """Maxwell-Garnett。N は脱分極係数（球なら 1/3、板状なら大きくなる）。"""
    return eps_host * (1.0 + fv * (eps_incl - eps_host)
                       / (eps_host + (1.0 - fv) * N * (eps_incl - eps_host)))


def mix_bruggeman(eps_host, eps_incl, fv):
    """Bruggeman 対称有効媒質近似。母材と介在物を対等に扱う。"""
    from scipy.optimize import fsolve

    def residual(x):
        e = x[0] + 1j * x[1]
        r = (1 - fv) * (eps_host - e) / (eps_host + 2 * e) \
            + fv * (eps_incl - e) / (eps_incl + 2 * e)
        return [r.real, r.imag]

    sol = fsolve(residual, [eps_host.real, eps_host.imag], full_output=False)
    return sol[0] + 1j * sol[1]


def mix_power_law(eps_host, eps_incl, fv, k):
    """べき乗則。k=1/2 が CRIM、k->0 が Lichtenecker（対数）混合。"""
    if abs(k) < 1e-9:
        return np.exp((1 - fv) * np.log(eps_host) + fv * np.log(eps_incl))
    return ((1 - fv) * eps_host ** k + fv * eps_incl ** k) ** (1.0 / k)


def apply_rule(rule, eps_host, eps_incl, fv, N_opt=None):
    """混合則名を受け取って適用する。"""
    if rule == 'MG':
        return mix_maxwell_garnett(eps_host, eps_incl, fv)
    if rule == 'MG_fitN':
        return mix_maxwell_garnett(eps_host, eps_incl, fv, N=N_opt)
    if rule == 'Bruggeman':
        return mix_bruggeman(eps_host, eps_incl, fv)
    if rule == 'Lichtenecker':
        return mix_power_law(eps_host, eps_incl, fv, 0.0)
    if rule == 'CRIM':
        return mix_power_law(eps_host, eps_incl, fv, 0.5)
    raise ValueError('unknown rule: {}'.format(rule))


def fit_depolarisation_factor():
    """Table 7 の実測 eps'' に最も合う MG の脱分極係数 N を求める。

    球（N=1/3）で合わない場合、粒子形状で説明できるかを見るための診断。
    N > 1/3 は扁平（板状）、N < 1/3 は伸長（針状）に対応する。
    """
    def cost(N):
        res = []
        for f, d in MEASURED.items():
            e_i = cole_cole(f, **COLE_COLE['pure'])
            for w in VALIDATION_WT:
                m = mix_maxwell_garnett(EPS_BYTOWNITE, e_i, volume_fraction(w), N=N)
                res.append(-m.imag / d[w][1] - 1.0)
        return np.sqrt(np.mean(np.array(res) ** 2))

    r = minimize_scalar(cost, bounds=(0.05, 0.95), method='bounded')
    return r.x


# ---------------------------------------------------------------------------
# 3. 多極 Debye フィット（提供スクリプトと同一方針）
# ---------------------------------------------------------------------------

def multi_debye(f, eps_inf, poles):
    """eps*(w) = eps_inf + sum_i delta_eps_i / (1 + i w tau_i)。eps'' > 0 で返す。"""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    eps_star = np.full_like(w, eps_inf, dtype=complex)
    for de, tau in poles:
        eps_star += de / (1 + 1j * w * tau)
    return eps_star.real, -eps_star.imag


def fit_debye(n_poles, target_re, target_im, tau_hint, de_hint):
    """任意のスペクトル（実測でも混合則でも）を n 極 Debye でフィットする。

    eps_inf も自由変数にしている点が提供スクリプトとの違い。混合則で作った
    スペクトルは eps_inf が既知でないため。GPR 帯域を 5 倍重み付けする。
    """
    if n_poles == 1:
        tau_seeds = [tau_hint]
    elif n_poles == 2:
        tau_seeds = [tau_hint * 0.2, tau_hint * 5.0]
    else:
        tau_seeds = [tau_hint * 0.05, tau_hint * 1.0, tau_hint * 20.0]
    tau_seeds = [max(t, TAU_MIN * 1.1) for t in tau_seeds]

    eps_inf_0 = max(target_re[-1], 1.0)
    x0 = np.array([eps_inf_0] + [de_hint / n_poles] * n_poles
                  + [np.log(t) for t in tau_seeds])
    lo = [1.0] + [0.0] * n_poles + [np.log(TAU_MIN)] * n_poles
    hi = [target_re[0] * 1.5] + [de_hint * 5 + 1e-3] * n_poles \
        + [np.log(tau_hint * 200)] * n_poles

    w_freq = np.ones(N_F)
    w_freq[(freqs >= GPR_LO) & (freqs <= GPR_HI * 2)] *= 5.0

    def residuals(x):
        eps_inf = x[0]
        des = x[1:1 + n_poles]
        taus = np.exp(x[1 + n_poles:])
        d_re, d_im = multi_debye(freqs, eps_inf, list(zip(des, taus)))
        return np.concatenate([(d_re - target_re) / target_re * w_freq,
                               (d_im - target_im) / np.maximum(target_im, 1e-9) * w_freq])

    r = least_squares(residuals, x0, bounds=(lo, hi), method='trf',
                      ftol=1e-12, xtol=1e-12, max_nfev=20000)
    eps_inf = r.x[0]
    des = r.x[1:1 + n_poles]
    taus = np.exp(r.x[1 + n_poles:])
    order = np.argsort(taus)
    return eps_inf, list(zip(des[order], taus[order]))


def rms_pct(a, b):
    return np.sqrt(np.mean(((a - b) / np.maximum(np.abs(b), 1e-12)) ** 2)) * 100


# ---------------------------------------------------------------------------
# 4. 検証本体
# ---------------------------------------------------------------------------

def spectrum_from_rule(rule, wt_pct, N_opt):
    """混合則で作った複素誘電率スペクトル (eps', eps'')。"""
    fv = volume_fraction(wt_pct)
    re, im = np.empty(N_F), np.empty(N_F)
    for i, f in enumerate(freqs):
        e_i = cole_cole(f, **COLE_COLE['pure'])
        m = apply_rule(rule, EPS_BYTOWNITE, e_i, fv, N_opt)
        re[i], im[i] = m.real, -m.imag
    return re, im


def compare_at_measured_bands(N_opt, _p):
    """Table 7 の実測点で混合則を検証する（フィットを介さない直接比較）。"""
    _p("\n" + "=" * 78)
    _p("検証 1: Table 7 の実測点との直接比較")
    _p("=" * 78)

    summary = {}
    for rule in MIXING_RULES:
        err_re, err_im = [], []
        _p(f"\n--- {rule} ---")
        _p(f"{'band':>5}{'f[GHz]':>9}{'wt%':>6}{'vol%':>7} | "
           f"{'混合則 eps':>11}{'実測':>8}{'誤差':>9} | {'混合則 eps_im':>14}{'実測':>8}{'誤差':>9}")
        for f in sorted(MEASURED):
            e_i = cole_cole(f, **COLE_COLE['pure'])
            for w in VALIDATION_WT:
                fv = volume_fraction(w)
                m = apply_rule(rule, EPS_BYTOWNITE, e_i, fv, N_opt)
                meas_re, meas_im = MEASURED[f][w]
                d_re = 100 * (m.real / meas_re - 1)
                d_im = 100 * (-m.imag / meas_im - 1)
                err_re.append(d_re)
                err_im.append(d_im)
                _p(f"{BAND_NAME[f]:>5}{f / 1e9:>9.3f}{w:>6}{fv * 100:>7.2f} | "
                   f"{m.real:>11.3f}{meas_re:>8.2f}{d_re:>8.2f}% | "
                   f"{-m.imag:>14.4f}{meas_im:>8.3f}{d_im:>8.1f}%")
        rms_re = np.sqrt(np.mean(np.array(err_re) ** 2))
        rms_im = np.sqrt(np.mean(np.array(err_im) ** 2))
        summary[rule] = (rms_re, rms_im)
        _p(f"  RMS: eps' {rms_re:.2f}%,  eps'' {rms_im:.2f}%")

    _p("\n" + "-" * 78)
    _p("混合則まとめ（12 点 = 4 帯域 x 3 試料の RMS 誤差）")
    _p(f"{'混合則':<16}{'eps RMS':>12}{'eps_im RMS':>14}{'判定':>26}")
    for rule, (a, b) in summary.items():
        verdict = 'eps は良好だが eps_im が不一致' if (a < 10 and b > 10) else \
                  ('両方良好' if (a < 10 and b <= 10) else '不一致')
        _p(f"{rule:<16}{a:>11.2f}%{b:>13.2f}%{verdict:>26}")
    return summary


def compare_debye_fits(N_opt, _p, buf_figs):
    """実測 Cole-Cole と混合則スペクトルを、それぞれ Debye でフィットして比較。"""
    _p("\n" + "=" * 78)
    _p("検証 2: 多極 Debye フィットの比較（gprMax に渡すパラメータ）")
    _p("=" * 78)

    for w in VALIDATION_WT:
        cc = COLE_COLE[w]
        meas_re, meas_im = np.empty(N_F), np.empty(N_F)
        for i, f in enumerate(freqs):
            e = cole_cole(f, **cc)
            meas_re[i], meas_im[i] = e.real, -e.imag

        mg_re, mg_im = spectrum_from_rule('MG', w, N_opt)

        _p("\n" + "-" * 78)
        _p(f"{w} wt% ilmenite  (vol% = {volume_fraction(w) * 100:.2f})")
        _p("-" * 78)

        gpr = (freqs >= GPR_LO) & (freqs <= GPR_HI)
        _p(f"  スペクトル同士の GPR 帯域 RMS 差: "
           f"eps' {rms_pct(mg_re[gpr], meas_re[gpr]):.2f}%, "
           f"eps'' {rms_pct(mg_im[gpr], meas_im[gpr]):.2f}%")

        for src, (re, im) in [('Boivin 実測 (Cole-Cole)', (meas_re, meas_im)),
                              ('MG 混合則', (mg_re, mg_im))]:
            _p(f"\n  [{src}]")
            for n in [1, 2, 3]:
                eps_inf, poles = fit_debye(n, re, im, cc['tau'], cc['delta_eps'])
                d_re, d_im = multi_debye(freqs, eps_inf, poles)
                pole_str = ''.join(f' {de:.6f} {tau:.6e}' for de, tau in poles)
                _p(f"    {n}極: eps_inf={eps_inf:.4f}, sum(De)={sum(d for d, _ in poles):.4f}"
                   f"  GPR RMS: eps' {rms_pct(d_re[gpr], re[gpr]):.2f}%,"
                   f" eps'' {rms_pct(d_im[gpr], im[gpr]):.2f}%")
                if n == 2:
                    _p(f"      #material: {eps_inf:.6f} 0 1 0 mat_{w}wt")
                    _p(f"      #add_dispersion_debye: {n}{pole_str} mat_{w}wt")

        buf_figs.append((w, meas_re, meas_im, mg_re, mg_im))


def diagnose_relaxation_shift(_p):
    """緩和時間が希釈でどう動くかを調べる。

    混合則は介在物の tau をほぼ保存する。したがって実測の tau が
    イルメナイト量とともに動いているなら、混合則では原理的に再現できない。
    """
    _p("\n" + "=" * 78)
    _p("診断: 緩和時間の希釈依存性")
    _p("=" * 78)
    _p(f"{'試料':>8}{'tau [ps]':>12}{'f_peak [GHz]':>15}{'alpha':>9}")
    for k in [10, 15, 20, 'pure']:
        cc = COLE_COLE[k]
        _p(f"{str(k):>8}{cc['tau'] * 1e12:>12.1f}"
           f"{1.0 / (2 * np.pi * cc['tau']) / 1e9:>15.3f}{cc['alpha']:>9.3f}")
    ratio = COLE_COLE['pure']['tau'] / np.mean([COLE_COLE[w]['tau'] for w in VALIDATION_WT])
    _p(f"\n  純イルメナイトの tau は混合物の平均の {ratio:.1f} 倍")
    _p("  = 希釈すると緩和ピークが高周波側へ大きく移動している")
    _p("  混合則は介在物の tau を保存するため、この移動を再現できない。")
    _p("  これが eps'' が合わない直接の原因であり、脱分極係数や体積分率の")
    _p("  調整では解消しない（形状ではなく時定数の問題であるため）。")


def extrapolate_low_ilmenite(N_opt, _p):
    """低イルメナイト域へ混合則で外挿し、非分散仮定と比較する。"""
    _p("\n" + "=" * 78)
    _p("参考: 低イルメナイト域への外挿（Boivin 未測定域）")
    _p("=" * 78)
    _p("Boivin は 1, 5 wt% では有意な分散を検出できずフィット不能と報告している。")
    _p("混合則による外挿がそれと整合するかを見る。")
    _p(f"\n{'wt%':>6}{'vol%':>8}{'eps@0.5':>10}{'eps@1.25':>10}{'eps@2.0':>10}"
       f"{'tand@0.5':>11}{'tand@1.25':>11}{'tand@2.0':>11}{'tand比':>9}")
    for w in EXTRAPOLATE_WT + VALIDATION_WT:
        fv = volume_fraction(w)
        vals = []
        for f in (0.5e9, 1.25e9, 2.0e9):
            e_i = cole_cole(f, **COLE_COLE['pure'])
            m = apply_rule('MG', EPS_BYTOWNITE, e_i, fv, N_opt)
            vals.append((m.real, -m.imag))
        td = [im / re for re, im in vals]
        _p(f"{w:>6.1f}{fv * 100:>8.2f}{vals[0][0]:>10.4f}{vals[1][0]:>10.4f}{vals[2][0]:>10.4f}"
           f"{td[0]:>11.5f}{td[1]:>11.5f}{td[2]:>11.5f}{td[2] / td[0]:>9.3f}")
    _p("\n  参考: 純バイトウナイト（Table 7 実測）は全帯域で tand = 0.002（tand比 = 1.000）")


# ---------------------------------------------------------------------------
# 5. 作図
# ---------------------------------------------------------------------------

def plot_comparison(w, meas_re, meas_im, mg_re, mg_im, N_opt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Mixing rule vs Boivin+2022 measurement — {w} wt% ilmenite '
                 f'(vol% = {volume_fraction(w) * 100:.2f})', fontsize=12)
    ax_re, ax_im, ax_td, ax_err = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def deco(ax):
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (GHz)')
        ax.axvspan(GPR_LO / 1e9, GPR_HI / 1e9, alpha=0.12, color='gray',
                   label='LUPEX GPR band')
        ax.axvline(FC_GPR / 1e9, color='gray', lw=0.8, ls='-.', alpha=0.6)
        ax.set_xlim(F_MIN / 1e9, F_MAX / 1e9)
        ax.grid(True, which='both', alpha=0.2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x:.1g}' if x < 1 else f'{int(x)}'))

    rules_to_plot = ['MG', 'MG_fitN', 'Bruggeman', 'CRIM']
    colors = {'MG': 'r', 'MG_fitN': 'darkorange', 'Bruggeman': 'b', 'CRIM': 'purple'}

    ax_re.plot(freqs / 1e9, meas_re, 'k-', lw=2.5, label='Boivin measured', zorder=5)
    ax_im.plot(freqs / 1e9, meas_im, 'k-', lw=2.5, label='Boivin measured', zorder=5)
    ax_td.plot(freqs / 1e9, meas_im / meas_re, 'k-', lw=2.5,
               label='Boivin measured', zorder=5)
    for rule in rules_to_plot:
        re, im = spectrum_from_rule(rule, w, N_opt)
        lab = rule if rule != 'MG_fitN' else f'MG (N={N_opt:.3f})'
        ax_re.plot(freqs / 1e9, re, color=colors[rule], ls='--', lw=1.6, label=lab)
        ax_im.plot(freqs / 1e9, im, color=colors[rule], ls='--', lw=1.6, label=lab)
        ax_td.plot(freqs / 1e9, im / re, color=colors[rule], ls='--', lw=1.6, label=lab)
        ax_err.plot(freqs / 1e9, 100 * np.abs(im / meas_im - 1),
                    color=colors[rule], ls='--', lw=1.6, label=lab)

    # Table 7 の実測点を重ねる
    for f in sorted(MEASURED):
        ax_re.plot(f / 1e9, MEASURED[f][w][0], 'ks', ms=7, zorder=6)
        ax_im.plot(f / 1e9, MEASURED[f][w][1], 'ks', ms=7, zorder=6)
        ax_td.plot(f / 1e9, MEASURED[f][w][1] / MEASURED[f][w][0], 'ks', ms=7, zorder=6)

    for ax in (ax_re, ax_im, ax_td, ax_err):
        deco(ax)
    ax_re.set_ylabel(r"$\varepsilon_r'$"); ax_re.set_title('Real permittivity')
    ax_re.legend(fontsize=8)
    ax_im.set_ylabel(r"$\varepsilon_r''$"); ax_im.set_title('Imaginary permittivity')
    ax_im.set_yscale('log'); ax_im.legend(fontsize=8)
    ax_td.set_ylabel(r'$\tan\delta$'); ax_td.set_title('Loss tangent')
    ax_td.legend(fontsize=8)
    ax_err.set_ylabel(r"$|\varepsilon''_{mix}/\varepsilon''_{meas} - 1|$ (%)")
    ax_err.set_title('Relative error in imaginary permittivity')
    ax_err.axhline(10, color='gray', lw=0.8, ls='--')
    ax_err.set_yscale('log'); ax_err.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f'mixing_vs_measured_{w}wt'
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.{ext}'),
                    dpi=150 if ext == 'png' else 300, bbox_inches='tight')
    print(f'プロット保存: {OUTPUT_DIR}/{name}.png/.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------

def main():
    buf = io.StringIO()

    def _p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=buf)

    _p("=" * 78)
    _p("混合則の検証: MG(bytownite + pure ilmenite) vs Boivin+2022 実測")
    _p("=" * 78)
    _p(f"体積分率モード : {VOLUME_FRACTION_MODE}")
    _p(f"母材 (bytownite): eps' = {EPS_BYTOWNITE.real}, "
       f"eps'' = {-EPS_BYTOWNITE.imag}  (Table 7、全帯域で一定 = 非分散)")
    _p(f"gprMax tau_min  : {TAU_MIN * 1e12:.2f} ps  (dx = {DX} m, 2D)")

    check_volume_model(_p)

    N_opt = fit_depolarisation_factor()
    _p(f"\n実測 eps'' に最も合う MG の脱分極係数: N = {N_opt:.4f}  "
       f"(球なら 1/3 = 0.3333)")

    compare_at_measured_bands(N_opt, _p)
    diagnose_relaxation_shift(_p)

    figs = []
    compare_debye_fits(N_opt, _p, figs)
    extrapolate_low_ilmenite(N_opt, _p)

    for w, mre, mim, gre, gim in figs:
        plot_comparison(w, mre, mim, gre, gim, N_opt)

    buf.write(f"\n[Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())
    print(f'\nAll results saved to: {out_path}')


if __name__ == '__main__':
    main()