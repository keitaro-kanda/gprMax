#!/usr/bin/env python3
"""
watch_progress.py -- gprMax の計算ログを監視し、進捗を1行で表示する

使い方（計算を実行しているのとは別のターミナルで）:
    python3 watch_progress.py                 # このスクリプトのあるフォルダ以下を監視
    python3 watch_progress.py /path/to/ascan_v01
    python3 watch_progress.py . --pattern '*.out.log'   # ログファイル名のパターンを指定
終了は Ctrl+C（計算には影響しない）。

仕組み:
    フォルダ以下で「いちばん最近更新されたログファイル」を探し、
    その末尾にある gprMax（tqdm）の進捗表示
        "...  45%|####      | 12345/27000 [00:15<00:18, 830.12it/s]"
    を読み取って、ケース名・ステップ・速度・残り時間を1行にまとめて上書き表示する。
"""
import argparse
import fnmatch
import os
import re
import sys
import time

PROG = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:?]+),\s*([0-9.]+)\s*(it/s|s/it)"
)


def newest_log(root, pattern):
    newest, newest_t = None, -1.0
    for d, _, files in os.walk(root):
        for f in fnmatch.filter(files, pattern):
            p = os.path.join(d, f)
            try:
                t = os.path.getmtime(p)
            except OSError:
                continue
            if t > newest_t:
                newest, newest_t = p, t
    return newest


def last_progress(path, nbytes=16384):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            tail = fh.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    # tqdm は \r で同じ行を上書きするので、\r と \n の両方で区切って最後の進捗を探す
    for chunk in reversed(re.split(r"[\r\n]", tail)):
        m = PROG.search(chunk)
        if m:
            return m
    return None


def main():
    ap = argparse.ArgumentParser(description="gprMax の進捗を1行で表示")
    ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--pattern", default="*.log", help="ログファイル名のパターン（既定: *.log）")
    ap.add_argument("--interval", type=float, default=2.0, help="更新間隔 [s]（既定: 2）")
    ap.add_argument("--rescan", type=float, default=10.0, help="最新ログを探し直す間隔 [s]（既定: 10）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    log, last_scan = None, 0.0
    try:
        while True:
            now = time.time()
            if log is None or now - last_scan > args.rescan:
                log = newest_log(root, args.pattern)
                last_scan = now
            if log is None:
                line = f"ログファイル（{args.pattern}）が {root} 以下に見つかりません"
            else:
                case = os.path.relpath(log, root)
                m = last_progress(log)
                if m is None:
                    line = f"{case}  （進捗表示を待っています）"
                else:
                    pct, cur, tot, elapsed, remain, rate, unit = m.groups()
                    r = float(rate)
                    sps = r if unit == "it/s" else (1.0 / r if r else 0.0)
                    line = (f"{case}  {pct:>3}%  {int(cur):,}/{int(tot):,} steps  "
                            f"{sps:,.1f} steps/s  経過 {elapsed}  残り {remain}")
                try:
                    age = now - os.path.getmtime(log)
                except OSError:
                    age = 0
                if age > 60:
                    line += f"  （{int(age)} 秒更新なし）"
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
