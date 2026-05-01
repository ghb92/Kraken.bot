#!/usr/bin/env python3
"""
Backtester pour la stratégie de botv4.py.

Réutilise les fonctions add_indicators(), score(), trend_1h logic
et la même mécanique d'entrée/sortie (SL/TP/trailing) que le bot live.

USAGE:
    python3 backtest.py
    python3 backtest.py --pair XBTUSD --capital 10000

DONNÉES:
    Par défaut, fetch via l'API publique Kraken (~720 dernières bougies 15min
    soit ~7.5 jours, et ~720 bougies 1h soit ~30 jours pour le filtre tendance).
    Pour un backtest plus long, fournis un CSV via --csv path/to/data.csv
    avec colonnes: t,o,h,l,c,v
"""

import argparse
import urllib.request
import json
import time
import sys
import math
import pandas as pd
import numpy as np

from botv4 import (
    add_indicators, score,
    RISK_PER_TRADE, MAX_POSITION_PCT, TRAIL_PCT,
    TRADE_H_MIN, TRADE_H_MAX, COOLDOWN,
)

KRAKEN_OHLC_URL = 'https://api.kraken.com/0/public/OHLC'


def fetch_kraken_ohlc(pair, interval):
    """Fetch OHLC depuis l'API publique Kraken (~720 dernières bougies)."""
    url = f'{KRAKEN_OHLC_URL}?pair={pair}&interval={interval}'
    req = urllib.request.Request(url, headers={'User-Agent': 'KrakenBacktest/1.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    if data.get('error'):
        raise RuntimeError(f'Kraken API error: {data["error"]}')
    result = data['result']
    key = [k for k in result.keys() if k != 'last'][0]
    df = pd.DataFrame(result[key], columns=['t', 'o', 'h', 'l', 'c', 'v', 'vw', 'n'])
    df['t'] = pd.to_numeric(df['t'])
    return df


def load_csv(path):
    df = pd.read_csv(path)
    needed = {'t', 'o', 'h', 'l', 'c', 'v'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Colonnes manquantes dans le CSV: {missing}')
    if 'vw' not in df.columns: df['vw'] = 0
    if 'n' not in df.columns: df['n'] = 0
    return df


def compute_trend_1h(df_1h):
    """Annote df_1h avec la colonne 'trend' (bull/bear/neutral)."""
    df_1h = df_1h.copy()
    cond_bull = (df_1h['ef'] > df_1h['es']) & (df_1h['es'] > df_1h['et'])
    cond_bear = (df_1h['ef'] < df_1h['es']) & (df_1h['es'] < df_1h['et'])
    df_1h['trend'] = np.where(cond_bull, 'bull',
                       np.where(cond_bear, 'bear', 'neutral'))
    return df_1h


def trend_at(df_1h_indexed, ts):
    """Retourne la dernière tendance 1h connue à l'instant ts."""
    sub = df_1h_indexed[df_1h_indexed['t'] <= ts]
    if len(sub) == 0:
        return 'neutral'
    return sub['trend'].iloc[-1]


def position_size(bal, price, sl_distance, risk):
    if sl_distance <= 0 or price <= 0:
        return 0.0
    risk_based = (bal * risk) / sl_distance
    cap        = (bal * MAX_POSITION_PCT) / price
    return min(risk_based, cap)


def run_backtest(df_15m, df_1h, capital_init=10000.0, verbose=False):
    """
    Rejoue la stratégie sur df_15m bougie par bougie.
    Hypothèses de simulation:
      - Entrée et sortie à la clôture de la bougie courante (close[i]).
      - Filtres SL/TP/trailing évalués sur le close de chaque bougie
        (plus pessimiste que high/low intra-bougie).
      - Frais: 0.26% taker par côté (frais standard Kraken).
    """
    FEE = 0.0026
    df_1h = compute_trend_1h(df_1h)

    capital  = capital_init
    equity   = []        # liste de (timestamp, equity)
    trades   = []        # liste de dicts (entry, exit, pnl, reason, ...)
    pos      = None      # position courante: dict ou None
    sl_streak = 0
    last_sl_ts = -math.inf

    # Warmup: laisser les indicateurs se stabiliser
    start = max(60, df_15m.iloc[0].name + 60)

    for i in range(start, len(df_15m)):
        row = df_15m.iloc[i]
        ts  = row['t']
        p   = row['c']
        atr = row['atr']
        if pd.isna(atr) or pd.isna(row['rsi']) or pd.isna(row['macd']):
            equity.append((ts, capital + (mark_to_market(pos, p) if pos else 0)))
            continue

        sl_dist = atr * 1.5
        tp_dist = atr * 2.5

        # Filtre heure de trading
        hour_utc = pd.to_datetime(ts, unit='s').hour
        in_hours = TRADE_H_MIN <= hour_utc < TRADE_H_MAX

        # ── Gestion position ouverte
        if pos is not None:
            chg = (p - pos['entry']) / pos['entry'] * (1 if pos['dir'] == 'buy' else -1)

            # Update peak
            if pos['dir'] == 'buy' and p > pos['peak']:
                pos['peak'] = p
            elif pos['dir'] == 'sell' and p < pos['peak']:
                pos['peak'] = p

            # Trailing stop
            trail = False
            if pos['dir'] == 'buy':
                trail = p <= pos['peak'] * (1 - TRAIL_PCT) and chg > 0
            else:
                trail = p >= pos['peak'] * (1 + TRAIL_PCT) and chg > 0

            exit_reason = None
            if chg >= pos['tp'] / pos['entry']:
                exit_reason = 'TP'
            elif trail:
                exit_reason = 'TRAIL'
            elif chg <= -pos['sl'] / pos['entry']:
                exit_reason = 'SL'

            if exit_reason:
                gross = (p - pos['entry']) * pos['size'] * (1 if pos['dir'] == 'buy' else -1)
                fees  = (pos['entry'] * pos['size'] + p * pos['size']) * FEE
                net   = gross - fees
                capital += net
                trades.append({
                    'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                    'dir': pos['dir'], 'entry': pos['entry'], 'exit': p,
                    'size': pos['size'], 'pnl': net, 'reason': exit_reason,
                    'duration_bars': i - pos['entry_idx'],
                })
                if exit_reason == 'SL':
                    sl_streak += 1
                    last_sl_ts = ts
                else:
                    sl_streak = 0
                if verbose:
                    dt = pd.to_datetime(ts, unit='s')
                    print(f'  {dt} {exit_reason} {pos["dir"]} pnl={net:+.2f} cap={capital:.2f}')
                pos = None

        # ── Entrée
        if pos is None and in_hours:
            sc = score(df_15m.iloc[:i+1])
            cooldown_ok = (ts - last_sl_ts) > COOLDOWN
            risk = current_risk(sl_streak)
            trend = trend_at(df_1h, ts)

            if sc >= 50 and trend != 'bear' and cooldown_ok:
                sz = position_size(capital, p, sl_dist, risk)
                if sz > 0:
                    pos = {'dir': 'buy', 'entry': p, 'size': sz,
                           'sl': sl_dist, 'tp': tp_dist, 'peak': p,
                           'entry_ts': ts, 'entry_idx': i}
            elif sc <= -50 and trend != 'bull' and cooldown_ok:
                sz = position_size(capital, p, sl_dist, risk)
                if sz > 0:
                    pos = {'dir': 'sell', 'entry': p, 'size': sz,
                           'sl': sl_dist, 'tp': tp_dist, 'peak': p,
                           'entry_ts': ts, 'entry_idx': i}

        equity.append((ts, capital + (mark_to_market(pos, p) if pos else 0)))

    return trades, equity


def current_risk(sl_streak):
    if sl_streak >= 3: return RISK_PER_TRADE * 0.5
    if sl_streak >= 2: return RISK_PER_TRADE * 0.75
    return RISK_PER_TRADE


def mark_to_market(pos, price):
    if pos is None: return 0
    return (price - pos['entry']) * pos['size'] * (1 if pos['dir'] == 'buy' else -1)


def metrics(trades, equity, capital_init):
    if not trades:
        return {'n_trades': 0, 'final_equity': capital_init,
                'total_return_pct': 0, 'win_rate': 0, 'profit_factor': 0,
                'max_drawdown_pct': 0, 'sharpe': 0, 'avg_win': 0, 'avg_loss': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x <= 0]
    eq = pd.DataFrame(equity, columns=['t', 'eq'])
    peak = eq['eq'].cummax()
    dd = (eq['eq'] - peak) / peak
    returns = eq['eq'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * math.sqrt(365 * 24 * 4)
              if returns.std() > 0 else 0)
    final = equity[-1][1]
    return {
        'n_trades': len(trades),
        'final_equity': final,
        'total_return_pct': (final / capital_init - 1) * 100,
        'win_rate': len(wins) / len(trades) * 100,
        'profit_factor': (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf'),
        'max_drawdown_pct': dd.min() * 100,
        'sharpe': sharpe,
        'avg_win': (sum(wins) / len(wins)) if wins else 0,
        'avg_loss': (sum(losses) / len(losses)) if losses else 0,
        'reasons': pd.Series([t['reason'] for t in trades]).value_counts().to_dict(),
    }


def print_report(pair, m, period_days):
    print(f'\n  ═══ {pair} ═══ (période ~{period_days:.1f} jours)')
    print(f'  Trades         : {m["n_trades"]}')
    if m['n_trades'] == 0:
        print('  Aucun trade exécuté sur la période.')
        return
    print(f'  Win rate       : {m["win_rate"]:.1f}%')
    print(f'  Profit factor  : {m["profit_factor"]:.2f}')
    print(f'  Capital final  : ${m["final_equity"]:,.2f}')
    print(f'  Return total   : {m["total_return_pct"]:+.2f}%')
    print(f'  Drawdown max   : {m["max_drawdown_pct"]:.2f}%')
    print(f'  Sharpe (annu.) : {m["sharpe"]:.2f}')
    print(f'  Win moyen      : {m["avg_win"]:+.2f}$')
    print(f'  Loss moyenne   : {m["avg_loss"]:+.2f}$')
    print(f'  Sorties        : {m["reasons"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', nargs='+', default=['XBTUSD', 'ETHUSD', 'SOLUSD'])
    ap.add_argument('--capital', type=float, default=10000.0)
    ap.add_argument('--csv-15m', help='CSV bougies 15min (au lieu de fetch Kraken)')
    ap.add_argument('--csv-1h', help='CSV bougies 1h (au lieu de fetch Kraken)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print('═' * 60)
    print('  BACKTEST botv4 — stratégie RSI+MACD+EMA+BB+ATR')
    print('═' * 60)
    print(f'  Capital initial  : ${args.capital:,.2f}')
    print(f'  Risque/trade     : {RISK_PER_TRADE*100:.2f}%')
    print(f'  Position max     : {MAX_POSITION_PCT*100:.0f}% du capital')
    print(f'  Trailing stop    : {TRAIL_PCT*100:.1f}%')
    print(f'  Heures trading   : {TRADE_H_MIN}h-{TRADE_H_MAX}h UTC')
    print(f'  Frais simulés    : 0.26% taker (Kraken)')

    summary = []
    for pair in args.pairs:
        print(f'\n  → Chargement {pair}...')
        try:
            if args.csv_15m:
                df_15m_raw = load_csv(args.csv_15m)
                df_1h_raw  = load_csv(args.csv_1h) if args.csv_1h else None
            else:
                df_15m_raw = fetch_kraken_ohlc(pair, 15)
                time.sleep(1)  # respect rate limit
                df_1h_raw  = fetch_kraken_ohlc(pair, 60)
                time.sleep(1)
        except Exception as e:
            print(f'  ❌ Erreur fetch {pair}: {e}')
            continue

        if df_1h_raw is None:
            print(f'  ❌ Manque les données 1h pour {pair}')
            continue

        df_15m = add_indicators(df_15m_raw).reset_index(drop=True)
        df_1h  = add_indicators(df_1h_raw).reset_index(drop=True)

        period_days = (df_15m['t'].iloc[-1] - df_15m['t'].iloc[0]) / 86400
        trades, equity = run_backtest(df_15m, df_1h, args.capital, args.verbose)
        m = metrics(trades, equity, args.capital)
        print_report(pair, m, period_days)
        summary.append((pair, m, period_days))

    # Résumé multi-paires
    print('\n' + '═' * 60)
    print('  RÉSUMÉ')
    print('═' * 60)
    print(f'  {"Pair":<10} {"Trades":>7} {"WinRate":>9} {"PF":>6} {"Return":>9} {"MaxDD":>9}')
    for pair, m, _ in summary:
        if m['n_trades'] == 0:
            print(f'  {pair:<10} {"0":>7} {"-":>9} {"-":>6} {"-":>9} {"-":>9}')
        else:
            pf = m['profit_factor']
            pf_str = f'{pf:.2f}' if pf != float('inf') else '∞'
            print(f'  {pair:<10} {m["n_trades"]:>7} {m["win_rate"]:>8.1f}% '
                  f'{pf_str:>6} {m["total_return_pct"]:>+8.2f}% {m["max_drawdown_pct"]:>+8.2f}%')


if __name__ == '__main__':
    main()
