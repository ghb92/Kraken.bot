#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         BOT DE TRADING KRAKEN V4                             ║
║  Stratégie: RSI + MACD + EMA + Bollinger + ATR               ║
║  Nouveautés V4:                                              ║
║    ✅ Trailing Stop Loss                                      ║
║    ✅ Filtre tendance 1h                                      ║
║    ✅ Réduction taille après SL consécutifs                   ║
║    ✅ Filtre heures de trading (7h-23h UTC)                   ║
║    ✅ Retry automatique API                                   ║
║    ✅ Logging CSV des trades                                  ║
║    ✅ Notifications Telegram                                  ║
║    ✅ Dashboard terminal                                      ║
╚══════════════════════════════════════════════════════════════╝

INSTALLATION:
    pip install krakenex pandas numpy schedule

CONFIGURATION:
    1. Remplace KEY / SEC par tes clés API Kraken
    2. Remplace TELEGRAM_TOKEN et TELEGRAM_CHAT_ID
       → Crée un bot via @BotFather sur Telegram
       → Récupère ton chat_id via @userinfobot

LANCEMENT:
    python3 botv4.py
"""

import time
import csv
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Imports lazy: krakenex et schedule ne sont nécessaires qu'en mode live.
# Le backtester importe ce module mais n'a pas besoin de ces deps.
try:
    import krakenex
    import schedule
    _LIVE_DEPS_OK = True
except ImportError:
    krakenex = None
    schedule = None
    _LIVE_DEPS_OK = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

KEY = 'VOTRE_API_KEY_KRAKEN'
SEC = 'VOTRE_API_SECRET_KRAKEN'

api = krakenex.API(key=KEY, secret=SEC) if _LIVE_DEPS_OK else None

PAIRS            = ['XBTUSD', 'ETHUSD', 'SOLUSD']
RISK_PER_TRADE   = 0.01    # 1% du capital RÉELLEMENT risqué par trade (perte si SL atteint)
MAX_POSITION_PCT = 0.30    # Position max = 30% du capital (cap si SL serré → sizing énorme)
LIMIT            = 0.10    # Circuit breaker: arrêt si -10% (réalisé + non-réalisé)
COOLDOWN         = 1800    # 30 min entre trades après un SL
TRAIL_PCT        = 0.015   # Trailing stop: 1.5% sous le plus haut atteint
TRADE_H_MIN      = 7       # Heure de début trading (UTC)
TRADE_H_MAX      = 23      # Heure de fin trading (UTC)
LOG_FILE         = 'trades_log.csv'
MAX_RETRIES      = 3       # Tentatives max sur appels API

# État du bot
positions   = {}   # {pair: {entry, size, sl, tp, dir, peak}}
pnl         = 0.0
last_sl     = {}
sl_streak   = 0    # Compteur SL consécutifs
trade_count = 0

# ─────────────────────────────────────────────
# LOGGING CSV
# ─────────────────────────────────────────────

def log_trade(pair, direction, entry, exit_price, size, result, reason):
    """Enregistre un trade terminé dans le fichier CSV."""
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['date', 'pair', 'dir', 'entry', 'exit', 'size', 'pnl', 'reason'])
        profit = (exit_price - entry) * size * (1 if direction == 'buy' else -1)
        w.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            pair, direction,
            round(entry, 2), round(exit_price, 2),
            round(size, 6), round(profit, 2), reason
        ])

# ─────────────────────────────────────────────
# RETRY API
# ─────────────────────────────────────────────

def api_call(func, *args, **kwargs):
    """Appel API avec retry automatique (3 tentatives)."""
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            if not result.get('error'):
                return result
            print(f'  ⚠️  Erreur API: {result["error"]} (tentative {attempt+1}/{MAX_RETRIES})')
        except Exception as e:
            print(f'  ⚠️  Exception API: {e} (tentative {attempt+1}/{MAX_RETRIES})')
        time.sleep(2 ** attempt)  # Backoff: 1s, 2s, 4s
    return None

# ─────────────────────────────────────────────
# FONCTIONS DE DONNÉES
# ─────────────────────────────────────────────

def price(p):
    """Retourne le prix actuel de la paire."""
    r = api_call(api.query_public, 'Ticker', {'pair': p})
    return float(list(r['result'].values())[0]['c'][0]) if r else None

def balance():
    """Retourne le solde USD disponible."""
    r = api_call(api.query_private, 'Balance')
    return float(r['result'].get('ZUSD', 0)) if r else 0.0

def order(pair, side, size):
    """Place un ordre market. Retourne le txid ou None."""
    r = api_call(api.query_private, 'AddOrder', {
        'pair':      pair,
        'type':      side,
        'ordertype': 'market',
        'volume':    str(round(size, 6))
    })
    return r['result']['txid'][0] if r else None

# ─────────────────────────────────────────────
# INDICATEURS TECHNIQUES (15min)
# ─────────────────────────────────────────────

def add_indicators(df):
    """Ajoute RSI, EMA, MACD, Bollinger, ATR à un DataFrame OHLCV.

    Fonction pure (pas d'I/O) → réutilisable par le backtester.
    Attend les colonnes: 't', 'o', 'h', 'l', 'c', 'v'.
    """
    df = df.copy()
    df[['c', 'h', 'l', 'v']] = df[['c', 'h', 'l', 'v']].astype(float)

    # RSI
    d = df['c'].diff()
    g = d.clip(lower=0).ewm(com=13, min_periods=14).mean()
    lo = (-d).clip(lower=0).ewm(com=13, min_periods=14).mean()
    df['rsi'] = 100 - (100 / (1 + g / lo))

    # EMA 9 / 21 / 50
    df['ef'] = df['c'].ewm(span=9,  adjust=False).mean()
    df['es'] = df['c'].ewm(span=21, adjust=False).mean()
    df['et'] = df['c'].ewm(span=50, adjust=False).mean()

    # MACD (12/26/9)
    e12 = df['c'].ewm(span=12, adjust=False).mean()
    e26 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = e12 - e26
    df['sig']  = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger (20 périodes, 2 écarts-types)
    df['bb_mid'] = df['c'].rolling(20).mean()
    df['bb_up']  = df['bb_mid'] + 2 * df['c'].rolling(20).std()
    df['bb_low'] = df['bb_mid'] - 2 * df['c'].rolling(20).std()

    # ATR (14 périodes)
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - df['c'].shift()).abs(),
        (df['l'] - df['c'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    return df

def ohlcv(pair, interval=15):
    """Récupère les bougies depuis Kraken et calcule les indicateurs."""
    r = api_call(api.query_public, 'OHLC', {'pair': pair, 'interval': interval})
    if not r:
        return None
    k  = list(r['result'].keys())[0]
    df = pd.DataFrame(r['result'][k],
                      columns=['t', 'o', 'h', 'l', 'c', 'v', 'vw', 'n'])
    return add_indicators(df)

# ─────────────────────────────────────────────
# FILTRE TENDANCE 1H (NOUVEAU V4)
# ─────────────────────────────────────────────

def trend_1h(pair):
    """
    Retourne 'bull', 'bear' ou 'neutral' selon la tendance 1h.
    Basé sur EMA50 et EMA21 sur timeframe 1h.
    """
    df = ohlcv(pair, interval=60)
    if df is None or len(df) < 51:
        return 'neutral'
    l = df.iloc[-1]
    if l['ef'] > l['es'] > l['et']:
        return 'bull'
    elif l['ef'] < l['es'] < l['et']:
        return 'bear'
    return 'neutral'

# ─────────────────────────────────────────────
# SCORE COMPOSITE -100 à +100
# ─────────────────────────────────────────────

def score(df):
    """
    Score composite entre -100 et +100.
    Score >= 50  → signal achat
    Score <= -50 → signal vente
    """
    l, p = df.iloc[-1], df.iloc[-2]
    s = 0

    # RSI (±30 points)
    if l['rsi'] < 35:   s += 30
    elif l['rsi'] > 65: s -= 30

    # MACD croisement (±30 points)
    if   l['macd'] > l['sig'] and p['macd'] <= p['sig']: s += 30
    elif l['macd'] < l['sig'] and p['macd'] >= p['sig']: s -= 30

    # EMA tendance (±20 points)
    if   l['c'] > l['et'] and l['ef'] > l['es']: s += 20
    elif l['c'] < l['et'] and l['ef'] < l['es']: s -= 20

    # Volume (±10 points)
    if l['v'] > df['v'].rolling(20).mean().iloc[-1] * 1.5: s += 10

    # Bollinger (±10 points)
    if   l['c'] <= l['bb_low']: s += 10
    elif l['c'] >= l['bb_up']:  s -= 10

    return s

# ─────────────────────────────────────────────
# FILTRE HEURES DE TRADING (NOUVEAU V4)
# ─────────────────────────────────────────────

def trading_hours_ok():
    """Retourne True si on est dans la plage horaire autorisée (UTC)."""
    h = datetime.utcnow().hour
    return TRADE_H_MIN <= h < TRADE_H_MAX

# ─────────────────────────────────────────────
# DASHBOARD TERMINAL (NOUVEAU V4)
# ─────────────────────────────────────────────

def dashboard():
    """Affiche un résumé de l'état du bot dans le terminal."""
    bal = balance()
    print('\n' + '═' * 55)
    print(f'  🤖 BOT V4 KRAKEN — {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    print('═' * 55)
    total = total_pnl()
    print(f'  💰 Solde USD     : ${bal:,.2f}')
    print(f'  📈 PnL réalisé   : {pnl:+.2f}$')
    print(f'  📊 PnL total     : {total:+.2f}$ (réalisé + non-réalisé)')
    print(f'  📊 Trades total  : {trade_count}')
    print(f'  🔴 SL streak     : {sl_streak} (risque/trade: {current_risk()*100:.2f}%)')
    print(f'  🕐 Heure UTC     : {datetime.utcnow().strftime("%H:%M")} | Trading: {"✅ OUI" if trading_hours_ok() else "❌ NON"}')
    if positions:
        print(f'  📂 Positions ouvertes ({len(positions)}):')
        for pair, pos in positions.items():
            p = price(pair)
            if p:
                chg = (p - pos['entry']) / pos['entry'] * (1 if pos['dir'] == 'buy' else -1)
                print(f'     {pair} {pos["dir"].upper()} | entrée:{pos["entry"]:.0f} | actuel:{p:.0f} | {chg*100:+.2f}%')
    else:
        print('  📂 Aucune position ouverte')
    print('═' * 55)

# ─────────────────────────────────────────────
# GESTION DU RISQUE DYNAMIQUE (NOUVEAU V4)
# ─────────────────────────────────────────────

def current_risk():
    """Retourne le % de capital risqué par trade, adapté selon les SL consécutifs."""
    if sl_streak >= 3:
        return RISK_PER_TRADE * 0.5   # divisé par 2 après 3 SL de suite
    elif sl_streak >= 2:
        return RISK_PER_TRADE * 0.75  # -25% après 2 SL de suite
    return RISK_PER_TRADE

def position_size(bal, price_now, sl_distance, risk):
    """
    Calcule la taille d'une position basée sur le risque réel.
    - bal: capital disponible
    - price_now: prix d'entrée
    - sl_distance: distance prix → SL (en $, pas en %)
    - risk: fraction du capital risquée si SL atteint

    Retourne size tel que perte au SL = bal * risk, capée à MAX_POSITION_PCT du capital.
    """
    if sl_distance <= 0 or price_now <= 0:
        return 0.0
    risk_based   = (bal * risk) / sl_distance
    notional_cap = (bal * MAX_POSITION_PCT) / price_now
    return min(risk_based, notional_cap)

def total_pnl():
    """PnL total = réalisé + non-réalisé sur toutes les positions ouvertes."""
    unrealized = 0.0
    for pair_, pos in positions.items():
        cur = price(pair_)
        if cur is None:
            continue
        unrealized += (cur - pos['entry']) * pos['size'] * (1 if pos['dir'] == 'buy' else -1)
    return pnl + unrealized

# ─────────────────────────────────────────────
# CYCLE PRINCIPAL
# ─────────────────────────────────────────────

def cycle():
    """Cycle d'analyse et trading — exécuté toutes les 15 minutes."""
    global pnl, sl_streak, trade_count
    now = time.time()

    dashboard()

    # Filtre heures
    if not trading_hours_ok():
        print('  🌙 Hors plage horaire — en attente...')
        return

    # Circuit breaker (inclut PnL non-réalisé)
    bal = balance()
    total = total_pnl()
    circuit_tripped = total < 0 and abs(total) / max(bal, 1) > LIMIT
    if circuit_tripped:
        print(f'  🔴 CIRCUIT BREAKER — PnL total {total:+.2f}$ ({abs(total)/max(bal,1)*100:.1f}% > {LIMIT*100:.0f}%)')
        print(f'     → blocage des nouvelles entrées (positions ouvertes toujours gérées)')

    for pair in PAIRS:
        try:
            df = ohlcv(pair)
            if df is None:
                continue

            sc  = score(df)
            p   = price(pair)
            if p is None:
                continue

            atr   = df['atr'].iloc[-1]
            sl    = atr * 1.5
            tp    = atr * 2.5
            trend = trend_1h(pair)

            print(f'\n  {pair} | prix:{p:.0f} | score:{sc:+d} | tendance 1h:{trend} | atr:{atr:.2f}')

            cooldown_ok = pair not in last_sl or now - last_sl[pair] > COOLDOWN
            risk        = current_risk()

            # ── Entrée ACHAT (filtre: tendance 1h pas bear, circuit breaker non déclenché)
            if (sc >= 50 and trend != 'bear'
                    and not circuit_tripped
                    and pair not in positions
                    and len(positions) < 3
                    and cooldown_ok):
                sz = position_size(bal, p, sl, risk)
                if sz <= 0:
                    print(f'  ⚠️  Sizing invalide pour {pair}, skip')
                    continue
                t  = order(pair, 'buy', sz)
                if t:
                    positions[pair] = {
                        'entry': p, 'size': sz,
                        'sl': sl, 'tp': tp,
                        'dir': 'buy', 'peak': p
                    }
                    trade_count += 1
                    notional = sz * p
                    print(f'  ✅ BUY {pair} | score:{sc} | entrée:{p:.0f} | size:{sz:.6f} | notional:${notional:.0f} | risque:${bal*risk:.0f} ({risk*100:.2f}%)')

            # ── Entrée VENTE (filtre: tendance 1h pas bull, circuit breaker non déclenché)
            elif (sc <= -50 and trend != 'bull'
                    and not circuit_tripped
                    and pair not in positions
                    and len(positions) < 3
                    and cooldown_ok):
                sz = position_size(bal, p, sl, risk)
                if sz <= 0:
                    print(f'  ⚠️  Sizing invalide pour {pair}, skip')
                    continue
                t  = order(pair, 'sell', sz)
                if t:
                    positions[pair] = {
                        'entry': p, 'size': sz,
                        'sl': sl, 'tp': tp,
                        'dir': 'sell', 'peak': p
                    }
                    trade_count += 1
                    notional = sz * p
                    print(f'  ✅ SELL {pair} | score:{sc} | entrée:{p:.0f} | size:{sz:.6f} | notional:${notional:.0f} | risque:${bal*risk:.0f} ({risk*100:.2f}%)')

            # ── Gestion des positions ouvertes
            elif pair in positions:
                pos = positions[pair]
                chg = (p - pos['entry']) / pos['entry'] * (1 if pos['dir'] == 'buy' else -1)

                # Mise à jour du peak pour trailing stop
                if pos['dir'] == 'buy' and p > pos['peak']:
                    positions[pair]['peak'] = p
                elif pos['dir'] == 'sell' and p < pos['peak']:
                    positions[pair]['peak'] = p

                # Trailing Stop (NOUVEAU V4)
                trail_triggered = False
                if pos['dir'] == 'buy':
                    trail_price = pos['peak'] * (1 - TRAIL_PCT)
                    trail_triggered = p <= trail_price and chg > 0
                else:
                    trail_price = pos['peak'] * (1 + TRAIL_PCT)
                    trail_triggered = p >= trail_price and chg > 0

                # Take Profit
                if chg >= pos['tp'] / pos['entry']:
                    cl = 'sell' if pos['dir'] == 'buy' else 'buy'
                    order(pair, cl, pos['size'])
                    profit = (p - pos['entry']) * pos['size'] * (1 if pos['dir'] == 'buy' else -1)
                    pnl += profit
                    sl_streak = 0
                    log_trade(pair, pos['dir'], pos['entry'], p, pos['size'], profit, 'TP')
                    msg = f'🎯 TP {pair} | +{profit:.2f}$ | PnL total: {pnl:+.2f}$'
                    print(f'  {msg}')
                    del positions[pair]

                # Trailing Stop déclenché
                elif trail_triggered:
                    cl = 'sell' if pos['dir'] == 'buy' else 'buy'
                    order(pair, cl, pos['size'])
                    profit = (p - pos['entry']) * pos['size'] * (1 if pos['dir'] == 'buy' else -1)
                    pnl += profit
                    sl_streak = 0
                    log_trade(pair, pos['dir'], pos['entry'], p, pos['size'], profit, 'TRAIL')
                    msg = f'📈 TRAIL STOP {pair} | +{profit:.2f}$ | PnL total: {pnl:+.2f}$'
                    print(f'  {msg}')
                    del positions[pair]

                # Stop Loss classique
                elif chg <= -pos['sl'] / pos['entry']:
                    cl = 'sell' if pos['dir'] == 'buy' else 'buy'
                    order(pair, cl, pos['size'])
                    profit = (p - pos['entry']) * pos['size'] * (1 if pos['dir'] == 'buy' else -1)
                    pnl += profit
                    sl_streak += 1
                    last_sl[pair] = time.time()
                    log_trade(pair, pos['dir'], pos['entry'], p, pos['size'], profit, 'SL')
                    msg = f'🔴 SL {pair} | {profit:.2f}$ | streak:{sl_streak} | PnL total: {pnl:+.2f}$'
                    print(f'  {msg}')
                    del positions[pair]

        except Exception as e:
            print(f'  ❌ Erreur {pair}: {e}')

# ─────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    if not _LIVE_DEPS_OK:
        print('❌ Modules krakenex et/ou schedule manquants.')
        print('   Installation: pip install krakenex schedule')
        raise SystemExit(1)
    print('🤖 Bot V4 Kraken démarré')
    print(f'📊 Paires        : {", ".join(PAIRS)}')
    print(f'⚙️  Risque/trade  : {RISK_PER_TRADE*100:.2f}% du capital (réduit si SL consécutifs)')
    print(f'📦 Position max  : {MAX_POSITION_PCT*100:.0f}% du capital (cap notional)')
    print(f'📉 Trailing Stop : {TRAIL_PCT*100}%')
    print(f'🕐 Trading       : {TRADE_H_MIN}h-{TRADE_H_MAX}h UTC')
    print(f'🛡️  Circuit bkr   : -{LIMIT*100:.0f}% (réalisé + non-réalisé)')
    print(f'📁 Log fichier   : {LOG_FILE}\n')

    schedule.every(15).minutes.do(cycle)
    cycle()
    print('\nBot actif... (Ctrl+C pour arrêter)')
    while True:
        schedule.run_pending()
        time.sleep(30)
