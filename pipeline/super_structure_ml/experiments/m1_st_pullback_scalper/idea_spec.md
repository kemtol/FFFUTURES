# M1 SuperTrend Pullback Scalper Idea Spec

## Research Question

Can a standalone M1 SuperTrend pullback engine produce frequent scalps with
positive expectancy after commission and conservative exit assumptions?

This is not a 5m signal refinement layer. M1 generates its own candidates.

## Candidate Definition

Long candidate:

- SuperTrend direction remains bullish.
- Close is above SuperTrend.
- Low touches the SuperTrend pullback band.
- Candle closes bullish.
- Optional trend filter: close above M1 DEMA100.
- Skip if RSI/CCI already overbought.

Short candidate:

- SuperTrend direction remains bearish.
- Close is below SuperTrend.
- High touches the SuperTrend pullback band.
- Candle closes bearish.
- Optional trend filter: close below M1 DEMA100.
- Skip if RSI/CCI already oversold.

## Why Skip Overbought/Oversold

The model should not buy after momentum is already exhausted or short after
panic selling is already stretched. The desired setup is continuation after a
pullback, not chase-after-extension.

Initial skip rules:

- Skip long if `rsi_7 >= 72` or `cci >= 180`.
- Skip short if `rsi_7 <= 28` or `cci <= -180`.

These are research defaults, not live constants.

## Conditional Exit Idea

The signal is evaluated at M1 close. Entry execution is at the next M1 open
after the signal candle closes.

Initial SL:

- Long: below SuperTrend by buffer.
- Short: above SuperTrend by buffer.
- Risk must be within configured min/max bounds.

Exit is the first event among:

- SL touch.
- R-multiple target touch.
- Momentum exhaustion while position is profitable.
- SuperTrend flip.
- Timeout.

The first implementation assumes SL first if SL and TP are both touched in the
same M1 bar. This is intentionally conservative.

## Initial Feature Families

Trend/pullback:

- `st_gap_atr`
- `touch_distance_atr`
- `pullback_band_atr`
- `dema_50/100/200` distances
- `dema_stack`
- `st_slope_5_atr`

Momentum:

- `signal_adx`
- `signal_cci`
- `cci_abs`
- `rsi_7`
- `close_slope_3_atr`
- `close_slope_5_atr`

Candle quality:

- `wick_ratio`
- `candle_body_atr`
- `bar_range_atr`
- `directional_close_pos`

VWAP context:

- `dist_to_ct_vwap_atr`
- `ct_vwap_slope_20_atr`
- `vwap_deviation_z_50`

Session/time:

- `hour_utc`
- `dow`
- `session_cluster`

Risk/outcome:

- `risk_pts`
- `hold_bars`
- `exit_reason`
- `pnl_usd`
- `label`

## Success Criteria

Before model training:

- Enough events for 2023-2026 research.
- Raw mechanical expectancy is not catastrophically negative.
- Trade frequency can plausibly reach 1-3 trades/day after ML filtering.

Before live consideration:

- OOT positive expectancy after commission.
- Max drawdown compatible with Topstep 50K constraints.
- Clear daily trade cap and daily loss cap.
- No hidden dependency on 5m CONS signals.
