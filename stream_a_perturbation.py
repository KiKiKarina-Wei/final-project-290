

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as timemod
import os
import warnings
from scipy.optimize import linprog

warnings.filterwarnings('ignore')


P_MAX   = 10.0                    # MW power rating
E_TOTAL = 40.0                    # MWh nameplate capacity
SOC_MIN_PCT = 0.10                # SOC floor (10%)
SOC_MAX_PCT = 0.90                # SOC ceiling (90%)
E_MIN   = E_TOTAL * SOC_MIN_PCT  # 4 MWh
E_MAX   = E_TOTAL * SOC_MAX_PCT  # 36 MWh
ETA_RT  = 0.85                    # round-trip efficiency
ETA_C   = np.sqrt(ETA_RT)        # charge-side efficiency  (sqrt(0.85) = 0.9220)
ETA_D   = np.sqrt(ETA_RT)        # discharge-side efficiency

MASTER_PATH = "master_hourly.csv"
PF_PATH     = "pf_daily_revenue.csv"
OUTPUT_DIR  = "stream_a_output"

SEED = 42
_N_VARS = 72
_T = 24
_A_eq = np.zeros((_T, _N_VARS))
_b_eq = np.zeros(_T)

for _t in range(_T):
    _A_eq[_t, _t]       = -ETA_C        # c[t]
    _A_eq[_t, 24 + _t]  =  1.0 / ETA_D  # d[t]
    _A_eq[_t, 48 + _t]  =  1.0          # soc[t+1]
    if _t > 0:
        _A_eq[_t, 48 + _t - 1] = -1.0   # -soc[t]
    else:
        _b_eq[_t] = E_MIN               # soc[0] = E_MIN


_A_ub = np.zeros((1, _N_VARS))
_A_ub[0, 0:24] = ETA_C
_b_ub = np.array([E_MAX - E_MIN])       # 32 MWh usable

_bounds_base = (
    [(0, P_MAX)] * 24 +                  # c[t]
    [(0, P_MAX)] * 24 +                  # d[t]
    [(E_MIN, E_MAX)] * 23 +              # soc[1..23]
    [(E_MIN, E_MIN)]                     # soc[24] = E_MIN (terminal)
)


def solve_dispatch_fast(forecast_prices, real_prices):
    fp = np.asarray(forecast_prices, dtype=np.float64)
    rp = np.asarray(real_prices, dtype=np.float64)

    # Objective: minimise  sum(c*p) - sum(d*p)  [= negative revenue]
    obj = np.zeros(_N_VARS)
    obj[0:24]  =  fp
    obj[24:48] = -fp

    res = linprog(obj, A_ub=_A_ub, b_ub=_b_ub,
                  A_eq=_A_eq, b_eq=_b_eq,
                  bounds=_bounds_base, method='highs',
                  options={'presolve': True, 'disp': False})

    if not res.success:
        return {
            'status': 'Infeasible',
            'actual_revenue': 0.0, 'forecast_revenue': 0.0,
            'charge': np.zeros(24), 'discharge': np.zeros(24),
        }

    c_vals = res.x[0:24]
    d_vals = res.x[24:48]

    actual_rev   = float(np.sum((d_vals - c_vals) * rp) * 1000)
    forecast_rev = float(np.sum((d_vals - c_vals) * fp) * 1000)

    return {
        'status': 'Optimal',
        'actual_revenue': actual_rev,
        'forecast_revenue': forecast_rev,
        'charge': c_vals,
        'discharge': d_vals,
    }



def perturb_gaussian_noise(real_prices, noise_level, rng):
    sigma = noise_level * np.mean(np.abs(real_prices))
    noise = rng.normal(0, sigma, size=24)
    return np.maximum(real_prices + noise, 0.01)


def perturb_peak_shift(real_prices, shift_hours, n_peaks=3):
    p = real_prices.copy()
    top_idx = np.argsort(p)[-n_peaks:]
    peak_vals = p[top_idx].copy()
    top_set = set(top_idx)

    # Fill original peak positions with neighbour average
    for idx in sorted(top_idx):
        nbrs = []
        if idx > 0  and (idx - 1) not in top_set:
            nbrs.append(real_prices[idx - 1])
        if idx < 23 and (idx + 1) not in top_set:
            nbrs.append(real_prices[idx + 1])
        p[idx] = np.mean(nbrs) if nbrs else p[idx]

    # Place peaks at shifted positions
    new_pos = np.clip(top_idx + shift_hours, 0, 23)
    for val, npos in zip(peak_vals, new_pos):
        p[int(npos)] = val

    return np.maximum(p, 0.01)


def perturb_spike_underestimate(real_prices, scale_down, p90_threshold):
    p = real_prices.copy()
    mask = p > p90_threshold
    p[mask] *= (1.0 - scale_down)
    return np.maximum(p, 0.01)


def build_scenarios():
    scenarios = []

    for level in [0.05, 0.10, 0.20, 0.40]:
        scenarios.append({
            'name': f'noise_{int(level*100)}pct',
            'type': 'noise',
            'param': level,
            'param_label': f'{int(level*100)}%',
        })

    for shift in [-3, -2, -1, 1, 2, 3]:
        sign = '+' if shift > 0 else ''
        scenarios.append({
            'name': f'shift_{sign}{shift}h',
            'type': 'peak_shift',
            'param': shift,
            'param_label': f'{sign}{shift}h',
        })

    for scale in [0.20, 0.40, 0.60]:
        scenarios.append({
            'name': f'spike_under_{int(scale*100)}pct',
            'type': 'spike_underestimate',
            'param': scale,
            'param_label': f'{int(scale*100)}%',
        })

    return scenarios

def main():
    print("=" * 70)
    print("  Stream A: Perturbation Experiment Pipeline")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] Loading data ...")
    df = pd.read_csv(MASTER_PATH)
    pf = pd.read_csv(PF_PATH)

    df['date'] = df['timestamp_jst'].str[:10]
    dates = df['date'].unique()
    print(f"  {len(df):,} hourly rows,  {len(dates)} days")

    # Build lookup: date -> 24-hour price vector
    day_prices = {}
    for date in dates:
        p = df.loc[df['date'] == date, 'price_tokyo'].values
        if len(p) == 24:
            day_prices[date] = p
    print(f"  {len(day_prices)} complete days ready")

    # PF revenue lookup
    pf_revenue = dict(zip(pf['date'], pf['revenue_jpy']))

    # Global P90 price threshold
    p90 = float(df['price_tokyo'].quantile(0.90))
    print(f"  P90 price threshold: {p90:.2f} JPY/kWh")

    test_day = list(day_prices.keys())[0]
    res_test = solve_dispatch_fast(day_prices[test_day], day_prices[test_day])
    pf_test  = pf_revenue[test_day]
    print(f"  Sanity check ({test_day}): LP={res_test['actual_revenue']:.2f}, "
          f"PF={pf_test:.2f}, match={abs(res_test['actual_revenue']-pf_test)<1}")

    scenarios = build_scenarios()
    n_sc = len(scenarios)
    n_days = len(day_prices)
    total = n_sc * n_days
    print(f"\n[2/5] {n_sc} scenarios defined:")
    for s in scenarios:
        print(f"       {s['name']:<30s}  type={s['type']:<25s}  param={s['param']}")
    print(f"\n  Total LP solves: {n_sc} x {n_days} = {total:,}")

    # ---- Monte Carlo loop ----
    print(f"\n[3/5] Running experiments ...")
    rng = np.random.default_rng(SEED)
    results = []
    t0 = timemod.time()
    count = 0

    sorted_dates = sorted(day_prices.keys())

    for s in scenarios:
        s_type  = s['type']
        s_param = s['param']
        s_name  = s['name']

        for date in sorted_dates:
            real_p = day_prices[date]

            # Generate perturbed forecast
            if s_type == 'noise':
                forecast_p = perturb_gaussian_noise(real_p, s_param, rng)
            elif s_type == 'peak_shift':
                forecast_p = perturb_peak_shift(real_p, s_param)
            elif s_type == 'spike_underestimate':
                forecast_p = perturb_spike_underestimate(real_p, s_param, p90)
            else:
                raise ValueError(f"Unknown type: {s_type}")

            # Optimise with forecast, evaluate with real
            res = solve_dispatch_fast(forecast_p, real_p)

            pf_rev = pf_revenue.get(date, 0.0)
            act_rev = res['actual_revenue']
            loss = pf_rev - act_rev
            loss_pct = (loss / pf_rev * 100) if pf_rev > 0 else 0.0

            results.append({
                'date':              date,
                'scenario':          s_name,
                'scenario_type':     s_type,
                'scenario_param':    s_param,
                'pf_revenue':        pf_rev,
                'actual_revenue':    act_rev,
                'revenue_loss':      loss,
                'revenue_loss_pct':  loss_pct,
                'status':            res['status'],
            })

            count += 1
            if count % 3000 == 0:
                elapsed = timemod.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate
                print(f"  {count:>6,}/{total:,} "
                      f"({count/total*100:5.1f}%)  "
                      f"{elapsed:5.0f}s elapsed,  ETA {eta:4.0f}s")

    elapsed = timemod.time() - t0
    print(f"  Done!  {count:,} solves in {elapsed:.1f}s  "
          f"({count/elapsed:.0f} solves/s)")
    print(f"\n[4/5] Saving results ...")
    res_df = pd.DataFrame(results)

    res_path = os.path.join(OUTPUT_DIR, "stream_a_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"  {res_path}  ({len(res_df):,} rows)")

    # Summary per scenario
    summary = res_df.groupby(
        ['scenario', 'scenario_type', 'scenario_param']
    ).agg(
        mean_pf_revenue       = ('pf_revenue',       'mean'),
        mean_actual_revenue   = ('actual_revenue',   'mean'),
        mean_revenue_loss     = ('revenue_loss',     'mean'),
        mean_revenue_loss_pct = ('revenue_loss_pct', 'mean'),
        median_revenue_loss_pct = ('revenue_loss_pct', 'median'),
        std_revenue_loss_pct  = ('revenue_loss_pct', 'std'),
        p95_revenue_loss_pct  = ('revenue_loss_pct', lambda x: np.percentile(x, 95)),
        max_revenue_loss_pct  = ('revenue_loss_pct', 'max'),
        total_actual_revenue  = ('actual_revenue',   'sum'),
        total_pf_revenue      = ('pf_revenue',       'sum'),
        n_days                = ('date',             'count'),
        n_infeasible          = ('status',
                                 lambda x: (x != 'Optimal').sum()),
    ).reset_index()

    summary['RER'] = summary['total_actual_revenue'] / summary['total_pf_revenue']
    summary['total_revenue_loss_pct'] = (1 - summary['RER']) * 100

    sum_path = os.path.join(OUTPUT_DIR, "stream_a_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"  {sum_path}  ({len(summary)} rows)")


    print("\n" + "=" * 100)
    print(f"{'Scenario':<30s} {'Type':<22s} {'Mean Loss%':>10s} "
          f"{'Med Loss%':>10s} {'P95 Loss%':>10s} {'RER':>8s} {'TotalLoss%':>10s}")
    print("-" * 100)
    for _, r in summary.sort_values(
        ['scenario_type', 'scenario_param']
    ).iterrows():
        print(f"{r['scenario']:<30s} {r['scenario_type']:<22s} "
              f"{r['mean_revenue_loss_pct']:>10.2f} "
              f"{r['median_revenue_loss_pct']:>10.2f} "
              f"{r['p95_revenue_loss_pct']:>10.2f} "
              f"{r['RER']:>8.4f} "
              f"{r['total_revenue_loss_pct']:>10.2f}")
    print("=" * 100)


    print(f"\n[5/5] Generating plots ...")
    plot_all(res_df, summary)

    print(f"\n{'=' * 70}")
    print(f"  All outputs saved to:  {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


def plot_all(res_df, summary):
    plt.rcParams.update({
        'figure.dpi': 150, 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 12,
    })
    _plot_noise(res_df, summary)
    _plot_shift(res_df, summary)
    _plot_spike(res_df, summary)
    _plot_heatmap(summary)


def _plot_noise(res_df, summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = res_df[res_df.scenario_type == 'noise']
    levels = [0.05, 0.10, 0.20, 0.40]
    labels = ['5%', '10%', '20%', '40%']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    ax = axes[0]
    bdata = [data[data.scenario_param == p]['revenue_loss_pct'].values for p in levels]
    bp = ax.boxplot(bdata, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xlabel('Noise Level (% of daily mean price)')
    ax.set_ylabel('Revenue Loss (%)')
    ax.set_title('(a) Revenue Loss Distribution by Noise Level')
    ax.axhline(0, color='gray', ls='--', alpha=.5)
    ax.grid(axis='y', alpha=.3)

    ax = axes[1]
    ns = summary[summary.scenario_type == 'noise'].sort_values('scenario_param')
    x = ns.scenario_param * 100
    ax.plot(x, ns.mean_revenue_loss_pct,  'o-',  color='#2c3e50', lw=2, ms=8,
            label='Mean loss %')
    ax.plot(x, ns.p95_revenue_loss_pct,   's--', color='#e74c3c', lw=2, ms=8,
            label='P95 loss % (downside)')
    ax.fill_between(x, ns.mean_revenue_loss_pct.values,
                    ns.p95_revenue_loss_pct.values, alpha=.15, color='#e74c3c')
    ax.set_xlabel('Noise Level (%)'); ax.set_ylabel('Revenue Loss (%)')
    ax.set_title('(b) Mean & Downside Risk vs Noise')
    ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_revenue_loss_vs_noise.png")
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def _plot_shift(res_df, summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = res_df[res_df.scenario_type == 'peak_shift']
    shifts = [-3, -2, -1, 1, 2, 3]
    labels = [f'{s:+d}h' for s in shifts]
    cols = ['#1abc9c','#2ecc71','#82e0aa','#f9e79f','#f39c12','#e67e22']

    ax = axes[0]
    bdata = [data[data.scenario_param == s]['revenue_loss_pct'].values for s in shifts]
    bp = ax.boxplot(bdata, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xlabel('Peak Shift (hours)')
    ax.set_ylabel('Revenue Loss (%)')
    ax.set_title('(a) Revenue Loss by Peak Timing Error')
    ax.axhline(0, color='gray', ls='--', alpha=.5); ax.grid(axis='y', alpha=.3)

    ax = axes[1]
    ss = summary[summary.scenario_type == 'peak_shift'].copy()
    ss['abs_shift'] = ss.scenario_param.abs()
    sym = ss.groupby('abs_shift').agg(
        mean_loss=('mean_revenue_loss_pct', 'mean'),
        p95_loss =('p95_revenue_loss_pct',  'mean'),
    ).reset_index()
    ax.plot(sym.abs_shift, sym.mean_loss, 'o-',  color='#2c3e50', lw=2, ms=8,
            label='Mean loss %')
    ax.plot(sym.abs_shift, sym.p95_loss,  's--', color='#e74c3c', lw=2, ms=8,
            label='P95 loss %')
    ax.fill_between(sym.abs_shift, sym.mean_loss.values, sym.p95_loss.values,
                    alpha=.15, color='#e74c3c')
    ax.set_xlabel('|Peak Shift| (hours)'); ax.set_ylabel('Revenue Loss (%)')
    ax.set_title('(b) Mean & Downside Risk vs |Shift|')
    ax.set_xticks([1,2,3]); ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_revenue_loss_vs_shift.png")
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def _plot_spike(res_df, summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = res_df[res_df.scenario_type == 'spike_underestimate']
    scales = [0.20, 0.40, 0.60]
    labels = ['20%', '40%', '60%']
    cols = ['#f39c12', '#e67e22', '#e74c3c']

    ax = axes[0]
    bdata = [data[data.scenario_param == s]['revenue_loss_pct'].values for s in scales]
    bp = ax.boxplot(bdata, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_xlabel('Spike Underestimation (%)')
    ax.set_ylabel('Revenue Loss (%)')
    ax.set_title('(a) Revenue Loss by Spike Underestimation')
    ax.axhline(0, color='gray', ls='--', alpha=.5); ax.grid(axis='y', alpha=.3)

    ax = axes[1]
    sk = summary[summary.scenario_type == 'spike_underestimate'].sort_values('scenario_param')
    x = sk.scenario_param * 100
    ax.plot(x, sk.mean_revenue_loss_pct, 'o-',  color='#2c3e50', lw=2, ms=8,
            label='Mean loss %')
    ax.plot(x, sk.p95_revenue_loss_pct,  's--', color='#e74c3c', lw=2, ms=8,
            label='P95 loss %')
    ax.fill_between(x, sk.mean_revenue_loss_pct.values,
                    sk.p95_revenue_loss_pct.values, alpha=.15, color='#e74c3c')
    ax.set_xlabel('Spike Underestimation (%)'); ax.set_ylabel('Revenue Loss (%)')
    ax.set_title('(b) Mean & Downside Risk vs Spike Underestimation')
    ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_revenue_loss_vs_spike.png")
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


def _plot_heatmap(summary):
    fig, ax = plt.subplots(figsize=(11, 7))

    plot_df = summary[['scenario', 'scenario_type',
                       'mean_revenue_loss_pct',
                       'p95_revenue_loss_pct', 'RER']].copy()
    type_order = {'noise': 0, 'peak_shift': 1, 'spike_underestimate': 2}
    plot_df['sort_key'] = (plot_df.scenario_type.map(type_order) * 1000
                           + plot_df.mean_revenue_loss_pct)
    plot_df = plot_df.sort_values('sort_key')

    y = np.arange(len(plot_df))
    ax.barh(y, plot_df.mean_revenue_loss_pct.values, height=0.6,
            color='#3498db', alpha=0.75, label='Mean Loss %')
    ax.barh(y, plot_df.p95_revenue_loss_pct.values, height=0.6,
            color='#e74c3c', alpha=0.30, label='P95 Loss %')

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.scenario.values, fontsize=9)
    ax.set_xlabel('Revenue Loss (%)')
    ax.set_title('All Scenarios: Mean vs Downside Revenue Loss')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=.3)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row.p95_revenue_loss_pct + 0.2, i,
                f"RER={row.RER:.4f}", va='center', fontsize=8, color='#555')

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_heatmap_all_scenarios.png")
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
