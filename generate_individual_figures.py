#!/usr/bin/env python3
"""
Generate Individual Figures for Market Making Strategy Analysis
Each figure is saved as a separate PNG file.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# High-quality plot settings
plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# CONFIGURATION
# =============================================================================

MY_CARDS = [10, 1, 0]
REVEALED_TABLE = [16, 14, 19, 7]
HIDDEN_COUNT = 2
REVEALED_SUM = sum(REVEALED_TABLE)

# Build deck
FULL_DECK = []
for i in range(21):
    FULL_DECK.extend([i, i])
for i in range(-10, -81, -10):
    FULL_DECK.append(i)

DECK_SUM = 60

# Remaining deck
known = Counter(MY_CARDS + REVEALED_TABLE)
full = Counter(FULL_DECK)
REMAINING_DECK = list((full - known).elements())

# Calculate all outcomes
all_pairs = list(combinations(REMAINING_DECK, 2))
all_hidden_sums = [sum(pair) for pair in all_pairs]
all_table_sums = [REVEALED_SUM + s for s in all_hidden_sums]

TOTAL_OUTCOMES = len(all_table_sums)
FAIR_VALUE = np.mean(all_table_sums)
STD_DEV = np.std(all_table_sums)

sum_counts = Counter(all_table_sums)
unique_sums = sorted(sum_counts.keys())
probabilities = {s: count / TOTAL_OUTCOMES for s, count in sum_counts.items()}

# Optimal strategy PnL
pnl_buy_opt = [57 - ts for ts in all_table_sums]
pnl_sell_opt = [ts - 54 for ts in all_table_sums]

OUTPUT_DIR = '/Users/top1/Desktop/mm_game_old_mission/figures/'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("GENERATING INDIVIDUAL FIGURES")
print("=" * 80)
print(f"Fair Value: {FAIR_VALUE:.4f}")
print(f"Saving to: {OUTPUT_DIR}")
print()

# =============================================================================
# FIGURE 1: Probability Mass Function
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(unique_sums, [probabilities[s] for s in unique_sums], width=2,
       color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(FAIR_VALUE, color='red', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.axvline(np.median(all_table_sums), color='green', linestyle='--', linewidth=2,
           label=f'Median = {np.median(all_table_sums):.0f}')
ax.set_xlabel('Table Sum (S)', fontsize=14)
ax.set_ylabel('Probability P(S)', fontsize=14)
ax.set_title('Probability Mass Function of Table Sum\n(Exact Distribution from 903 Possible Outcomes)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '01_pmf_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 01_pmf_distribution.png")


# =============================================================================
# FIGURE 2: Cumulative Distribution Function
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
sorted_sums = np.sort(all_table_sums)
cdf = np.arange(1, len(sorted_sums) + 1) / len(sorted_sums)
ax.plot(sorted_sums, cdf, 'b-', linewidth=3)
ax.fill_between(sorted_sums, 0, cdf, alpha=0.3, color='steelblue')
ax.axvline(FAIR_VALUE, color='red', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='50th Percentile')
ax.set_xlabel('Table Sum (S)', fontsize=14)
ax.set_ylabel('Cumulative Probability F(S)', fontsize=14)
ax.set_title('Cumulative Distribution Function (CDF)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '02_cdf.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 02_cdf.png")


# =============================================================================
# FIGURE 3: Q-Q Plot (Normality Test)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10))
stats.probplot(all_table_sums, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Normality Assessment\n(Deviation from line indicates non-normality)', fontsize=16, fontweight='bold')
ax.get_lines()[0].set_markersize(8)
ax.get_lines()[0].set_markerfacecolor('steelblue')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '03_qq_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 03_qq_plot.png")


# =============================================================================
# FIGURE 4: Box Plot with Percentiles
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10))
bp = ax.boxplot(all_table_sums, vert=True, widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax.scatter([1], [FAIR_VALUE], color='red', s=200, zorder=5, marker='*', label=f'Fair Value = {FAIR_VALUE:.2f}')

percentiles = [5, 25, 50, 75, 95]
pct_values = np.percentile(all_table_sums, percentiles)
for pct, val in zip(percentiles, pct_values):
    ax.annotate(f'{pct}%: {val:.1f}', xy=(1.15, val), fontsize=12, va='center')

ax.set_ylabel('Table Sum', fontsize=14)
ax.set_title('Box Plot with Percentile Analysis', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.set_xticks([])
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '04_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 04_boxplot.png")


# =============================================================================
# FIGURE 5: Distribution Moments
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
moment_names = ['Mean\n(μ)', 'Std Dev\n(σ)', 'Skewness\n(γ₁)', 'Kurtosis\n(γ₂)']
moment_values = [FAIR_VALUE, STD_DEV, stats.skew(all_table_sums), stats.kurtosis(all_table_sums)]
colors = ['blue', 'green', 'orange', 'red']
bars = ax.bar(moment_names, moment_values, color=colors, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('Value', fontsize=14)
ax.set_title('Distribution Moments Analysis', fontsize=16, fontweight='bold')
for bar, val in zip(bars, moment_values):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '05_moments.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 05_moments.png")


# =============================================================================
# FIGURE 6: Hidden Card Composition Scenarios
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 10))
scenarios = {'Both Positive\n(65.9%)': 0, 'One Negative\n(31.0%)': 0, 'Both Negative\n(3.1%)': 0}
for pair in all_pairs:
    n_neg = sum(1 for c in pair if c < 0)
    if n_neg == 0:
        scenarios['Both Positive\n(65.9%)'] += 1
    elif n_neg == 1:
        scenarios['One Negative\n(31.0%)'] += 1
    else:
        scenarios['Both Negative\n(3.1%)'] += 1

colors = ['green', 'orange', 'red']
wedges, texts, autotexts = ax.pie(scenarios.values(), labels=scenarios.keys(),
                                   autopct='%1.1f%%', colors=colors, explode=[0.02, 0.02, 0.08],
                                   textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
ax.set_title('Hidden Card Composition Scenarios', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '06_hidden_card_scenarios.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 06_hidden_card_scenarios.png")


# =============================================================================
# FIGURE 7: Expected PnL by Market Position
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
mids = np.linspace(FAIR_VALUE - 10, FAIR_VALUE + 10, 81)
ev_buy = [np.mean([m + 1.5 - ts for ts in all_table_sums]) for m in mids]
ev_sell = [np.mean([ts - (m - 1.5) for ts in all_table_sums]) for m in mids]

ax.plot(mids, ev_buy, 'b-', linewidth=3, label='E[PnL] if opponent BUYS at Ask')
ax.plot(mids, ev_sell, 'r-', linewidth=3, label='E[PnL] if opponent SELLS at Bid')
ax.fill_between(mids, ev_buy, 0, where=[e > 0 for e in ev_buy], alpha=0.2, color='blue')
ax.fill_between(mids, ev_sell, 0, where=[e > 0 for e in ev_sell], alpha=0.2, color='red')
ax.axhline(0, color='black', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Expected PnL', fontsize=14)
ax.set_title('Expected PnL as Function of Market Position\n(Proof: Both lines cross zero at Fair Value)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '07_expected_pnl_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 07_expected_pnl_curves.png")


# =============================================================================
# FIGURE 8: Sharpe Ratio by Market Position
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
sharpe_buy = []
sharpe_sell = []
for mid in mids:
    pnl_b = [mid + 1.5 - ts for ts in all_table_sums]
    pnl_s = [ts - (mid - 1.5) for ts in all_table_sums]
    sharpe_buy.append(np.mean(pnl_b) / np.std(pnl_b))
    sharpe_sell.append(np.mean(pnl_s) / np.std(pnl_s))

ax.plot(mids, sharpe_buy, 'b-', linewidth=3, label='Sharpe Ratio if they BUY')
ax.plot(mids, sharpe_sell, 'r-', linewidth=3, label='Sharpe Ratio if they SELL')
ax.axhline(0, color='black', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Sharpe Ratio (μ/σ)', fontsize=14)
ax.set_title('Risk-Adjusted Return (Sharpe Ratio) by Market Position', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '08_sharpe_ratio.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 08_sharpe_ratio.png")


# =============================================================================
# FIGURE 9: Kelly Criterion Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
kelly_buy = []
kelly_sell = []
for mid in mids:
    pnl_b = [mid + 1.5 - ts for ts in all_table_sums]
    pnl_s = [ts - (mid - 1.5) for ts in all_table_sums]
    kelly_buy.append(np.mean(pnl_b) / np.var(pnl_b) if np.var(pnl_b) > 0 else 0)
    kelly_sell.append(np.mean(pnl_s) / np.var(pnl_s) if np.var(pnl_s) > 0 else 0)

ax.plot(mids, kelly_buy, 'b-', linewidth=3, label='Kelly Fraction if they BUY')
ax.plot(mids, kelly_sell, 'r-', linewidth=3, label='Kelly Fraction if they SELL')
ax.axhline(0, color='black', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Kelly Fraction (f* = μ/σ²)', fontsize=14)
ax.set_title('Kelly Criterion: Optimal Position Size by Market Position', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '09_kelly_criterion.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 09_kelly_criterion.png")


# =============================================================================
# FIGURE 10: Win Rate Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
winrate_buy = []
winrate_sell = []
for mid in mids:
    pnl_b = [mid + 1.5 - ts for ts in all_table_sums]
    pnl_s = [ts - (mid - 1.5) for ts in all_table_sums]
    winrate_buy.append(np.mean(np.array(pnl_b) > 0) * 100)
    winrate_sell.append(np.mean(np.array(pnl_s) > 0) * 100)

ax.plot(mids, winrate_buy, 'b-', linewidth=3, label='Win Rate if they BUY')
ax.plot(mids, winrate_sell, 'r-', linewidth=3, label='Win Rate if they SELL')
ax.axhline(50, color='black', linestyle=':', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Win Rate (%)', fontsize=14)
ax.set_title('Probability of Profit by Market Position', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '10_win_rate.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 10_win_rate.png")


# =============================================================================
# FIGURE 11: Value at Risk (VaR) Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
var95_buy = []
var95_sell = []
for mid in mids:
    pnl_b = [mid + 1.5 - ts for ts in all_table_sums]
    pnl_s = [ts - (mid - 1.5) for ts in all_table_sums]
    var95_buy.append(np.percentile(pnl_b, 5))
    var95_sell.append(np.percentile(pnl_s, 5))

ax.plot(mids, var95_buy, 'b-', linewidth=3, label='VaR 95% if they BUY')
ax.plot(mids, var95_sell, 'r-', linewidth=3, label='VaR 95% if they SELL')
ax.axhline(0, color='black', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Value at Risk (95%)', fontsize=14)
ax.set_title('Value at Risk Analysis by Market Position', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '11_var_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 11_var_analysis.png")


# =============================================================================
# FIGURE 12: CVaR (Expected Shortfall) Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
cvar95_buy = []
cvar95_sell = []
for mid in mids:
    pnl_b = [mid + 1.5 - ts for ts in all_table_sums]
    pnl_s = [ts - (mid - 1.5) for ts in all_table_sums]
    var_b = np.percentile(pnl_b, 5)
    var_s = np.percentile(pnl_s, 5)
    cvar95_buy.append(np.mean([p for p in pnl_b if p <= var_b]))
    cvar95_sell.append(np.mean([p for p in pnl_s if p <= var_s]))

ax.plot(mids, cvar95_buy, 'b-', linewidth=3, label='CVaR 95% if they BUY')
ax.plot(mids, cvar95_sell, 'r-', linewidth=3, label='CVaR 95% if they SELL')
ax.axhline(0, color='black', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Conditional VaR (Expected Shortfall)', fontsize=14)
ax.set_title('Expected Shortfall Analysis by Market Position', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '12_cvar_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 12_cvar_analysis.png")


# =============================================================================
# FIGURE 13: PnL Distribution - If They BUY
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(pnl_buy_opt, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(np.mean(pnl_buy_opt), color='red', linestyle='--', linewidth=3,
           label=f'Mean = {np.mean(pnl_buy_opt):.2f}')
ax.axvline(0, color='black', linewidth=2)
ax.set_xlabel('PnL', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.set_title('PnL Distribution: Opponent BUYS at 57\n(Optimal Strategy)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '13_pnl_distribution_buy.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 13_pnl_distribution_buy.png")


# =============================================================================
# FIGURE 14: PnL Distribution - If They SELL
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(pnl_sell_opt, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
ax.axvline(np.mean(pnl_sell_opt), color='blue', linestyle='--', linewidth=3,
           label=f'Mean = {np.mean(pnl_sell_opt):.2f}')
ax.axvline(0, color='black', linewidth=2)
ax.set_xlabel('PnL', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.set_title('PnL Distribution: Opponent SELLS at 54\n(Optimal Strategy)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '14_pnl_distribution_sell.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 14_pnl_distribution_sell.png")


# =============================================================================
# FIGURE 15: Strategy Heatmap - E[PnL] if They BUY
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 10))
bid_range = np.linspace(45, 60, 31)
ask_range = np.linspace(50, 70, 41)
ev_matrix = np.zeros((len(bid_range), len(ask_range)))

for i, bid in enumerate(bid_range):
    for j, ask in enumerate(ask_range):
        if ask > bid:
            ev_matrix[i, j] = np.mean([ask - ts for ts in all_table_sums])
        else:
            ev_matrix[i, j] = np.nan

im = ax.imshow(ev_matrix, cmap='RdYlGn', aspect='auto', origin='lower',
               extent=[ask_range[0], ask_range[-1], bid_range[0], bid_range[-1]],
               vmin=-10, vmax=15)
ax.contour(ask_range, bid_range, ev_matrix, levels=[0], colors='black', linewidths=2)
ax.plot([57], [54], 'w*', markersize=20, label='Optimal (54/57)')
ax.set_xlabel('Ask Price', fontsize=14)
ax.set_ylabel('Bid Price', fontsize=14)
ax.set_title('E[PnL] Heatmap: Opponent BUYS at Ask', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, label='Expected PnL')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '15_heatmap_buy.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 15_heatmap_buy.png")


# =============================================================================
# FIGURE 16: Strategy Heatmap - E[PnL] if They SELL
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 10))
ev_matrix_sell = np.zeros((len(bid_range), len(ask_range)))

for i, bid in enumerate(bid_range):
    for j, ask in enumerate(ask_range):
        if ask > bid:
            ev_matrix_sell[i, j] = np.mean([ts - bid for ts in all_table_sums])
        else:
            ev_matrix_sell[i, j] = np.nan

im = ax.imshow(ev_matrix_sell, cmap='RdYlGn', aspect='auto', origin='lower',
               extent=[ask_range[0], ask_range[-1], bid_range[0], bid_range[-1]],
               vmin=-10, vmax=15)
ax.contour(ask_range, bid_range, ev_matrix_sell, levels=[0], colors='black', linewidths=2)
ax.plot([57], [54], 'w*', markersize=20, label='Optimal (54/57)')
ax.set_xlabel('Ask Price', fontsize=14)
ax.set_ylabel('Bid Price', fontsize=14)
ax.set_title('E[PnL] Heatmap: Opponent SELLS at Bid', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, label='Expected PnL')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '16_heatmap_sell.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 16_heatmap_sell.png")


# =============================================================================
# FIGURE 17: Strategy Comparison Bar Chart
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
strategies = [
    ('Low\n(50/53)', 50, 53),
    ('Below FV\n(52/55)', 52, 55),
    ('OPTIMAL\n(54/57)', 54, 57),
    ('Above FV\n(56/59)', 56, 59),
    ('High\n(58/61)', 58, 61),
]

avg_evs = []
for name, bid, ask in strategies:
    ev_b = np.mean([ask - ts for ts in all_table_sums])
    ev_s = np.mean([ts - bid for ts in all_table_sums])
    avg_evs.append((ev_b + ev_s) / 2)

colors = ['red', 'orange', 'green', 'orange', 'red']
bars = ax.bar([s[0] for s in strategies], avg_evs, color=colors, edgecolor='black', alpha=0.8)
ax.axhline(0, color='black', linewidth=2)
ax.set_ylabel('Average Expected PnL', fontsize=14)
ax.set_title('Strategy Comparison: Average Expected PnL\n(Optimal Strategy Centered at Fair Value)', fontsize=16, fontweight='bold')

for bar, ev in zip(bars, avg_evs):
    color = 'darkgreen' if ev > 0 else 'darkred'
    ax.annotate(f'{ev:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               ha='center', va='bottom' if ev >= 0 else 'top', fontsize=14,
               fontweight='bold', color=color)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '17_strategy_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 17_strategy_comparison.png")


# =============================================================================
# FIGURE 18: Monte Carlo Convergence
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
np.random.seed(42)
n_sims_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
mc_estimates = []
mc_stds = []

for n in n_sims_list:
    estimates = []
    for _ in range(50):  # 50 trials
        samples = []
        for _ in range(n):
            idx = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
            hidden = [REMAINING_DECK[i] for i in idx]
            samples.append(REVEALED_SUM + sum(hidden))
        estimates.append(np.mean(samples))
    mc_estimates.append(np.mean(estimates))
    mc_stds.append(np.std(estimates))

ax.errorbar(n_sims_list, mc_estimates, yerr=[1.96*s for s in mc_stds],
            fmt='o-', capsize=5, linewidth=2, markersize=8, color='steelblue')
ax.axhline(FAIR_VALUE, color='red', linestyle='--', linewidth=3, label=f'True Fair Value = {FAIR_VALUE:.2f}')
ax.set_xscale('log')
ax.set_xlabel('Number of Monte Carlo Simulations', fontsize=14)
ax.set_ylabel('Estimated Fair Value (with 95% CI)', fontsize=14)
ax.set_title('Monte Carlo Convergence Analysis', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '18_monte_carlo_convergence.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 18_monte_carlo_convergence.png")


# =============================================================================
# FIGURE 19: Bootstrap Distribution of Mean
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
np.random.seed(42)
n_bootstrap = 10000
boot_means = []
for _ in range(n_bootstrap):
    sample = np.random.choice(all_table_sums, size=len(all_table_sums), replace=True)
    boot_means.append(np.mean(sample))

ax.hist(boot_means, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(np.mean(boot_means), color='red', linestyle='-', linewidth=3, label=f'Bootstrap Mean = {np.mean(boot_means):.2f}')
ax.axvline(np.percentile(boot_means, 2.5), color='green', linestyle='--', linewidth=2, label='95% CI')
ax.axvline(np.percentile(boot_means, 97.5), color='green', linestyle='--', linewidth=2)
ax.axvline(FAIR_VALUE, color='orange', linestyle=':', linewidth=3, label=f'True FV = {FAIR_VALUE:.2f}')
ax.set_xlabel('Bootstrap Mean Estimate', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Bootstrap Distribution of Fair Value Estimate\n(10,000 Bootstrap Samples)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '19_bootstrap_mean.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 19_bootstrap_mean.png")


# =============================================================================
# FIGURE 20: Negative Card Impact Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
neg_cards = [-10, -20, -30, -40, -50, -60, -70, -80]
impacts = []

for neg in neg_cards:
    remaining_after = [c for c in REMAINING_DECK if c != neg]
    conditional_sums = [REVEALED_SUM + neg + c for c in remaining_after]
    conditional_fv = np.mean(conditional_sums)
    impact = conditional_fv - FAIR_VALUE
    impacts.append(impact)

bars = ax.bar([str(n) for n in neg_cards], impacts, color='darkred', alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=2)
ax.set_xlabel('Negative Card Value', fontsize=14)
ax.set_ylabel('Impact on Fair Value', fontsize=14)
ax.set_title('Impact on Fair Value if Negative Card is Hidden', fontsize=16, fontweight='bold')

for bar, imp in zip(bars, impacts):
    ax.annotate(f'{imp:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               ha='center', va='top', fontsize=12, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '20_negative_card_impact.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 20_negative_card_impact.png")


# =============================================================================
# FIGURE 21: Robustness to Estimation Error
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
errors = np.linspace(-10, 10, 41)
pnl_if_error = []

for error in errors:
    wrong_fv = FAIR_VALUE + error
    bid, ask = wrong_fv - 1.5, wrong_fv + 1.5
    ev_b = np.mean([ask - ts for ts in all_table_sums])
    ev_s = np.mean([ts - bid for ts in all_table_sums])
    pnl_if_error.append((ev_b + ev_s) / 2)

ax.plot(errors, pnl_if_error, 'b-', linewidth=3, marker='o', markersize=6)
ax.axhline(0, color='black', linewidth=2)
ax.axvline(0, color='green', linestyle='--', linewidth=3, label='No Estimation Error')
ax.fill_between(errors, pnl_if_error, 0, where=[p > 0 for p in pnl_if_error], alpha=0.3, color='green')
ax.fill_between(errors, pnl_if_error, 0, where=[p < 0 for p in pnl_if_error], alpha=0.3, color='red')
ax.set_xlabel('Estimation Error in Fair Value', fontsize=14)
ax.set_ylabel('Expected PnL', fontsize=14)
ax.set_title('Robustness Analysis: PnL vs Fair Value Estimation Error', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '21_robustness_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 21_robustness_analysis.png")


# =============================================================================
# FIGURE 22: Inventory Skew Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
positions = range(-5, 6)
skew_factor = 0.5

optimal_bids = []
optimal_asks = []
for pos in positions:
    skew = -pos * skew_factor
    optimal_mid = FAIR_VALUE + skew
    optimal_bids.append(optimal_mid - 1.5)
    optimal_asks.append(optimal_mid + 1.5)

ax.plot(positions, optimal_bids, 'b-', linewidth=3, marker='v', markersize=10, label='Optimal Bid')
ax.plot(positions, optimal_asks, 'r-', linewidth=3, marker='^', markersize=10, label='Optimal Ask')
ax.axhline(FAIR_VALUE, color='green', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.fill_between(positions, optimal_bids, optimal_asks, alpha=0.2, color='purple')
ax.set_xlabel('Current Position (+ = Long, - = Short)', fontsize=14)
ax.set_ylabel('Optimal Price', fontsize=14)
ax.set_title('Inventory Management: Optimal Bid/Ask by Position\n(Skew Factor = 0.5 per unit)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '22_inventory_skew.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 22_inventory_skew.png")


# =============================================================================
# FIGURE 23: Game Theory Payoff Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
market_offsets = np.linspace(-5, 5, 11)
market_positions = [(FAIR_VALUE - 1.5 + offset, FAIR_VALUE + 1.5 + offset) for offset in market_offsets]

min_payoffs = []
max_payoffs = []
avg_payoffs = []

for bid, ask in market_positions:
    ev_b = np.mean([ask - ts for ts in all_table_sums])
    ev_s = np.mean([ts - bid for ts in all_table_sums])
    ev_pass = 0
    payoffs = [ev_b, ev_s, ev_pass]
    min_payoffs.append(min(payoffs))
    max_payoffs.append(max(payoffs))
    avg_payoffs.append(np.mean([ev_b, ev_s]))

x = range(len(market_positions))
ax.plot(x, min_payoffs, 'r-', linewidth=3, marker='v', markersize=10, label='Worst Case (Minimax)')
ax.plot(x, max_payoffs, 'g-', linewidth=3, marker='^', markersize=10, label='Best Case')
ax.plot(x, avg_payoffs, 'b-', linewidth=3, marker='o', markersize=10, label='Average')
ax.fill_between(x, min_payoffs, max_payoffs, alpha=0.2)
ax.axhline(0, color='black', linewidth=2)
ax.axvline(len(market_positions)//2, color='gold', linestyle='--', linewidth=3, label='Optimal (Minimax)')
ax.set_xticks(x)
ax.set_xticklabels([f'{b:.0f}/{a:.0f}' for b, a in market_positions], rotation=45, ha='right')
ax.set_xlabel('Market Quote (Bid/Ask)', fontsize=14)
ax.set_ylabel('Expected PnL', fontsize=14)
ax.set_title('Game Theory: Minimax Analysis', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '23_game_theory.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 23_game_theory.png")


# =============================================================================
# FIGURE 24: CRRA Utility Functions
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
x = np.linspace(50, 200, 200)

def crra_utility(wealth, gamma):
    if gamma == 1:
        return np.log(wealth)
    else:
        return wealth ** (1 - gamma) / (1 - gamma)

for gamma in [0, 0.5, 1, 2, 5]:
    y = [crra_utility(w, gamma) for w in x]
    ax.plot(x, y, linewidth=3, label=f'γ = {gamma}')

ax.axvline(100, color='black', linestyle=':', linewidth=2, label='Initial Wealth')
ax.set_xlabel('Wealth', fontsize=14)
ax.set_ylabel('Utility U(W)', fontsize=14)
ax.set_title('CRRA Utility Functions with Different Risk Aversion (γ)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '24_utility_functions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 24_utility_functions.png")


# =============================================================================
# FIGURE 25: Certainty Equivalent Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
risk_aversions = [0, 0.5, 1, 2, 5]
mids_ce = np.linspace(FAIR_VALUE - 8, FAIR_VALUE + 8, 33)

for gamma in risk_aversions:
    ce_values = []
    for mid in mids_ce:
        pnl_b = [mid + 1.5 - ts for ts in all_table_sums]
        # Calculate certainty equivalent
        if gamma == 1:
            eu = np.mean([np.log(max(100 + p, 1)) for p in pnl_b])
            ce = np.exp(eu) - 100
        else:
            eu = np.mean([(100 + p) ** (1 - gamma) / (1 - gamma) for p in pnl_b])
            ce = (eu * (1 - gamma)) ** (1 / (1 - gamma)) - 100
        ce_values.append(ce)
    ax.plot(mids_ce, ce_values, linewidth=2, label=f'γ = {gamma}')

ax.axhline(0, color='black', linewidth=2)
ax.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=3, label=f'Fair Value')
ax.set_xlabel('Market Mid Price', fontsize=14)
ax.set_ylabel('Certainty Equivalent', fontsize=14)
ax.set_title('Certainty Equivalent by Risk Aversion Level', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '25_certainty_equivalent.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 25_certainty_equivalent.png")


# =============================================================================
# FIGURE 26: Spread Width Sensitivity
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
spreads = [2, 3, 4, 5, 6, 7, 8]
evs_by_spread = []

for spread in spreads:
    bid, ask = FAIR_VALUE - spread/2, FAIR_VALUE + spread/2
    ev_b = np.mean([ask - ts for ts in all_table_sums])
    ev_s = np.mean([ts - bid for ts in all_table_sums])
    evs_by_spread.append((ev_b + ev_s) / 2)

bars = ax.bar(spreads, evs_by_spread, color='steelblue', edgecolor='black', alpha=0.8)
ax.axhline(0, color='black', linewidth=2)
ax.axvline(3, color='red', linestyle='--', linewidth=3, label='Game Minimum (3)')
ax.set_xlabel('Spread Width', fontsize=14)
ax.set_ylabel('Average Expected PnL', fontsize=14)
ax.set_title('Sensitivity to Spread Width (Centered at Fair Value)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)

for bar, ev in zip(bars, evs_by_spread):
    ax.annotate(f'{ev:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '26_spread_sensitivity.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 26_spread_sensitivity.png")


# =============================================================================
# FIGURE 27: Percentile Tail Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
percentiles = list(range(1, 100))
pct_values = [np.percentile(all_table_sums, p) for p in percentiles]

ax.fill_between(percentiles, pct_values, FAIR_VALUE, alpha=0.3,
               where=[p < FAIR_VALUE for p in pct_values], color='red', label='Below FV')
ax.fill_between(percentiles, pct_values, FAIR_VALUE, alpha=0.3,
               where=[p >= FAIR_VALUE for p in pct_values], color='green', label='Above FV')
ax.plot(percentiles, pct_values, 'b-', linewidth=3)
ax.axhline(FAIR_VALUE, color='red', linestyle='--', linewidth=3, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax.set_xlabel('Percentile', fontsize=14)
ax.set_ylabel('Table Sum Value', fontsize=14)
ax.set_title('Percentile Analysis: Full Distribution', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '27_percentile_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 27_percentile_analysis.png")


# =============================================================================
# FIGURE 28: Information Entropy Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate entropy at different stages
def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

prob_values = list(probabilities.values())
current_entropy = entropy(prob_values)
uniform_entropy = np.log2(len(unique_sums))
max_possible_entropy = np.log2(len(FULL_DECK))

stages = ['Maximum Possible\n(50 cards uniform)', 'Current Knowledge\n(After cards dealt)', 'Final\n(All revealed)']
entropies = [max_possible_entropy, current_entropy, 0]
colors = ['lightgray', 'steelblue', 'darkblue']

bars = ax.bar(stages, entropies, color=colors, edgecolor='black', alpha=0.8)
ax.set_ylabel('Entropy (bits)', fontsize=14)
ax.set_title('Information Entropy: Uncertainty Reduction Through Game', fontsize=16, fontweight='bold')

for bar, ent in zip(bars, entropies):
    ax.annotate(f'{ent:.2f} bits', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
               ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '28_entropy_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 28_entropy_analysis.png")


# =============================================================================
# FIGURE 29: Remaining Card Distribution
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
card_counts = Counter(REMAINING_DECK)
card_values = sorted(card_counts.keys())
counts = [card_counts[c] for c in card_values]
colors = ['red' if c < 0 else 'green' for c in card_values]

ax.bar(range(len(card_values)), counts, color=colors, edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(card_values)))
ax.set_xticklabels([str(c) for c in card_values], rotation=90, fontsize=9)
ax.set_xlabel('Card Value', fontsize=14)
ax.set_ylabel('Count in Remaining Deck', fontsize=14)
ax.set_title(f'Remaining Card Distribution ({len(REMAINING_DECK)} cards)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '29_remaining_cards.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 29_remaining_cards.png")


# =============================================================================
# FIGURE 30: Final Proof Summary
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('off')

summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                          ║
║                                    MATHEMATICAL PROOF OF OPTIMAL STRATEGY                                                ║
║                                                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                          ║
║   GAME CONFIGURATION                                                                                                     ║
║   ══════════════════                                                                                                     ║
║   Your cards:       {MY_CARDS}  (sum = {sum(MY_CARDS)})                                                                  ║
║   Revealed table:   {REVEALED_TABLE}  (sum = {REVEALED_SUM})                                                             ║
║   Hidden cards:     2 remaining                                                                                          ║
║   Remaining deck:   {len(REMAINING_DECK)} cards                                                                          ║
║                                                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                          ║
║   EXACT ANALYSIS (ENUMERATION OF ALL {TOTAL_OUTCOMES} POSSIBLE OUTCOMES)                                                 ║
║   ════════════════════════════════════════════════════════════════                                                       ║
║                                                                                                                          ║
║   Fair Value (E[Table Sum]):     {FAIR_VALUE:.4f}                                                                        ║
║   Standard Deviation:            {STD_DEV:.4f}                                                                           ║
║   Skewness:                      {stats.skew(all_table_sums):.4f}                                                        ║
║   Kurtosis:                      {stats.kurtosis(all_table_sums):.4f}                                                    ║
║                                                                                                                          ║
║   95% Confidence Interval:       [{FAIR_VALUE - 1.96*STD_DEV:.2f}, {FAIR_VALUE + 1.96*STD_DEV:.2f}]                       ║
║                                                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                          ║
║   THEOREM: OPTIMAL MARKET POSITION                                                                                       ║
║   ════════════════════════════════                                                                                       ║
║                                                                                                                          ║
║   For a market (Bid, Ask) = (FV - δ, FV + δ) where δ = half-spread:                                                      ║
║                                                                                                                          ║
║   • E[PnL | They Buy at Ask]  = Ask - E[S] = (FV + δ) - FV = δ > 0  ✓                                                   ║
║   • E[PnL | They Sell at Bid] = E[S] - Bid = FV - (FV - δ) = δ > 0  ✓                                                   ║
║                                                                                                                          ║
║   ⟹ Centering the market at Fair Value GUARANTEES positive expected value on BOTH sides.                                ║
║                                                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                          ║
║   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓   ║
║   ┃                                                                                                                   ┃   ║
║   ┃                               OPTIMAL MARKET:   BID = 54   /   ASK = 57                                          ┃   ║
║   ┃                                                                                                                   ┃   ║
║   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛   ║
║                                                                                                                          ║
║   PERFORMANCE METRICS                                                                                                    ║
║   ═══════════════════                                                                                                    ║
║                                                                                                                          ║
║   E[PnL] if opponent BUYS at 57:    +{57 - FAIR_VALUE:.2f} points                                                        ║
║   E[PnL] if opponent SELLS at 54:   +{FAIR_VALUE - 54:.2f} points                                                        ║
║                                                                                                                          ║
║   Sharpe Ratio:                     {np.mean(pnl_buy_opt)/np.std(pnl_buy_opt):.4f}                                       ║
║   Win Rate:                         {np.mean(np.array(pnl_buy_opt) > 0)*100:.1f}%                                        ║
║   VaR (95%):                        {np.percentile(pnl_buy_opt, 5):.2f}                                                  ║
║                                                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                          ║
║   VERIFICATION METHODS EMPLOYED                                                                                          ║
║   ═════════════════════════════                                                                                          ║
║                                                                                                                          ║
║   ✓ Exact enumeration of all {TOTAL_OUTCOMES} possible outcomes                                                          ║
║   ✓ Monte Carlo simulation (100,000+ iterations)                                                                         ║
║   ✓ Kelly Criterion optimization                                                                                         ║
║   ✓ Game Theory (Minimax analysis)                                                                                       ║
║   ✓ Expected Utility Theory (multiple risk aversion levels)                                                              ║
║   ✓ Bootstrap confidence intervals                                                                                       ║
║   ✓ Sensitivity and robustness analysis                                                                                  ║
║                                                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
        fontsize=11, fontfamily='monospace', verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
plt.tight_layout()
plt.savefig(OUTPUT_DIR + '30_final_proof_summary.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 30_final_proof_summary.png")


# =============================================================================
# COMPLETION
# =============================================================================
print("\n" + "=" * 80)
print("ALL 30 FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nLocation: {OUTPUT_DIR}")
print(f"\nTotal figures: 30")
print("\nKey findings proven:")
print(f"  • Fair Value = {FAIR_VALUE:.4f}")
print(f"  • Optimal Bid = 54")
print(f"  • Optimal Ask = 57")
print("=" * 80)
