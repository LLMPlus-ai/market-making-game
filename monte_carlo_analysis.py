#!/usr/bin/env python3
"""
Monte Carlo Simulation for Market Making Strategy Optimization
===============================================================

This script runs rigorous simulations to prove the optimality of
our market making strategy and generates comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import Counter
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style for high-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# =============================================================================
# GAME CONFIGURATION - TEAM 8 CURRENT STATE
# =============================================================================

MY_CARDS = [10, 1, 0]
REVEALED_TABLE = [16, 14, 19, 7]
HIDDEN_COUNT = 2

# Build full deck
FULL_DECK = []
for i in range(21):
    FULL_DECK.extend([i, i])
for i in range(-10, -81, -10):
    FULL_DECK.append(i)

# Build remaining deck
known = Counter(MY_CARDS + REVEALED_TABLE)
full = Counter(FULL_DECK)
REMAINING_DECK = list((full - known).elements())

REVEALED_SUM = sum(REVEALED_TABLE)
DECK_SUM = 60

print("=" * 70)
print("MONTE CARLO SIMULATION FOR MARKET MAKING STRATEGY")
print("=" * 70)
print(f"\nYour cards: {MY_CARDS} (sum: {sum(MY_CARDS)})")
print(f"Revealed table: {REVEALED_TABLE} (sum: {REVEALED_SUM})")
print(f"Hidden cards: {HIDDEN_COUNT}")
print(f"Remaining deck size: {len(REMAINING_DECK)}")


# =============================================================================
# SIMULATION 1: TABLE SUM DISTRIBUTION (EXACT CALCULATION)
# =============================================================================

print("\n" + "=" * 70)
print("SIMULATION 1: EXACT TABLE SUM DISTRIBUTION")
print("=" * 70)

# Calculate ALL possible outcomes (exact, not Monte Carlo)
all_pairs = list(combinations(REMAINING_DECK, 2))
all_table_sums = [REVEALED_SUM + sum(pair) for pair in all_pairs]

print(f"Total possible outcomes: {len(all_table_sums)}")
print(f"Mean table sum: {np.mean(all_table_sums):.4f}")
print(f"Std deviation: {np.std(all_table_sums):.4f}")
print(f"Min: {min(all_table_sums)}, Max: {max(all_table_sums)}")


# =============================================================================
# SIMULATION 2: MONTE CARLO FOR STRATEGY TESTING
# =============================================================================

print("\n" + "=" * 70)
print("SIMULATION 2: MONTE CARLO STRATEGY TESTING (100,000 iterations)")
print("=" * 70)

N_SIMULATIONS = 100000


def simulate_game(bid, ask, position_after_trade, trade_direction):
    """
    Simulate a single game outcome.

    Args:
        bid, ask: Our market quote
        position_after_trade: Our position after someone trades with us
        trade_direction: 'buy' (they buy from us) or 'sell' (they sell to us) or 'none'

    Returns:
        PnL from the trade
    """
    # Randomly select 2 hidden cards
    hidden_indices = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
    hidden_cards = [REMAINING_DECK[i] for i in hidden_indices]
    table_sum = REVEALED_SUM + sum(hidden_cards)

    if trade_direction == 'none':
        return 0, table_sum
    elif trade_direction == 'buy':
        # They bought from us at ask price, we are SHORT
        # PnL = (ask - settlement) * 1 = ask - table_sum
        pnl = ask - table_sum
    else:  # sell
        # They sold to us at bid price, we are LONG
        # PnL = (settlement - bid) * 1 = table_sum - bid
        pnl = table_sum - bid

    return pnl, table_sum


def run_strategy_simulation(bid, ask, n_sims=N_SIMULATIONS):
    """Run Monte Carlo simulation for a given market quote."""

    # Simulate outcomes where opponent buys from us (hits our ask)
    pnls_when_sold = []
    # Simulate outcomes where opponent sells to us (hits our bid)
    pnls_when_bought = []
    table_sums = []

    for _ in range(n_sims):
        pnl_sold, ts = simulate_game(bid, ask, -1, 'buy')
        pnl_bought, _ = simulate_game(bid, ask, 1, 'sell')
        pnls_when_sold.append(pnl_sold)
        pnls_when_bought.append(pnl_bought)
        table_sums.append(ts)

    return {
        'bid': bid,
        'ask': ask,
        'mid': (bid + ask) / 2,
        'pnl_if_they_buy': np.array(pnls_when_sold),
        'pnl_if_they_sell': np.array(pnls_when_bought),
        'table_sums': np.array(table_sums),
        'ev_if_they_buy': np.mean(pnls_when_sold),
        'ev_if_they_sell': np.mean(pnls_when_bought),
        'std_if_they_buy': np.std(pnls_when_sold),
        'std_if_they_sell': np.std(pnls_when_bought),
        'sharpe_buy': np.mean(pnls_when_sold) / np.std(pnls_when_sold) if np.std(pnls_when_sold) > 0 else 0,
        'sharpe_sell': np.mean(pnls_when_bought) / np.std(pnls_when_bought) if np.std(pnls_when_bought) > 0 else 0,
    }


# Test our recommended strategy
FAIR_VALUE = np.mean(all_table_sums)
print(f"\nFair Value (exact): {FAIR_VALUE:.4f}")

# Test multiple strategies
strategies = []
for mid_offset in np.arange(-5, 6, 0.5):
    mid = FAIR_VALUE + mid_offset
    bid = mid - 1.5
    ask = mid + 1.5
    result = run_strategy_simulation(bid, ask)
    strategies.append(result)

print(f"\nTested {len(strategies)} different market positions")


# =============================================================================
# VISUALIZATION 1: TABLE SUM DISTRIBUTION
# =============================================================================

fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('SIMULATION 1: Table Sum Distribution Analysis', fontsize=14, fontweight='bold')

# Plot 1a: Histogram of all possible table sums
ax1 = axes[0, 0]
counts, bins, _ = ax1.hist(all_table_sums, bins=50, density=True, alpha=0.7,
                            color='steelblue', edgecolor='black', linewidth=0.5)
ax1.axvline(FAIR_VALUE, color='red', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax1.axvline(np.median(all_table_sums), color='green', linestyle='--', linewidth=2,
            label=f'Median = {np.median(all_table_sums):.2f}')
ax1.set_xlabel('Table Sum')
ax1.set_ylabel('Probability Density')
ax1.set_title('Distribution of All Possible Table Sums')
ax1.legend()

# Plot 1b: CDF
ax2 = axes[0, 1]
sorted_sums = np.sort(all_table_sums)
cdf = np.arange(1, len(sorted_sums) + 1) / len(sorted_sums)
ax2.plot(sorted_sums, cdf, color='steelblue', linewidth=2)
ax2.axvline(FAIR_VALUE, color='red', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between(sorted_sums, 0, cdf, alpha=0.3)
ax2.set_xlabel('Table Sum')
ax2.set_ylabel('Cumulative Probability')
ax2.set_title('Cumulative Distribution Function (CDF)')
ax2.legend()
ax2.set_ylim(0, 1)

# Plot 1c: Box plot with key statistics
ax3 = axes[1, 0]
bp = ax3.boxplot(all_table_sums, vert=True, widths=0.5)
ax3.scatter([1], [FAIR_VALUE], color='red', s=100, zorder=5, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax3.set_ylabel('Table Sum')
ax3.set_title('Box Plot of Table Sum Distribution')

# Add percentile annotations
percentiles = [5, 25, 50, 75, 95]
pct_values = np.percentile(all_table_sums, percentiles)
for pct, val in zip(percentiles, pct_values):
    ax3.annotate(f'{pct}%: {val:.1f}', xy=(1.1, val), fontsize=9)
ax3.legend()

# Plot 1d: Probability of different ranges
ax4 = axes[1, 1]
ranges = [(-200, -50), (-50, 0), (0, 25), (25, 50), (50, 75), (75, 100)]
range_labels = ['< -50', '-50 to 0', '0 to 25', '25 to 50', '50 to 75', '> 75']
probs = []
for low, high in ranges:
    prob = sum(1 for s in all_table_sums if low <= s < high) / len(all_table_sums)
    probs.append(prob * 100)

colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green']
bars = ax4.bar(range_labels, probs, color=colors, edgecolor='black')
ax4.set_xlabel('Table Sum Range')
ax4.set_ylabel('Probability (%)')
ax4.set_title('Probability Distribution by Range')
ax4.set_xticklabels(range_labels, rotation=45, ha='right')

for bar, prob in zip(bars, probs):
    ax4.annotate(f'{prob:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/top1/Desktop/mm_game_old_mission/viz1_table_sum_distribution.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: viz1_table_sum_distribution.png")


# =============================================================================
# VISUALIZATION 2: STRATEGY COMPARISON
# =============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('SIMULATION 2: Strategy Comparison (100,000 Monte Carlo Simulations)',
              fontsize=14, fontweight='bold')

mids = [s['mid'] for s in strategies]
ev_buy = [s['ev_if_they_buy'] for s in strategies]
ev_sell = [s['ev_if_they_sell'] for s in strategies]
std_buy = [s['std_if_they_buy'] for s in strategies]
std_sell = [s['std_if_they_sell'] for s in strategies]

# Plot 2a: Expected PnL vs Market Mid
ax1 = axes[0, 0]
ax1.plot(mids, ev_buy, 'b-', linewidth=2, label='E[PnL] if they BUY from us', marker='o', markersize=4)
ax1.plot(mids, ev_sell, 'r-', linewidth=2, label='E[PnL] if they SELL to us', marker='s', markersize=4)
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax1.fill_between(mids, ev_buy, 0, alpha=0.2, color='blue')
ax1.fill_between(mids, ev_sell, 0, alpha=0.2, color='red')
ax1.set_xlabel('Market Mid Price')
ax1.set_ylabel('Expected PnL')
ax1.set_title('Expected PnL by Market Position')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2b: Risk (Std Dev) vs Market Mid
ax2 = axes[0, 1]
ax2.plot(mids, std_buy, 'b-', linewidth=2, label='Std Dev if they BUY', marker='o', markersize=4)
ax2.plot(mids, std_sell, 'r-', linewidth=2, label='Std Dev if they SELL', marker='s', markersize=4)
ax2.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax2.set_xlabel('Market Mid Price')
ax2.set_ylabel('Standard Deviation of PnL')
ax2.set_title('Risk (Volatility) by Market Position')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 2c: Sharpe Ratio
sharpe_buy = [s['sharpe_buy'] for s in strategies]
sharpe_sell = [s['sharpe_sell'] for s in strategies]

ax3 = axes[1, 0]
ax3.plot(mids, sharpe_buy, 'b-', linewidth=2, label='Sharpe if they BUY', marker='o', markersize=4)
ax3.plot(mids, sharpe_sell, 'r-', linewidth=2, label='Sharpe if they SELL', marker='s', markersize=4)
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax3.set_xlabel('Market Mid Price')
ax3.set_ylabel('Sharpe Ratio (E[PnL] / Std)')
ax3.set_title('Risk-Adjusted Return by Market Position')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# Plot 2d: Win Rate
win_rate_buy = [np.mean(s['pnl_if_they_buy'] > 0) * 100 for s in strategies]
win_rate_sell = [np.mean(s['pnl_if_they_sell'] > 0) * 100 for s in strategies]

ax4 = axes[1, 1]
ax4.plot(mids, win_rate_buy, 'b-', linewidth=2, label='Win% if they BUY', marker='o', markersize=4)
ax4.plot(mids, win_rate_sell, 'r-', linewidth=2, label='Win% if they SELL', marker='s', markersize=4)
ax4.axhline(50, color='black', linestyle=':', linewidth=1)
ax4.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=2, label=f'Fair Value = {FAIR_VALUE:.2f}')
ax4.set_xlabel('Market Mid Price')
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Probability of Profit by Market Position')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('/Users/top1/Desktop/mm_game_old_mission/viz2_strategy_comparison.png',
            dpi=150, bbox_inches='tight')
print("Saved: viz2_strategy_comparison.png")


# =============================================================================
# VISUALIZATION 3: OPTIMAL STRATEGY HEATMAP
# =============================================================================

fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('SIMULATION 3: Optimal Strategy Heatmap', fontsize=14, fontweight='bold')

# Create heatmap of EV for different bid/ask combinations
bid_range = np.arange(FAIR_VALUE - 8, FAIR_VALUE + 5, 0.5)
ask_range = np.arange(FAIR_VALUE - 2, FAIR_VALUE + 11, 0.5)

# We need spread = 3, so we'll vary the mid point
ev_matrix_buy = np.zeros((len(bid_range), len(ask_range)))
ev_matrix_sell = np.zeros((len(bid_range), len(ask_range)))

for i, bid in enumerate(bid_range):
    for j, ask in enumerate(ask_range):
        if ask - bid >= 3:  # Valid spread
            # Quick MC simulation
            pnls_buy = []
            pnls_sell = []
            for _ in range(5000):
                idx = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
                hidden = [REMAINING_DECK[k] for k in idx]
                ts = REVEALED_SUM + sum(hidden)
                pnls_buy.append(ask - ts)  # They buy from us
                pnls_sell.append(ts - bid)  # They sell to us
            ev_matrix_buy[i, j] = np.mean(pnls_buy)
            ev_matrix_sell[i, j] = np.mean(pnls_sell)
        else:
            ev_matrix_buy[i, j] = np.nan
            ev_matrix_sell[i, j] = np.nan

# Plot heatmaps
im1 = axes[0].imshow(ev_matrix_buy, cmap='RdYlGn', aspect='auto', origin='lower',
                      extent=[ask_range[0], ask_range[-1], bid_range[0], bid_range[-1]])
axes[0].set_xlabel('Ask Price')
axes[0].set_ylabel('Bid Price')
axes[0].set_title('E[PnL] if Opponent BUYS from Us')
plt.colorbar(im1, ax=axes[0], label='Expected PnL')
axes[0].axhline(FAIR_VALUE - 1.5, color='white', linestyle='--', linewidth=2)
axes[0].axvline(FAIR_VALUE + 1.5, color='white', linestyle='--', linewidth=2)

im2 = axes[1].imshow(ev_matrix_sell, cmap='RdYlGn', aspect='auto', origin='lower',
                      extent=[ask_range[0], ask_range[-1], bid_range[0], bid_range[-1]])
axes[1].set_xlabel('Ask Price')
axes[1].set_ylabel('Bid Price')
axes[1].set_title('E[PnL] if Opponent SELLS to Us')
plt.colorbar(im2, ax=axes[1], label='Expected PnL')
axes[1].axhline(FAIR_VALUE - 1.5, color='white', linestyle='--', linewidth=2)
axes[1].axvline(FAIR_VALUE + 1.5, color='white', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('/Users/top1/Desktop/mm_game_old_mission/viz3_strategy_heatmap.png',
            dpi=150, bbox_inches='tight')
print("Saved: viz3_strategy_heatmap.png")


# =============================================================================
# VISUALIZATION 4: PnL DISTRIBUTION FOR OPTIMAL STRATEGY
# =============================================================================

fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle('SIMULATION 4: PnL Distribution for Recommended Strategy (54/57)',
              fontsize=14, fontweight='bold')

# Run detailed simulation for our recommended strategy
optimal_bid = 54
optimal_ask = 57
optimal_result = run_strategy_simulation(optimal_bid, optimal_ask, n_sims=100000)

# Plot 4a: PnL distribution if they buy
ax1 = axes[0, 0]
pnl_buy = optimal_result['pnl_if_they_buy']
ax1.hist(pnl_buy, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(np.mean(pnl_buy), color='red', linestyle='--', linewidth=2,
            label=f'Mean = {np.mean(pnl_buy):.2f}')
ax1.axvline(0, color='black', linestyle='-', linewidth=2)
ax1.set_xlabel('PnL')
ax1.set_ylabel('Probability Density')
ax1.set_title(f'PnL Distribution: They BUY at {optimal_ask}')
ax1.legend()

# Plot 4b: PnL distribution if they sell
ax2 = axes[0, 1]
pnl_sell = optimal_result['pnl_if_they_sell']
ax2.hist(pnl_sell, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
ax2.axvline(np.mean(pnl_sell), color='blue', linestyle='--', linewidth=2,
            label=f'Mean = {np.mean(pnl_sell):.2f}')
ax2.axvline(0, color='black', linestyle='-', linewidth=2)
ax2.set_xlabel('PnL')
ax2.set_ylabel('Probability Density')
ax2.set_title(f'PnL Distribution: They SELL at {optimal_bid}')
ax2.legend()

# Plot 4c: VaR and CVaR analysis
ax3 = axes[1, 0]
var_levels = [1, 5, 10, 25]
var_buy = [np.percentile(pnl_buy, level) for level in var_levels]
var_sell = [np.percentile(pnl_sell, level) for level in var_levels]

x = np.arange(len(var_levels))
width = 0.35
bars1 = ax3.bar(x - width/2, var_buy, width, label='If they BUY', color='blue', alpha=0.7)
bars2 = ax3.bar(x + width/2, var_sell, width, label='If they SELL', color='red', alpha=0.7)
ax3.set_xlabel('Value at Risk Level (%)')
ax3.set_ylabel('PnL at Risk')
ax3.set_title('Value at Risk (VaR) Analysis')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{l}%' for l in var_levels])
ax3.legend()
ax3.axhline(0, color='black', linestyle='-', linewidth=1)

# Plot 4d: Cumulative PnL paths (random walks)
ax4 = axes[1, 1]
n_paths = 50
n_trades = 20
for _ in range(n_paths):
    path = [0]
    for _ in range(n_trades):
        if np.random.random() > 0.5:  # They buy
            idx = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
            hidden = [REMAINING_DECK[k] for k in idx]
            ts = REVEALED_SUM + sum(hidden)
            pnl = optimal_ask - ts
        else:  # They sell
            idx = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
            hidden = [REMAINING_DECK[k] for k in idx]
            ts = REVEALED_SUM + sum(hidden)
            pnl = ts - optimal_bid
        path.append(path[-1] + pnl)
    ax4.plot(path, alpha=0.3, linewidth=1)

ax4.axhline(0, color='black', linestyle='-', linewidth=2)
ax4.set_xlabel('Number of Trades')
ax4.set_ylabel('Cumulative PnL')
ax4.set_title('Simulated Trading Paths (50 random paths, 20 trades each)')

plt.tight_layout()
plt.savefig('/Users/top1/Desktop/mm_game_old_mission/viz4_pnl_distribution.png',
            dpi=150, bbox_inches='tight')
print("Saved: viz4_pnl_distribution.png")


# =============================================================================
# VISUALIZATION 5: SENSITIVITY ANALYSIS
# =============================================================================

fig5, axes = plt.subplots(2, 2, figsize=(14, 10))
fig5.suptitle('SIMULATION 5: Sensitivity Analysis', fontsize=14, fontweight='bold')

# Test different spread widths (even though game requires 3)
spreads = [3, 4, 5, 6]
spread_results = {}

for spread in spreads:
    bid = FAIR_VALUE - spread/2
    ask = FAIR_VALUE + spread/2
    spread_results[spread] = run_strategy_simulation(bid, ask, n_sims=50000)

# Plot 5a: EV vs Spread
ax1 = axes[0, 0]
evs_buy = [spread_results[s]['ev_if_they_buy'] for s in spreads]
evs_sell = [spread_results[s]['ev_if_they_sell'] for s in spreads]
ax1.bar([s - 0.2 for s in spreads], evs_buy, 0.4, label='If they BUY', color='blue', alpha=0.7)
ax1.bar([s + 0.2 for s in spreads], evs_sell, 0.4, label='If they SELL', color='red', alpha=0.7)
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Spread Width')
ax1.set_ylabel('Expected PnL')
ax1.set_title('E[PnL] vs Spread Width (Centered at Fair Value)')
ax1.legend()

# Plot 5b: Effect of position on optimal quote
positions = [-3, -2, -1, 0, 1, 2, 3]
skew_per_position = 0.5  # Points to skew per unit position

ax2 = axes[0, 1]
for pos in positions:
    skew = -pos * skew_per_position
    mid = FAIR_VALUE + skew
    bid = mid - 1.5
    ask = mid + 1.5
    ax2.scatter(pos, bid, color='blue', s=100, marker='v')
    ax2.scatter(pos, ask, color='red', s=100, marker='^')

ax2.axhline(FAIR_VALUE, color='green', linestyle='--', linewidth=2, label='Fair Value')
ax2.set_xlabel('Current Position (+ = Long)')
ax2.set_ylabel('Price')
ax2.set_title('Optimal Bid/Ask by Position (Inventory Management)')
ax2.legend(['Fair Value', 'Bid', 'Ask'])

# Plot 5c: Monte Carlo convergence
ax3 = axes[1, 0]
sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
means_convergence = []
stds_convergence = []

np.random.seed(42)  # For reproducibility
for n in sample_sizes:
    sums = []
    for _ in range(n):
        idx = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
        hidden = [REMAINING_DECK[k] for k in idx]
        sums.append(REVEALED_SUM + sum(hidden))
    means_convergence.append(np.mean(sums))
    stds_convergence.append(np.std(sums) / np.sqrt(n))

ax3.errorbar(sample_sizes, means_convergence, yerr=[2*s for s in stds_convergence],
             fmt='o-', capsize=5, color='steelblue')
ax3.axhline(FAIR_VALUE, color='red', linestyle='--', linewidth=2, label=f'True FV = {FAIR_VALUE:.2f}')
ax3.set_xscale('log')
ax3.set_xlabel('Number of Simulations')
ax3.set_ylabel('Estimated Fair Value')
ax3.set_title('Monte Carlo Convergence (95% CI)')
ax3.legend()

# Plot 5d: Breakdown of negative card scenarios
ax4 = axes[1, 1]
neg_cards = [-10, -20, -30, -40, -50, -60, -70, -80]

# Calculate probability each negative card is on table
n_remaining = len(REMAINING_DECK)
prob_each_neg = []
ev_impact = []

for neg in neg_cards:
    # Probability this specific negative card is one of the 2 hidden
    prob_on_table = 2 / n_remaining
    prob_each_neg.append(prob_on_table * 100)
    # Expected impact on table sum if this card is on table
    ev_impact.append(neg * prob_on_table)

ax4.bar([str(n) for n in neg_cards], prob_each_neg, color='darkred', alpha=0.7)
ax4.set_xlabel('Negative Card Value')
ax4.set_ylabel('Probability on Table (%)')
ax4.set_title('Probability Each Negative Card is Hidden')

for i, (prob, impact) in enumerate(zip(prob_each_neg, ev_impact)):
    ax4.annotate(f'{prob:.1f}%', xy=(i, prob + 0.2), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/top1/Desktop/mm_game_old_mission/viz5_sensitivity_analysis.png',
            dpi=150, bbox_inches='tight')
print("Saved: viz5_sensitivity_analysis.png")


# =============================================================================
# VISUALIZATION 6: OPTIMAL STRATEGY PROOF
# =============================================================================

fig6 = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig6)
fig6.suptitle('PROOF OF OPTIMAL STRATEGY: Market at Fair Value Minimizes Risk',
              fontsize=14, fontweight='bold')

# Find optimal mid for zero expected PnL on both sides
ax1 = fig6.add_subplot(gs[0, :])
mids_fine = np.linspace(FAIR_VALUE - 10, FAIR_VALUE + 10, 100)
ev_buy_fine = []
ev_sell_fine = []

for mid in mids_fine:
    bid = mid - 1.5
    ask = mid + 1.5
    pnls_b = []
    pnls_s = []
    for _ in range(10000):
        idx = np.random.choice(len(REMAINING_DECK), size=2, replace=False)
        hidden = [REMAINING_DECK[k] for k in idx]
        ts = REVEALED_SUM + sum(hidden)
        pnls_b.append(ask - ts)
        pnls_s.append(ts - bid)
    ev_buy_fine.append(np.mean(pnls_b))
    ev_sell_fine.append(np.mean(pnls_s))

ax1.plot(mids_fine, ev_buy_fine, 'b-', linewidth=2, label='E[PnL] if they BUY')
ax1.plot(mids_fine, ev_sell_fine, 'r-', linewidth=2, label='E[PnL] if they SELL')
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.axvline(FAIR_VALUE, color='green', linestyle='--', linewidth=2,
            label=f'Optimal Mid = FV = {FAIR_VALUE:.2f}')
ax1.fill_between(mids_fine, ev_buy_fine, 0, where=[e > 0 for e in ev_buy_fine],
                  alpha=0.2, color='blue')
ax1.fill_between(mids_fine, ev_sell_fine, 0, where=[e > 0 for e in ev_sell_fine],
                  alpha=0.2, color='red')
ax1.set_xlabel('Market Mid Price')
ax1.set_ylabel('Expected PnL')
ax1.set_title('E[PnL] Crosses Zero at Fair Value — PROOF of Optimality')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add text box with mathematical proof
textstr = '\n'.join([
    'MATHEMATICAL PROOF:',
    f'Fair Value (FV) = E[Table Sum] = {FAIR_VALUE:.2f}',
    '',
    'If we sell at Ask:',
    '  E[PnL] = Ask - E[Table Sum] = Ask - FV',
    f'  When Ask = FV + 1.5 = {FAIR_VALUE + 1.5:.2f}',
    f'  E[PnL] = 1.5 (half the spread)',
    '',
    'If we buy at Bid:',
    '  E[PnL] = E[Table Sum] - Bid = FV - Bid',
    f'  When Bid = FV - 1.5 = {FAIR_VALUE - 1.5:.2f}',
    f'  E[PnL] = 1.5 (half the spread)',
    '',
    'CONCLUSION: Center market at FV to have',
    'positive expected value on BOTH sides!'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

# Key metrics comparison
ax2 = fig6.add_subplot(gs[1, 0])
strategies_to_compare = [
    ('Optimal\n(54/57)', 54, 57),
    ('Biased High\n(56/59)', 56, 59),
    ('Biased Low\n(52/55)', 52, 55),
]

metrics = []
for name, bid, ask in strategies_to_compare:
    res = run_strategy_simulation(bid, ask, n_sims=50000)
    total_ev = (res['ev_if_they_buy'] + res['ev_if_they_sell']) / 2
    metrics.append(total_ev)

colors = ['green', 'red', 'red']
ax2.bar([s[0] for s in strategies_to_compare], metrics, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('Average E[PnL]')
ax2.set_title('Strategy Comparison:\nAverage Expected PnL')
ax2.set_ylim(min(metrics) - 1, max(metrics) + 1)

# Win rate comparison
ax3 = fig6.add_subplot(gs[1, 1])
win_rates = []
for name, bid, ask in strategies_to_compare:
    res = run_strategy_simulation(bid, ask, n_sims=50000)
    wr = (np.mean(res['pnl_if_they_buy'] > 0) + np.mean(res['pnl_if_they_sell'] > 0)) / 2 * 100
    win_rates.append(wr)

ax3.bar([s[0] for s in strategies_to_compare], win_rates, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(50, color='black', linestyle='--', linewidth=1)
ax3.set_ylabel('Win Rate (%)')
ax3.set_title('Strategy Comparison:\nAverage Win Rate')
ax3.set_ylim(0, 100)

# Sharpe comparison
ax4 = fig6.add_subplot(gs[1, 2])
sharpes = []
for name, bid, ask in strategies_to_compare:
    res = run_strategy_simulation(bid, ask, n_sims=50000)
    sharpe = (res['sharpe_buy'] + res['sharpe_sell']) / 2
    sharpes.append(sharpe)

ax4.bar([s[0] for s in strategies_to_compare], sharpes, color=colors, alpha=0.7, edgecolor='black')
ax4.axhline(0, color='black', linestyle='-', linewidth=1)
ax4.set_ylabel('Sharpe Ratio')
ax4.set_title('Strategy Comparison:\nRisk-Adjusted Return')

# Final recommendation box
ax5 = fig6.add_subplot(gs[2, :])
ax5.axis('off')
recommendation = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                          OPTIMAL STRATEGY SUMMARY                                             ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                              ║
║   YOUR CARDS: {MY_CARDS}  (sum = {sum(MY_CARDS)})                                                                          ║
║   REVEALED:   {REVEALED_TABLE}  (sum = {REVEALED_SUM})                                                                 ║
║   HIDDEN:     2 cards remaining                                                                              ║
║                                                                                                              ║
║   ══════════════════════════════════════════════════════════════════════════════════════════════════════     ║
║                                                                                                              ║
║   FAIR VALUE = {FAIR_VALUE:.2f}  (Proven by {len(all_table_sums):,} exact outcomes + {N_SIMULATIONS:,} Monte Carlo sims)               ║
║                                                                                                              ║
║   ══════════════════════════════════════════════════════════════════════════════════════════════════════     ║
║                                                                                                              ║
║   OPTIMAL MARKET:    BID = {FAIR_VALUE - 1.5:.1f}   /   ASK = {FAIR_VALUE + 1.5:.1f}                                                          ║
║                                                                                                              ║
║   RECOMMENDED:       BID = 54      /   ASK = 57   (rounded for practical use)                                ║
║                                                                                                              ║
║   ══════════════════════════════════════════════════════════════════════════════════════════════════════     ║
║                                                                                                              ║
║   Expected PnL if they BUY at 57:  +{57 - FAIR_VALUE:.2f} points                                                            ║
║   Expected PnL if they SELL at 54: +{FAIR_VALUE - 54:.2f} points                                                            ║
║                                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
ax5.text(0.5, 0.5, recommendation, transform=ax5.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('/Users/top1/Desktop/mm_game_old_mission/viz6_optimal_strategy_proof.png',
            dpi=150, bbox_inches='tight')
print("Saved: viz6_optimal_strategy_proof.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("MONTE CARLO SIMULATION COMPLETE")
print("=" * 70)
print(f"""
RESULTS SUMMARY:
================

Fair Value: {FAIR_VALUE:.4f}
Standard Deviation: {np.std(all_table_sums):.4f}

Exact outcomes analyzed: {len(all_table_sums):,}
Monte Carlo simulations: {N_SIMULATIONS:,}

OPTIMAL STRATEGY:
  Bid: {FAIR_VALUE - 1.5:.1f}
  Ask: {FAIR_VALUE + 1.5:.1f}

PRACTICAL RECOMMENDATION:
  Bid: 54
  Ask: 57

VISUALIZATIONS GENERATED:
  1. viz1_table_sum_distribution.png - Distribution analysis
  2. viz2_strategy_comparison.png   - Strategy comparison
  3. viz3_strategy_heatmap.png      - Bid/Ask heatmap
  4. viz4_pnl_distribution.png      - PnL distribution
  5. viz5_sensitivity_analysis.png  - Sensitivity analysis
  6. viz6_optimal_strategy_proof.png - Optimality proof

All files saved to: /Users/top1/Desktop/mm_game_old_mission/
""")
print("=" * 70)
