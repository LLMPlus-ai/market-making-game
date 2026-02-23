"""
Market Making Game - Mathematical Model & Decision Engine
=========================================================

Deck: 50 cards
- Two of each: 0, 1, 2, ..., 20 (42 cards)
- One each of: -10, -20, -30, -40, -50, -60, -70, -80 (8 cards)
- Total sum of deck = +60

Goal: Estimate the sum of 6 table cards and make profitable markets.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import Counter
import itertools
from functools import lru_cache


# =============================================================================
# DECK MODEL
# =============================================================================

def create_full_deck() -> List[int]:
    """Create the full deck of 50 cards."""
    deck = []
    # Two of each 0-20
    for i in range(21):
        deck.extend([i, i])
    # One each of -10 to -80
    for i in range(-10, -81, -10):
        deck.append(i)
    return sorted(deck)


FULL_DECK = create_full_deck()
DECK_SUM = sum(FULL_DECK)  # Should be 60
DECK_SIZE = len(FULL_DECK)  # Should be 50

# Verify deck
assert DECK_SUM == 60, f"Deck sum should be 60, got {DECK_SUM}"
assert DECK_SIZE == 50, f"Deck size should be 50, got {DECK_SIZE}"


# =============================================================================
# GAME STATE
# =============================================================================

@dataclass
class GameState:
    """Tracks the current state of the game."""

    # Your private cards
    my_cards: List[int] = field(default_factory=list)

    # Revealed table cards
    revealed_table_cards: List[int] = field(default_factory=list)

    # Number of hidden table cards (starts at 6, decreases as cards revealed)
    hidden_table_count: int = 6

    # Current round (1, 2, or 3)
    current_round: int = 1

    # Your current position (positive = long, negative = short)
    position: int = 0

    # Your average entry price
    avg_entry_price: float = 0.0

    # Trade history for PnL tracking
    trades: List[Dict] = field(default_factory=list)

    # Observed market quotes from other teams (for Bayesian inference)
    observed_quotes: List[Tuple[float, float]] = field(default_factory=list)

    # Known cards held by others (if any information leaked)
    known_other_cards: List[int] = field(default_factory=list)

    def get_known_cards(self) -> List[int]:
        """Get all cards we know have been removed from the deck."""
        return self.my_cards + self.revealed_table_cards + self.known_other_cards

    def get_remaining_deck(self) -> List[int]:
        """Get cards that could still be on table or with other teams."""
        known = Counter(self.get_known_cards())
        full = Counter(FULL_DECK)
        remaining = full - known
        return list(remaining.elements())

    def get_remaining_sum(self) -> int:
        """Get sum of remaining unknown cards."""
        return DECK_SUM - sum(self.get_known_cards())

    def get_remaining_count(self) -> int:
        """Get count of remaining unknown cards."""
        return DECK_SIZE - len(self.get_known_cards())


# =============================================================================
# EXPECTED VALUE ESTIMATOR (Core Mathematical Model)
# =============================================================================

class ExpectedValueEstimator:
    """
    Estimates the expected value of the sum of 6 table cards.

    Mathematical Framework:
    -----------------------
    Let S = sum of 6 table cards (our target)
    Let Y = sum of your private cards (known)
    Let R = sum of revealed table cards (known)
    Let H = sum of hidden table cards (unknown, what we estimate)

    We know: S = R + H

    The hidden table cards are drawn from the remaining deck
    (excluding your cards and revealed table cards).

    E[H] = h × E[single remaining card]
         = h × (remaining_sum / remaining_count)

    where h = number of hidden table cards = 6 - |R|
    """

    def __init__(self, state: GameState):
        self.state = state

    def naive_expected_value(self) -> float:
        """
        Calculate expected value using only card removal information.
        This is the baseline estimate without behavioral inference.
        """
        remaining_deck = self.state.get_remaining_deck()
        remaining_sum = sum(remaining_deck)
        remaining_count = len(remaining_deck)

        if remaining_count == 0:
            # All cards known - this shouldn't happen in normal play
            return sum(self.state.revealed_table_cards)

        # Expected value per remaining card
        ev_per_card = remaining_sum / remaining_count

        # Expected sum of hidden table cards
        hidden_count = self.state.hidden_table_count
        expected_hidden_sum = hidden_count * ev_per_card

        # Total expected table sum
        revealed_sum = sum(self.state.revealed_table_cards)

        return revealed_sum + expected_hidden_sum

    def calculate_variance(self) -> float:
        """
        Calculate variance of the table sum estimate.

        Variance of sum of h cards drawn without replacement from
        remaining deck of n cards with values x_1, ..., x_n:

        Var(H) = h × (1 - (h-1)/(n-1)) × Var(X)

        where Var(X) is the variance of the remaining deck values.
        """
        remaining_deck = self.state.get_remaining_deck()
        n = len(remaining_deck)
        h = self.state.hidden_table_count

        if n <= 1 or h == 0:
            return 0.0

        # Variance of remaining deck values
        mean = np.mean(remaining_deck)
        var_x = np.var(remaining_deck)

        # Finite population correction for sampling without replacement
        fpc = (n - h) / (n - 1) if n > 1 else 1

        # Variance of sum of h cards
        variance = h * var_x * fpc

        return variance

    def calculate_std(self) -> float:
        """Calculate standard deviation of the estimate."""
        return np.sqrt(self.calculate_variance())

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the table sum.
        Uses normal approximation (valid for reasonable sample sizes).
        """
        from scipy import stats

        ev = self.naive_expected_value()
        std = self.calculate_std()

        z = stats.norm.ppf((1 + confidence) / 2)

        lower = ev - z * std
        upper = ev + z * std

        return (lower, upper)

    def full_distribution(self, sample_size: int = 10000) -> np.ndarray:
        """
        Monte Carlo simulation of possible table sums.
        Returns array of simulated sums for distribution analysis.
        """
        remaining_deck = self.state.get_remaining_deck()
        hidden_count = self.state.hidden_table_count
        revealed_sum = sum(self.state.revealed_table_cards)

        if hidden_count == 0:
            return np.array([revealed_sum])

        # Monte Carlo simulation
        sums = []
        for _ in range(sample_size):
            sample = np.random.choice(remaining_deck, size=hidden_count, replace=False)
            sums.append(revealed_sum + np.sum(sample))

        return np.array(sums)

    def probability_above(self, threshold: float, samples: int = 10000) -> float:
        """Calculate P(table_sum > threshold)."""
        distribution = self.full_distribution(samples)
        return np.mean(distribution > threshold)

    def probability_below(self, threshold: float, samples: int = 10000) -> float:
        """Calculate P(table_sum < threshold)."""
        distribution = self.full_distribution(samples)
        return np.mean(distribution < threshold)


# =============================================================================
# MARKET MAKING DECISION ENGINE
# =============================================================================

@dataclass
class MarketDecision:
    """Output of the market making decision engine."""

    # Our fair value estimate
    fair_value: float

    # Standard deviation of estimate
    uncertainty: float

    # Recommended bid-ask (if we're making the market)
    recommended_bid: float
    recommended_ask: float

    # Position-adjusted bid-ask (skewed based on inventory)
    adjusted_bid: float
    adjusted_ask: float

    # Trading recommendation
    action: str  # "BUY", "SELL", "PASS", "MAKE_MARKET"

    # Confidence in the action (0-1)
    confidence: float

    # Edge if trading at given prices
    edge_on_buy: float = 0.0
    edge_on_sell: float = 0.0

    # Explanation
    reasoning: str = ""


class MarketMaker:
    """
    Market Making Decision Engine

    Key Principles:
    1. Estimate fair value from card information
    2. Widen spread based on uncertainty
    3. Skew quotes to manage inventory risk
    4. Only trade when edge exceeds threshold
    """

    def __init__(self, state: GameState):
        self.state = state
        self.estimator = ExpectedValueEstimator(state)

        # Parameters
        self.MIN_SPREAD = 3  # Game rules: market must be 3 points wide
        self.EDGE_THRESHOLD = 0.5  # Minimum edge required to trade
        self.INVENTORY_SKEW = 0.3  # Points to skew per unit of inventory
        self.MAX_POSITION = 10  # Maximum position size

    def calculate_fair_value(self) -> Tuple[float, float]:
        """Returns (fair_value, uncertainty)."""
        fv = self.estimator.naive_expected_value()
        std = self.estimator.calculate_std()
        return fv, std

    def make_market(self) -> Tuple[float, float]:
        """
        Generate our bid-ask quote when we need to make the market.

        Strategy:
        - Center the spread around fair value
        - Skew based on current inventory
        - Spread is exactly 3 points (minimum required)
        """
        fv, std = self.calculate_fair_value()

        # Base spread (minimum 3 points as per rules)
        half_spread = self.MIN_SPREAD / 2

        # Inventory skew: if we're long, we want to sell (lower ask, lower bid)
        # if we're short, we want to buy (higher bid, higher ask)
        skew = -self.state.position * self.INVENTORY_SKEW

        bid = fv - half_spread + skew
        ask = fv + half_spread + skew

        # Round to reasonable precision
        bid = round(bid, 1)
        ask = round(ask, 1)

        # Ensure spread is exactly 3
        if ask - bid != 3:
            ask = bid + 3

        return bid, ask

    def evaluate_market(self, bid: float, ask: float) -> MarketDecision:
        """
        Evaluate whether to trade against a given market.

        Returns a complete decision with recommendation and edge calculation.
        """
        fv, std = self.calculate_fair_value()

        # Calculate edge
        edge_on_buy = fv - ask  # Positive if ask is below fair value
        edge_on_sell = bid - fv  # Positive if bid is above fair value

        # Determine action
        action = "PASS"
        confidence = 0.0
        reasoning = ""

        if edge_on_buy > self.EDGE_THRESHOLD:
            action = "BUY"
            confidence = min(1.0, edge_on_buy / (2 * std + 0.1))
            reasoning = f"Ask {ask} is {edge_on_buy:.2f} below fair value {fv:.2f}"

        elif edge_on_sell > self.EDGE_THRESHOLD:
            action = "SELL"
            confidence = min(1.0, edge_on_sell / (2 * std + 0.1))
            reasoning = f"Bid {bid} is {edge_on_sell:.2f} above fair value {fv:.2f}"

        else:
            reasoning = f"No edge: FV={fv:.2f}, Bid={bid}, Ask={ask}"
            confidence = 1.0 - max(abs(edge_on_buy), abs(edge_on_sell)) / std if std > 0 else 0.5

        # Check inventory limits
        if action == "BUY" and self.state.position >= self.MAX_POSITION:
            action = "PASS"
            reasoning = f"Would buy but at max long position ({self.MAX_POSITION})"

        elif action == "SELL" and self.state.position <= -self.MAX_POSITION:
            action = "PASS"
            reasoning = f"Would sell but at max short position ({-self.MAX_POSITION})"

        # Get recommended market (in case we need to make one)
        rec_bid, rec_ask = self.make_market()

        return MarketDecision(
            fair_value=fv,
            uncertainty=std,
            recommended_bid=rec_bid,
            recommended_ask=rec_ask,
            adjusted_bid=rec_bid,  # Already includes inventory adjustment
            adjusted_ask=rec_ask,
            action=action,
            confidence=confidence,
            edge_on_buy=edge_on_buy,
            edge_on_sell=edge_on_sell,
            reasoning=reasoning
        )

    def should_trade_with_joe(self, bid: float, ask: float) -> bool:
        """
        Should we trade with Joe (who is profitable 75% of the time)?

        Only trade with Joe if our edge is large enough to overcome
        the expected loss (Joe wins 75% = we lose 75%).
        """
        fv, std = self.calculate_fair_value()

        # Joe's expected edge against us is significant
        # We need massive edge to overcome this
        edge_on_buy = fv - ask
        edge_on_sell = bid - fv

        # Need at least 2x our normal threshold to trade with Joe
        joe_threshold = self.EDGE_THRESHOLD * 3

        return edge_on_buy > joe_threshold or edge_on_sell > joe_threshold


# =============================================================================
# BAYESIAN INFERENCE FROM TRADING BEHAVIOR
# =============================================================================

class BayesianInference:
    """
    Update fair value estimates based on observed trading behavior.

    Key insight: When someone makes a market or trades, they reveal
    information about their private cards.

    Example: If someone bids 10/asks 13, they likely believe FV is ~11.5
    This means they probably hold cards summing above average (1.2 per card).
    """

    def __init__(self, state: GameState):
        self.state = state
        self.estimator = ExpectedValueEstimator(state)

    def update_from_quote(self, bid: float, ask: float,
                          assumed_cards_held: int = 1) -> float:
        """
        Update fair value estimate based on observed market quote.

        Returns an adjusted fair value incorporating the information.
        """
        # Their implied fair value
        their_fv = (bid + ask) / 2

        # Our naive fair value
        our_fv = self.estimator.naive_expected_value()

        # Weight their information based on how many cards they hold
        # More cards = more information = higher weight
        their_weight = assumed_cards_held / (assumed_cards_held +
                                              self.state.hidden_table_count)
        our_weight = 1 - their_weight

        # Bayesian-like weighted average
        adjusted_fv = our_fv * our_weight + their_fv * their_weight

        return adjusted_fv

    def infer_sum_from_quote(self, bid: float, ask: float,
                             cards_held: int = 1) -> Tuple[float, float]:
        """
        Infer the likely sum of cards held by the quoter.

        Returns (inferred_sum, confidence).
        """
        their_fv = (bid + ask) / 2
        our_fv = self.estimator.naive_expected_value()

        # They have 'cards_held' cards that shift their FV
        # If their FV is higher, their cards are likely above average
        fv_diff = their_fv - our_fv

        # Average value per card in remaining deck
        remaining = self.state.get_remaining_deck()
        avg_per_card = sum(remaining) / len(remaining) if remaining else 1.2

        # Infer their card sum (relative to average)
        inferred_excess = fv_diff * (self.state.hidden_table_count / cards_held)
        inferred_sum = cards_held * avg_per_card + inferred_excess

        # Confidence based on spread (tighter spread = more confident)
        spread = ask - bid
        confidence = 1.0 / (1.0 + spread / 3)  # 3 is minimum spread

        return inferred_sum, confidence


# =============================================================================
# POSITION & PNL TRACKER
# =============================================================================

class PositionTracker:
    """Track positions and calculate PnL."""

    def __init__(self, state: GameState):
        self.state = state

    def execute_trade(self, price: float, quantity: int, is_buy: bool):
        """
        Record a trade execution.

        Args:
            price: Trade price
            quantity: Number of contracts (positive)
            is_buy: True if buying, False if selling
        """
        signed_qty = quantity if is_buy else -quantity

        # Update position
        old_position = self.state.position
        new_position = old_position + signed_qty

        # Update average price (for PnL calculation)
        if signed_qty > 0:  # Buying
            if old_position >= 0:
                # Adding to long or starting new long
                total_cost = self.state.avg_entry_price * old_position + price * signed_qty
                self.state.avg_entry_price = total_cost / new_position if new_position != 0 else 0
            else:
                # Covering shorts
                if new_position > 0:
                    self.state.avg_entry_price = price
                elif new_position == 0:
                    self.state.avg_entry_price = 0
        else:  # Selling
            if old_position <= 0:
                # Adding to short or starting new short
                total_cost = self.state.avg_entry_price * abs(old_position) + price * abs(signed_qty)
                self.state.avg_entry_price = total_cost / abs(new_position) if new_position != 0 else 0
            else:
                # Closing longs
                if new_position < 0:
                    self.state.avg_entry_price = price
                elif new_position == 0:
                    self.state.avg_entry_price = 0

        self.state.position = new_position

        # Record trade
        self.state.trades.append({
            'price': price,
            'quantity': signed_qty,
            'position_after': new_position,
            'round': self.state.current_round
        })

    def calculate_pnl(self, settlement_price: float) -> float:
        """
        Calculate total PnL given the final settlement price.

        PnL = sum of (settlement - trade_price) * signed_quantity for each trade
        """
        total_pnl = 0.0
        for trade in self.state.trades:
            trade_pnl = (settlement_price - trade['price']) * trade['quantity']
            total_pnl += trade_pnl
        return total_pnl

    def current_exposure(self, settlement_estimate: float) -> float:
        """Calculate current unrealized PnL based on estimated settlement."""
        if self.state.position == 0:
            return 0.0

        # Unrealized = (estimated_settlement - avg_entry) * position
        if self.state.position > 0:
            return (settlement_estimate - self.state.avg_entry_price) * self.state.position
        else:
            return (self.state.avg_entry_price - settlement_estimate) * abs(self.state.position)


# =============================================================================
# MAIN GAME INTERFACE
# =============================================================================

class MarketMakingGame:
    """
    Main interface for the market making game.

    Usage:
        game = MarketMakingGame()
        game.add_my_card(5)
        game.add_my_card(12)

        decision = game.get_decision(bid=8, ask=11)
        print(decision)

        game.reveal_table_cards([3, 7])
        decision = game.get_decision(bid=10, ask=13)
    """

    def __init__(self):
        self.state = GameState()
        self.market_maker = MarketMaker(self.state)
        self.position_tracker = PositionTracker(self.state)
        self.bayesian = BayesianInference(self.state)

    def add_my_card(self, card_value: int):
        """Add a private card to your hand."""
        if card_value not in FULL_DECK:
            raise ValueError(f"Invalid card value: {card_value}. Valid cards: {sorted(set(FULL_DECK))}")
        self.state.my_cards.append(card_value)
        # Update market maker instance
        self.market_maker = MarketMaker(self.state)
        self.bayesian = BayesianInference(self.state)

    def reveal_table_cards(self, cards: List[int]):
        """Reveal table cards (happens each round)."""
        for card in cards:
            if card not in FULL_DECK:
                raise ValueError(f"Invalid card value: {card}")
            self.state.revealed_table_cards.append(card)
            self.state.hidden_table_count -= 1
        # Update instances
        self.market_maker = MarketMaker(self.state)
        self.bayesian = BayesianInference(self.state)

    def advance_round(self):
        """Move to the next round."""
        self.state.current_round += 1

    def get_decision(self, bid: float = None, ask: float = None) -> MarketDecision:
        """
        Get trading decision.

        If bid/ask provided: evaluates whether to trade
        If not provided: returns recommended market to make
        """
        if bid is not None and ask is not None:
            return self.market_maker.evaluate_market(bid, ask)
        else:
            # Just return info for making a market
            fv, std = self.market_maker.calculate_fair_value()
            rec_bid, rec_ask = self.market_maker.make_market()
            return MarketDecision(
                fair_value=fv,
                uncertainty=std,
                recommended_bid=rec_bid,
                recommended_ask=rec_ask,
                adjusted_bid=rec_bid,
                adjusted_ask=rec_ask,
                action="MAKE_MARKET",
                confidence=0.0,
                reasoning=f"You should make market. FV={fv:.2f}, Std={std:.2f}"
            )

    def execute_buy(self, price: float, quantity: int = 1):
        """Execute a buy trade."""
        self.position_tracker.execute_trade(price, quantity, is_buy=True)

    def execute_sell(self, price: float, quantity: int = 1):
        """Execute a sell trade."""
        self.position_tracker.execute_trade(price, quantity, is_buy=False)

    def get_fair_value(self) -> float:
        """Get current fair value estimate."""
        return self.market_maker.calculate_fair_value()[0]

    def get_uncertainty(self) -> float:
        """Get current uncertainty (std dev)."""
        return self.market_maker.calculate_fair_value()[1]

    def get_position(self) -> int:
        """Get current position."""
        return self.state.position

    def calculate_final_pnl(self, final_table_sum: int) -> float:
        """Calculate final PnL given the revealed table sum."""
        return self.position_tracker.calculate_pnl(final_table_sum)

    def status(self) -> str:
        """Get current game status."""
        fv, std = self.market_maker.calculate_fair_value()
        rec_bid, rec_ask = self.market_maker.make_market()

        lines = [
            "=" * 50,
            "MARKET MAKING GAME STATUS",
            "=" * 50,
            f"Round: {self.state.current_round}",
            f"Your cards: {self.state.my_cards} (sum: {sum(self.state.my_cards)})",
            f"Revealed table cards: {self.state.revealed_table_cards} (sum: {sum(self.state.revealed_table_cards)})",
            f"Hidden table cards: {self.state.hidden_table_count}",
            "",
            f"FAIR VALUE: {fv:.2f}",
            f"UNCERTAINTY (std): {std:.2f}",
            f"95% CI: [{fv - 1.96*std:.2f}, {fv + 1.96*std:.2f}]",
            "",
            f"RECOMMENDED MARKET: {rec_bid:.1f} / {rec_ask:.1f}",
            "",
            f"POSITION: {self.state.position}",
            f"AVG ENTRY: {self.state.avg_entry_price:.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def print_status(self):
        """Print current game status."""
        print(self.status())


# =============================================================================
# QUICK CALCULATION FUNCTIONS
# =============================================================================

def quick_fair_value(my_cards: List[int], revealed_table: List[int] = None) -> float:
    """
    Quick calculation of fair value.

    Args:
        my_cards: List of your private card values
        revealed_table: List of revealed table card values (optional)

    Returns:
        Expected sum of the 6 table cards
    """
    if revealed_table is None:
        revealed_table = []

    known_sum = sum(my_cards) + sum(revealed_table)
    known_count = len(my_cards) + len(revealed_table)

    remaining_sum = DECK_SUM - known_sum
    remaining_count = DECK_SIZE - known_count

    hidden_table_count = 6 - len(revealed_table)

    ev_per_card = remaining_sum / remaining_count
    expected_hidden = hidden_table_count * ev_per_card

    return sum(revealed_table) + expected_hidden


def quick_decision(my_cards: List[int],
                   revealed_table: List[int],
                   bid: float,
                   ask: float) -> str:
    """
    Quick decision on whether to trade.

    Returns: "BUY", "SELL", or "PASS"
    """
    fv = quick_fair_value(my_cards, revealed_table)

    edge_buy = fv - ask
    edge_sell = bid - fv

    if edge_buy > 0.5:
        return f"BUY at {ask} (FV={fv:.2f}, edge={edge_buy:.2f})"
    elif edge_sell > 0.5:
        return f"SELL at {bid} (FV={fv:.2f}, edge={edge_sell:.2f})"
    else:
        return f"PASS (FV={fv:.2f}, no edge)"


def suggest_market(my_cards: List[int],
                   revealed_table: List[int] = None,
                   position: int = 0) -> Tuple[float, float]:
    """
    Suggest bid/ask when you need to make the market.

    Args:
        my_cards: Your private cards
        revealed_table: Revealed table cards
        position: Your current position (positive=long)

    Returns:
        (bid, ask) tuple
    """
    if revealed_table is None:
        revealed_table = []

    fv = quick_fair_value(my_cards, revealed_table)

    # Skew based on position
    skew = -position * 0.3

    bid = fv - 1.5 + skew
    ask = fv + 1.5 + skew

    # Round to 0.5
    bid = round(bid * 2) / 2
    ask = bid + 3  # Ensure exactly 3 wide

    return bid, ask


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MARKET MAKING GAME - DECISION ENGINE")
    print("=" * 60)
    print()

    # Example game
    game = MarketMakingGame()

    # Round 1: You receive 2 cards (you're the team making first market)
    print("ROUND 1: You receive cards 15 and 8")
    game.add_my_card(15)
    game.add_my_card(8)

    game.print_status()
    print()

    # You need to make a market
    decision = game.get_decision()
    print(f"DECISION: Make market at {decision.recommended_bid}/{decision.recommended_ask}")
    print()

    # Someone hits your ask (buys from you)
    print("Someone buys from you at your ask price...")
    game.execute_sell(decision.recommended_ask, 1)

    # Round 2: You get another card, 2 table cards revealed
    print("\nROUND 2: You receive card 3, table reveals [7, -10]")
    game.advance_round()
    game.add_my_card(3)
    game.reveal_table_cards([7, -10])

    game.print_status()
    print()

    # Another team makes market 4/7
    print("Another team quotes: 4 / 7")
    decision = game.get_decision(bid=4, ask=7)
    print(f"Action: {decision.action}")
    print(f"Reasoning: {decision.reasoning}")
    print()

    # Quick calculation demo
    print("=" * 60)
    print("QUICK CALCULATION DEMO")
    print("=" * 60)
    print()

    print("Your cards: [15, 8, 3]")
    print("Revealed table: [7, -10]")
    print()

    fv = quick_fair_value([15, 8, 3], [7, -10])
    print(f"Fair Value: {fv:.2f}")

    bid, ask = suggest_market([15, 8, 3], [7, -10], position=-1)
    print(f"Suggested Market (with short position): {bid}/{ask}")

    result = quick_decision([15, 8, 3], [7, -10], 4, 7)
    print(f"Decision on 4/7 market: {result}")
