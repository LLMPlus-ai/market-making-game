#!/usr/bin/env python3
"""
Interactive Market Making Game Interface
=========================================

Run this script during the game to get real-time trading decisions.

Usage:
    python game_interface.py

Then follow the prompts to:
1. Enter your private cards
2. Enter revealed table cards
3. Enter market quotes to evaluate
4. Get buy/sell/pass recommendations
"""

from market_maker import (
    MarketMakingGame, quick_fair_value, quick_decision,
    suggest_market, FULL_DECK, DECK_SUM
)
import sys


def print_header():
    print("\n" + "=" * 60)
    print("   MARKET MAKING GAME - TRADING DECISION SYSTEM")
    print("=" * 60)
    print(f"   Deck: 50 cards, Total Sum = {DECK_SUM}")
    print("   Valid cards: 0-20 (two each), -10 to -80 (one each)")
    print("=" * 60 + "\n")


def parse_cards(input_str: str) -> list:
    """Parse comma-separated card values."""
    if not input_str.strip():
        return []
    try:
        cards = [int(x.strip()) for x in input_str.split(',')]
        for c in cards:
            if c not in FULL_DECK:
                print(f"WARNING: {c} is not a valid card!")
        return cards
    except ValueError:
        print("Invalid input. Enter comma-separated integers (e.g., 5, 12, -10)")
        return []


def interactive_mode():
    """Run the interactive game interface."""
    print_header()

    game = MarketMakingGame()
    current_round = 1

    while True:
        print(f"\n{'='*50}")
        print(f"  ROUND {current_round}")
        print(f"{'='*50}")

        # Get current state
        print(f"\nCurrent State:")
        print(f"  Your cards: {game.state.my_cards} (sum: {sum(game.state.my_cards)})")
        print(f"  Table revealed: {game.state.revealed_table_cards} (sum: {sum(game.state.revealed_table_cards)})")
        print(f"  Hidden table cards: {game.state.hidden_table_count}")
        print(f"  Position: {game.state.position}")

        # Calculate fair value
        fv, std = game.market_maker.calculate_fair_value()
        print(f"\n  >>> FAIR VALUE: {fv:.2f} (± {std:.2f}) <<<")
        print(f"  >>> 95% CI: [{fv - 1.96*std:.2f}, {fv + 1.96*std:.2f}] <<<")

        # Show recommended market
        rec_bid, rec_ask = game.market_maker.make_market()
        print(f"\n  Recommended market: {rec_bid:.1f} / {rec_ask:.1f}")

        # Menu
        print("\n" + "-" * 40)
        print("Options:")
        print("  1. Add my private card(s)")
        print("  2. Reveal table card(s)")
        print("  3. Evaluate a market (should I trade?)")
        print("  4. I executed a trade")
        print("  5. Next round")
        print("  6. Calculate final PnL")
        print("  7. Reset game")
        print("  8. Quick calculator mode")
        print("  q. Quit")
        print("-" * 40)

        choice = input("\nChoice: ").strip().lower()

        if choice == '1':
            cards_str = input("Enter your card(s) (comma-separated): ")
            cards = parse_cards(cards_str)
            for c in cards:
                game.add_my_card(c)
            print(f"Added cards: {cards}")

        elif choice == '2':
            cards_str = input("Enter revealed table card(s) (comma-separated): ")
            cards = parse_cards(cards_str)
            game.reveal_table_cards(cards)
            print(f"Revealed: {cards}")

        elif choice == '3':
            try:
                bid = float(input("Enter BID price: "))
                ask = float(input("Enter ASK price: "))

                decision = game.get_decision(bid, ask)

                print("\n" + "=" * 40)
                print(f"  FAIR VALUE: {decision.fair_value:.2f}")
                print(f"  Market: {bid} / {ask}")
                print(f"  Mid: {(bid + ask) / 2}")
                print()
                print(f"  Edge on BUY: {decision.edge_on_buy:.2f}")
                print(f"  Edge on SELL: {decision.edge_on_sell:.2f}")
                print()

                if decision.action == "BUY":
                    print(f"  >>> RECOMMENDATION: BUY at {ask} <<<")
                    print(f"  >>> {decision.reasoning} <<<")
                elif decision.action == "SELL":
                    print(f"  >>> RECOMMENDATION: SELL at {bid} <<<")
                    print(f"  >>> {decision.reasoning} <<<")
                else:
                    print(f"  >>> RECOMMENDATION: PASS <<<")
                    print(f"  >>> {decision.reasoning} <<<")

                print(f"\n  Confidence: {decision.confidence:.1%}")
                print("=" * 40)

            except ValueError:
                print("Invalid price input")

        elif choice == '4':
            try:
                direction = input("Buy or Sell? (b/s): ").strip().lower()
                price = float(input("At what price? "))
                qty = int(input("Quantity [1]: ") or "1")

                if direction.startswith('b'):
                    game.execute_buy(price, qty)
                    print(f"Executed: BUY {qty} @ {price}")
                else:
                    game.execute_sell(price, qty)
                    print(f"Executed: SELL {qty} @ {price}")

                print(f"New position: {game.state.position}")

            except ValueError:
                print("Invalid input")

        elif choice == '5':
            current_round += 1
            game.advance_round()
            print(f"Advanced to Round {current_round}")

        elif choice == '6':
            try:
                final_sum = int(input("Enter final sum of 6 table cards: "))
                pnl = game.calculate_final_pnl(final_sum)
                print(f"\n{'='*40}")
                print(f"  FINAL TABLE SUM: {final_sum}")
                print(f"  YOUR P&L: {pnl:+.1f} points")
                print(f"{'='*40}")
            except ValueError:
                print("Invalid input")

        elif choice == '7':
            game = MarketMakingGame()
            current_round = 1
            print("Game reset!")

        elif choice == '8':
            quick_calculator_mode()

        elif choice == 'q':
            print("Goodbye!")
            break


def quick_calculator_mode():
    """Quick calculation mode - no state tracking."""
    print("\n" + "=" * 50)
    print("QUICK CALCULATOR MODE")
    print("(Enter 'back' to return to main menu)")
    print("=" * 50)

    while True:
        print("\n--- Quick Calculator ---")
        my_cards_str = input("Your cards (comma-separated): ").strip()
        if my_cards_str.lower() == 'back':
            return

        my_cards = parse_cards(my_cards_str)

        table_str = input("Revealed table cards (comma-separated, or empty): ").strip()
        revealed = parse_cards(table_str) if table_str else []

        fv = quick_fair_value(my_cards, revealed)
        hidden_count = 6 - len(revealed)

        print(f"\n>>> FAIR VALUE: {fv:.2f}")
        print(f">>> Hidden cards: {hidden_count}")

        position_str = input("Your position (0 if none): ").strip()
        try:
            position = int(position_str) if position_str else 0
        except ValueError:
            position = 0

        bid, ask = suggest_market(my_cards, revealed, position)
        print(f"\n>>> RECOMMENDED MARKET: {bid} / {ask}")

        eval_market = input("\nEvaluate a market? (y/n): ").strip().lower()
        if eval_market == 'y':
            try:
                bid = float(input("  Bid: "))
                ask = float(input("  Ask: "))
                result = quick_decision(my_cards, revealed, bid, ask)
                print(f"\n>>> {result}")
            except ValueError:
                print("Invalid input")


def one_shot_mode(my_cards: list, revealed: list = None, bid: float = None, ask: float = None):
    """
    One-shot calculation for scripting.

    Example:
        python -c "from game_interface import one_shot_mode; one_shot_mode([5, 12], [3, 7], 8, 11)"
    """
    if revealed is None:
        revealed = []

    fv = quick_fair_value(my_cards, revealed)
    print(f"Fair Value: {fv:.2f}")

    rec_bid, rec_ask = suggest_market(my_cards, revealed)
    print(f"Recommended Market: {rec_bid} / {rec_ask}")

    if bid is not None and ask is not None:
        result = quick_decision(my_cards, revealed, bid, ask)
        print(f"Decision: {result}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode
        # Example: python game_interface.py 5,12 3,7 8 11
        my_cards = parse_cards(sys.argv[1])
        revealed = parse_cards(sys.argv[2]) if len(sys.argv) > 2 else []
        bid = float(sys.argv[3]) if len(sys.argv) > 3 else None
        ask = float(sys.argv[4]) if len(sys.argv) > 4 else None
        one_shot_mode(my_cards, revealed, bid, ask)
    else:
        interactive_mode()
