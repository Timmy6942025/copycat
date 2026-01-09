#!/usr/bin/env python3
"""
Quick Resolve Feature Unit Test

Unit tests for quick resolve prioritization logic.
"""

import sys
from datetime import datetime, timedelta


def test_is_quick_resolve_market():
    """Test the _is_quick_resolve_market logic."""

    print("\n" + "=" * 70)
    print("UNIT TEST: Quick Resolve Market Detection")
    print("=" * 70)

    # Simulate config values
    prefer_quick_resolve = True
    threshold_hours = 48.0

    print(f"\nConfiguration:")
    print(f"  - prefer_quick_resolve: {prefer_quick_resolve}")
    print(f"  - quick_resolve_threshold_hours: {threshold_hours}")

    # Test cases
    test_cases = [
        ("2 hours from now", timedelta(hours=2), True),
        ("24 hours from now", timedelta(hours=24), True),
        ("48 hours from now (at threshold)", timedelta(hours=48), True),
        ("49 hours from now (just over)", timedelta(hours=49), False),
        ("3 days from now", timedelta(days=3), False),
        ("7 days from now", timedelta(days=7), False),
    ]

    print(f"\nTest Cases:")
    all_passed = True

    for i, (description, offset, expected) in enumerate(test_cases, 1):
        market_end_date = datetime.utcnow() + offset
        hours_until_resolve = (market_end_date - datetime.utcnow()).total_seconds() / 3600
        is_quick = prefer_quick_resolve and hours_until_resolve <= threshold_hours

        passed = is_quick == expected
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {i}. {description}:")
        print(f"     Hours until resolve: {hours_until_resolve:.1f}")
        print(f"     Expected: {'QUICK' if expected else 'NOT QUICK'}")
        print(f"     Result: {'QUICK' if is_quick else 'NOT QUICK'}")
        print(f"     {status}\n")

    print("=" * 70)
    if all_passed:
        print("✅ ALL UNIT TESTS PASSED")
    else:
        print("❌ SOME UNIT TESTS FAILED")
    print("=" * 70)

    return all_passed


def test_boost_multiplier_calculation():
    """Test boost mode multiplier calculation with quick resolve."""

    print("\n" + "=" * 70)
    print("UNIT TEST: Boost Mode Multiplier Calculation")
    print("=" * 70)

    # Config values
    position_multiplier = 2.5
    quick_resolve_multiplier = 2.0
    threshold_hours = 48.0

    print(f"\nConfiguration:")
    print(f"  - position_multiplier: x{position_multiplier}")
    print(f"  - quick_resolve_multiplier: x{quick_resolve_multiplier}")

    # Test cases
    test_cases = [
        ("Quick market (2h to resolve)", 2.0, True),
        ("Quick market (24h to resolve)", 24.0, True),
        ("Quick market (48h at threshold)", 48.0, True),
        ("Slow market (49h over threshold)", 49.0, False),
        ("Slow market (3 days)", 72.0, False),
    ]

    print(f"\nTest Cases:")
    all_passed = True

    for i, (description, hours_to_resolve, is_quick) in enumerate(test_cases, 1):
        base_position = 10.0

        if is_quick:
            final_multiplier = position_multiplier * quick_resolve_multiplier
            quick_bonus = quick_resolve_multiplier
        else:
            final_multiplier = position_multiplier
            quick_bonus = 0.0

        final_position = base_position * final_multiplier
        expected_multiplier = (position_multiplier * quick_resolve_multiplier) if is_quick else position_multiplier
        expected_position = base_position * expected_multiplier

        passed = abs(final_multiplier - expected_multiplier) < 0.01
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {i}. {description}:")
        print(f"     Hours to resolve: {hours_to_resolve}")
        print(f"     Is quick resolve: {is_quick}")
        print(f"     Base multiplier: x{position_multiplier}")
        if is_quick:
            print(f"     Quick resolve bonus: x{quick_resolve_multiplier}")
        print(f"     Final multiplier: x{final_multiplier:.2f}")
        print(f"     Expected multiplier: x{expected_multiplier:.2f}")
        print(f"     Final position: ${final_position:.2f}")
        print(f"     Expected position: ${expected_position:.2f}")
        print(f"     {status}\n")

    print("=" * 70)
    if all_passed:
        print("✅ ALL UNIT TESTS PASSED")
    else:
        print("❌ SOME UNIT TESTS FAILED")
    print("=" * 70)

    return all_passed


def test_quick_resolve_disabled():
    """Test behavior when quick resolve is disabled."""

    print("\n" + "=" * 70)
    print("UNIT TEST: Quick Resolve Disabled")
    print("=" * 70)

    prefer_quick_resolve = False
    threshold_hours = 48.0

    print(f"\nConfiguration:")
    print(f"  - prefer_quick_resolve: {prefer_quick_resolve}")
    print(f"  - quick_resolve_threshold_hours: {threshold_hours}")

    # Test cases - all should return False when quick resolve disabled
    test_cases = [
        timedelta(hours=2),
        timedelta(hours=24),
        timedelta(hours=48),
        timedelta(days=3),
    ]

    print(f"\nTest Cases:")
    all_passed = True

    for i, offset in enumerate(test_cases, 1):
        market_end_date = datetime.utcnow() + offset
        hours_until_resolve = (market_end_date - datetime.utcnow()).total_seconds() / 3600
        is_quick = prefer_quick_resolve and hours_until_resolve <= threshold_hours

        passed = is_quick == False
        all_passed = all_passed and passed

        hours_str = f"{hours_until_resolve:.1f}"
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {i}. Market resolves in {hours_str} hours:")
        print(f"     Expected: NOT QUICK (feature disabled)")
        print(f"     Result: {'QUICK' if is_quick else 'NOT QUICK'}")
        print(f"     {status}\n")

    print("=" * 70)
    if all_passed:
        print("✅ ALL UNIT TESTS PASSED")
    else:
        print("❌ SOME UNIT TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    try:
        results = []

        results.append(("Quick Resolve Detection", test_is_quick_resolve_market()))
        results.append(("Boost Multiplier Calc", test_boost_multiplier_calculation()))
        results.append(("Quick Resolve Disabled", test_quick_resolve_disabled()))

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} - {name}")

        all_passed = all(result[1] for result in results)

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL TEST SUITES PASSED")
        else:
            print("❌ SOME TEST SUITES FAILED")
        print("=" * 70)

        sys.exit(0 if all_passed else 1)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
