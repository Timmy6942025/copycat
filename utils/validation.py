"""
Data Validation Utilities for CopyCat Trading System.

Provides consistent validation for numeric values, addresses, and API responses.
"""

from typing import Optional, Union
from decimal import Decimal, InvalidOperation
import logging


logger = logging.getLogger(__name__)


# =============================================================================
# NUMERIC VALIDATION
# =============================================================================


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Value to return if denominator is zero (default: 0.0)

    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def safe_divide_inf(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    Returns float('inf') only if explicitly requested.

    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Value to return if denominator is zero (default: 0.0)

    Returns:
        Division result or default value (never returns inf)
    """
    if denominator == 0:
        return default
    return numerator / denominator


def validate_positive_number(
    value: Union[float, int], name: str, allow_zero: bool = False
) -> tuple[bool, str]:
    """
    Validate that a value is a positive number.

    Args:
        value: The value to validate
        name: Name of the parameter for error messages
        allow_zero: Whether zero is allowed (default: False)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None:
        return False, f"{name} is required"

    if not isinstance(value, (int, float)):
        return False, f"{name} must be a number"

    if allow_zero:
        if value < 0:
            return False, f"{name} cannot be negative"
    else:
        if value <= 0:
            return False, f"{name} must be greater than zero"

    return True, ""


def validate_price(price: Optional[float], name: str = "price") -> tuple[bool, str]:
    """
    Validate a price value.

    Args:
        price: The price to validate
        name: Name of the parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if price is None:
        return False, f"{name} is required"

    if not isinstance(price, (int, float)):
        return False, f"{name} must be a number"

    if price < 0:
        return False, f"{name} cannot be negative"

    if price > 1.0:
        logger.warning(
            f"{name} {price:.4f} exceeds typical prediction market range (0-1)"
        )

    return True, ""


def validate_balance(
    balance: Optional[float], name: str = "balance"
) -> tuple[bool, str]:
    """
    Validate a balance value.

    Args:
        balance: The balance to validate
        name: Name of the parameter for error:
        Tuple of (is_valid, messages

    Returns error_message)
    """
    if balance is None:
        return False, f"{name} is required"

    if not isinstance(balance, (int, float)):
        return False, f"{name} must be a number"

    if balance < 0:
        return False, f"{name} cannot be negative"

    return True, ""


def validate_percentage(
    value: Optional[float], name: str = "percentage"
) -> tuple[bool, str]:
    """
    Validate a percentage value (0 to 1).

    Args:
        value: The percentage value to validate
        name: Name of the parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None:
        return False, f"{name} is required"

    if not isinstance(value, (int, float)):
        return False, f"{name} must be a number"

    if value < 0:
        return False, f"{name} cannot be negative"

    if value > 1.0:
        return False, f"{name} cannot exceed 1.0 (100%)"

    return True, ""


# =============================================================================
# ADDRESS VALIDATION
# =============================================================================


def validate_eth_address(
    address: Optional[str], name: str = "address"
) -> tuple[bool, str]:
    """
    Validate an Ethereum-style address.

    Args:
        address: The address to validate
        name: Name of the parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not address:
        return False, f"{name} is required"

    address = address.strip()

    if len(address) != 42:
        return (
            False,
            f"Invalid {name} length (expected 42 characters, got {len(address)})",
        )

    if not address.startswith("0x"):
        return False, f"{name} must start with 0x"

    # Check hex characters
    try:
        int(address[2:], 16)
    except ValueError:
        return False, f"{name} contains invalid hex characters"

    return True, ""


# =============================================================================
# API RESPONSE VALIDATION
# =============================================================================


def validate_api_response(response: Optional[dict], endpoint: str) -> tuple[bool, str]:
    """
    Validate API response structure.

    Args:
        response: The API response to validate
        endpoint: Name of the endpoint for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if response is None:
        return False, f"Empty response from {endpoint}"

    if not isinstance(response, dict):
        return (
            False,
            f"Invalid response type from {endpoint} (expected dict, got {type(response).__name__})",
        )

    return True, ""


def get_with_default(data: dict, key: str, default: any = None) -> any:
    """
    Safely get a value from a dictionary with a default.

    Args:
        data: The dictionary to search
        key: The key to look for
        default: Default value if key not found

    Returns:
        The value or default
    """
    if not isinstance(data, dict):
        return default

    value = data.get(key)
    if value is None:
        return default

    return value


# =============================================================================
# FINANCIAL CALCULATIONS
# =============================================================================


def calculate_win_rate(winning_trades: int, total_trades: int) -> float:
    """
    Calculate win rate with division by zero protection.

    Args:
        winning_trades: Number of winning trades
        total_trades: Total number of trades

    Returns:
        Win rate as a decimal (0 to 1), or 0 if no trades
    """
    if total_trades <= 0:
        return 0.0
    return winning_trades / total_trades


def calculate_profit_factor(gross_profits: float, gross_losses: float) -> float:
    """
    Calculate profit factor with division by zero protection.

    Args:
        gross_profits: Total profit from winning trades
        gross_losses: Total loss from losing trades

    Returns:
        Profit factor, or 0 if gross_losses is zero or negative
    """
    if gross_losses <= 0:
        return 0.0  # Return 0 instead of inf for safety
    return gross_profits / gross_losses


def calculate_kelly(
    win_rate: float, profit_loss_ratio: float, kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion position size.

    Args:
        win_rate: Probability of winning (0 to 1)
        profit_loss_ratio: Ratio of average win to average loss
        kelly_fraction: Fraction of Kelly to use (default: 0.25 for safety)

    Returns:
        Kelly fraction (0 to 1), or 0 if profit_loss_ratio <= 0
    """
    if profit_loss_ratio <= 0:
        return 0.0

    kelly = ((profit_loss_ratio * win_rate) - (1 - win_rate)) / profit_loss_ratio
    kelly = max(0, kelly) * kelly_fraction

    return kelly


def calculate_roi(profit: float, cost: float) -> float:
    """
    Calculate Return on Investment.

    Args:
        profit: Profit amount
        cost: Initial cost

    Returns:
        ROI as decimal, or 0 if cost is zero
    """
    if cost == 0:
        return 0.0
    return profit / cost


# =============================================================================
# DECIMAL UTILITIES
# =============================================================================


def to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
    """
    Convert a value to Decimal for precise financial calculations.

    Args:
        value: The value to convert

    Returns:
        Decimal representation
    """
    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as e:
        logger.warning(f"Failed to convert {value} to Decimal: {e}")
        return Decimal("0")


def safe_decimal_operation(
    a: Union[float, int, Decimal], b: Union[float, int, Decimal], operation: str = "add"
) -> Decimal:
    """
    Safely perform a decimal operation.

    Args:
        a: First operand
        b: Second operand
        operation: 'add', 'subtract', 'multiply', 'divide'

    Returns:
        Result as Decimal
    """
    a_dec = to_decimal(a)
    b_dec = to_decimal(b)

    if operation == "add":
        return a_dec + b_dec
    elif operation == "subtract":
        return a_dec - b_dec
    elif operation == "multiply":
        return a_dec * b_dec
    elif operation == "divide":
        if b_dec == 0:
            return Decimal("0")
        return a_dec / b_dec
    else:
        raise ValueError(f"Unknown operation: {operation}")
