---

## 12. Sandbox/Paper Trading Simulation Module

### 12.1 Overview and Purpose

The Sandbox Module provides a fully isolated, risk-free environment for testing and validating the trading bot's core logic before any real funds are deployed. It uses real-time market data while simulating all trading operations with virtual money.

**Primary Objectives**:
- **Development Testing**: Validate trader selection, copy trading, and risk management logic without financial risk
- **Algorithm Validation**: Verify that the bot can consistently identify profitable traders and generate positive returns
- **Performance Benchmarking**: Establish expected ROI, win rates, and risk metrics before live deployment
- **Stress Testing**: Run the system through various market conditions (bull, bear, volatile, sideways)
- **Parameter Tuning**: Optimize copy trading parameters (position sizing, confidence thresholds, etc.)

**Key Principles**:
- All trading logic executes identically to production
- Real-time data feeds are used for accurate simulation
- Virtual portfolio mirrors realistic trading constraints
- Complete audit trail and performance analytics

---

### 12.2 Sandbox Architecture

#### 12.2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SANDBOX SIMULATION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐│
│  │   Simulation       │    │   Virtual          │    │   Performance      ││
│  │   Engine           │◄──►│   Portfolio        │◄──►│   Tracker          ││
│  │   (Trade Sim)      │    │   Manager          │    │   & Analytics      ││
│  └─────────┬──────────┘    └─────────┬──────────┘    └─────────┬──────────┘│
│            │                         │                         │            │
│            │    ┌────────────────────┴────────────────────┐    │            │
│            └───►│       Simulation State Manager          │◄───┘            │
│                 │    (Mode: SANDBOX | LIVE)              │                 │
│                 └────────────────────┬────────────────────┘                 │
│                                      │                                      │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA & EXECUTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐│
│  │   Real-Time        │    │   Order            │    │   Market Data      ││
│  │   Market Data      │    │   Execution        │    │   (Read-Only)      ││
│  │   (Live Feeds)     │    │   (Simulated)      │    │                    ││
│  └────────────────────┘    └────────────────────┘    └────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 12.2.2 Mode Configuration

```python
@dataclass
class SandboxConfig:
    """Configuration for sandbox simulation mode."""

    # Mode Selection
    mode: str = "SANDBOX"  # "SANDBOX" or "LIVE"

    # Virtual Portfolio Settings
    initial_balance: float = 10000.0  # Starting USD balance
    max_portfolio_value: Optional[float] = None  # Cap at specific value

    # Simulation Constraints (mimic real trading limits)
    max_orders_per_day: int = 50
    max_position_size_pct: float = 0.10  # 10% max per single trade
    max_total_exposure_pct: float = 0.50  # 50% max total exposure
    min_order_size: float = 1.0  # Minimum $1 per trade

    # Execution Simulation
    simulate_slippage: bool = True
    slippage_model: str = "volume_weighted"  # "fixed", "volume_weighted", "realistic"
    simulate_fees: bool = True
    fee_model: str = "polymarket"  # Platform-specific fee structures
    simulate_fill_probability: bool = True
    fill_probability_model: str = "orderbook_depth"

    # Data Configuration
    use_historical_data: bool = False  # If True, run backtesting instead
    historical_data_range: Optional[Tuple[datetime, datetime]] = None
    replay_speed: float = 1.0  # 1.0 = real-time, >1 = faster playback

    # Persistence
    persist_results: bool = True
    results_storage_path: str = "./sandbox_results"
    auto_save_interval_seconds: int = 60

    # Logging & Analytics
    log_all_trades: bool = True
    generate_performance_reports: bool = True
    report_interval_hours: int = 24
    alert_on_thresholds: bool = True
    profit_alert_threshold: float = 0.05  # Alert at 5% profit
    loss_alert_threshold: float = -0.05  # Alert at 5% loss


@dataclass
class SimulationState:
    """Current state of sandbox simulation."""

    # Portfolio State
    balance: float
    positions: Dict[str, VirtualPosition]
    pending_orders: List[VirtualOrder]
    total_exposure: float
    total_value: float  # balance + positions value

    # Performance Metrics
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float

    # Trade History
    completed_trades: List[VirtualTrade]
    rejected_orders: List[RejectedOrder]
    order_fill_history: List[OrderFillRecord]

    # Runtime State
    start_time: datetime
    last_update: datetime
    simulation_days: int
    is_paused: bool
    is_completed: bool
```

---

### 12.3 Simulation Engine

#### 12.3.1 Virtual Order Execution

```python
class VirtualOrderExecutor:
    """
    Simulates order execution with realistic slippage, fees, and fill probability.
    Mimics actual exchange behavior without using real funds.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.orderbook_cache: Dict[str, OrderBook] = {}
        self.fill_probability_cache: Dict[str, float] = {}

    async def execute_virtual_order(
        self,
        order: VirtualOrder,
        current_market_data: MarketData
    ) -> VirtualOrderResult:
        """
        Execute a virtual order with realistic simulation.

        Process:
        1. Validate order against sandbox constraints
        2. Calculate realistic fill probability
        3. Simulate slippage based on order size and market liquidity
        4. Apply platform fees
        5. Determine fill price and quantity
        6. Record execution details
        """
        # Step 1: Constraint Validation
        validation_result = self._validate_order_constraints(order)
        if not validation_result.is_valid:
            return VirtualOrderResult(
                order_id=order.order_id,
                status="REJECTED",
                rejection_reason=validation_result.reason,
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                slippage=0,
                timestamp=datetime.utcnow()
            )

        # Step 2: Get Market Data
        orderbook = self._get_orderbook(order.market_id)
        if orderbook is None:
            return VirtualOrderResult(
                order_id=order.order_id,
                status="FAILED",
                rejection_reason="No market data available",
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                slippage=0,
                timestamp=datetime.utcnow()
            )

        # Step 3: Calculate Fill Probability
        fill_probability = self._calculate_fill_probability(
            order=order,
            orderbook=orderbook,
            market_data=current_market_data
        )

        # Step 4: Determine Execution Outcome
        if random.random() > fill_probability:
            return VirtualOrderResult(
                order_id=order.order_id,
                status="PARTIAL_FILL" if random.random() > 0.5 else "NO_FILL",
                rejection_reason="Order did not fill (simulated)",
                filled_quantity=0,
                average_price=0,
                total_fees=0,
                slippage=0,
                timestamp=datetime.utcnow()
            )

        # Step 5: Calculate Execution Price with Slippage
        execution_price = self._calculate_execution_price(
            order=order,
            orderbook=orderbook,
            side=order.side
        )

        # Step 6: Apply Fees
        fees = self._calculate_fees(
            order=order,
            execution_price=execution_price
        )

        # Step 7: Calculate Slippage
        slippage = self._calculate_slippage(
            order=order,
            reference_price=current_market_data.current_price,
            execution_price=execution_price
        )

        # Step 8: Determine Final Fill Quantity
        fill_quantity = self._calculate_fill_quantity(
            order=order,
            orderbook=orderbook,
            side=order.side
        )

        return VirtualOrderResult(
            order_id=order.order_id,
            status="FILLED",
            filled_quantity=fill_quantity,
            average_price=execution_price,
            total_fees=fees,
            slippage=slippage,
            timestamp=datetime.utcnow()
        )

    def _calculate_fill_probability(
        self,
        order: VirtualOrder,
        orderbook: OrderBook,
        market_data: MarketData
    ) -> float:
        """
        Calculate probability of order filling based on market conditions.

        Factors:
        - Order size vs available liquidity at price levels
        - Market volatility
        - Time priority (simulated)
        - Order type (market vs limit)
        """
        # Get liquidity at or better than limit price
        if order.order_type == "market":
            # Market orders fill immediately but may suffer slippage
            liquidity = self._get_available_liquidity(orderbook, order.side)
            fill_prob = min(1.0, liquidity / order.quantity)
        else:
            # Limit orders: fill probability based on how "aggressive" the price is
            limit_price = order.limit_price
            best_price = self._get_best_price(orderbook, order.side)

            if order.side == "buy":
                # Buy limit: price must be at or below limit to fill
                if limit_price >= best_price:
                    # Aggressive: high fill probability
                    fill_prob = self._estimate_limit_fill_probability(
                        orderbook, order, distance_from_best=limit_price - best_price
                    )
                else:
                    # Passive: lower fill probability
                    fill_prob = 0.3  # 30% chance of fill for passive limit
            else:
                # Sell limit: price must be at or above limit to fill
                if limit_price <= best_price:
                    fill_prob = self._estimate_limit_fill_probability(
                        orderbook, order, distance_from_best=best_price - limit_price
                    )
                else:
                    fill_prob = 0.3

        # Adjust for volatility
        volatility_factor = min(1.0, market_data.volatility / 0.5)  # Cap at 50% volatility
        fill_prob *= (1 - volatility_factor * 0.2)  # Reduce fill probability in high volatility

        return max(0.0, min(1.0, fill_prob))

    def _calculate_execution_price(
        self,
        order: VirtualOrder,
        orderbook: OrderBook,
        side: str
    ) -> float:
        """
        Calculate realistic execution price with slippage.

        Uses volume-weighted average price (VWAP) based on order size
        and available liquidity at different price levels.
        """
        if not self.config.simulate_slippage:
            return self._get_best_price(orderbook, side)

        # Get price levels from orderbook
        if side == "buy":
            asks = sorted(orderbook.asks, key=lambda x: x.price)
            price_levels = [(a.price, a.size) for a in asks]
        else:
            bids = sorted(orderbook.bids, key=lambda x: -x.price)
            price_levels = [(b.price, b.size) for b in bids]

        # Calculate VWAP based on order size
        remaining_qty = order.quantity
        total_cost = 0
        filled_qty = 0

        for price, size in price_levels:
            if remaining_qty <= 0:
                break

            fill_qty = min(remaining_qty, size)
            total_cost += fill_qty * price
            filled_qty += fill_qty
            remaining_qty -= fill_qty

        # If order exceeds available liquidity, use last price with penalty
        if filled_qty > 0:
            vwap = total_cost / filled_qty

            # Apply additional slippage for large orders
            size_penalty = min(0.05, (order.quantity / filled_qty - 1) * 0.01)
            if side == "buy":
                vwap *= (1 + size_penalty)
            else:
                vwap *= (1 - size_penalty)

            return vwap
        else:
            # No liquidity: use last resort price with large slippage
            return orderbook.last_trade_price * (1.1 if side == "buy" else 0.9)

    def _calculate_fees(
        self,
        order: VirtualOrder,
        execution_price: float
    ) -> float:
        """Calculate platform fees based on simulated fee model."""
        if not self.config.simulate_fees:
            return 0.0

        notional_value = order.quantity * execution_price

        if self.config.fee_model == "polymarket":
            # Polymarket: ~0.5% to 2% depending on volume and market
            base_fee_rate = 0.02  # 2% maximum
            volume_discount = min(0.015, order.trader_total_volume_30d / 100000)
            fee_rate = base_fee_rate - volume_discount
        elif self.config.fee_model == "kalshi":
            # Kalshi: ~2% flat fee
            fee_rate = 0.02
        else:
            fee_rate = 0.01  # Default 1%

        return notional_value * fee_rate

    def _calculate_slippage(
        self,
        order: VirtualOrder,
        reference_price: float,
        execution_price: float
    ) -> float:
        """Calculate slippage as percentage difference from reference price."""
        if reference_price == 0:
            return 0.0

        if order.side == "buy":
            return (execution_price - reference_price) / reference_price
        else:
            return (reference_price - execution_price) / reference_price
```

#### 12.3.2 Virtual Portfolio Manager

```python
class VirtualPortfolioManager:
    """
    Manages the virtual portfolio for sandbox simulation.
    Tracks positions, P&L, and risk metrics.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.positions: Dict[str, VirtualPosition] = {}
        self.balance = config.initial_balance
        self.pending_orders: List[VirtualOrder] = []

    async def execute_copy_trade(
        self,
        source_trade: TradeRecord,
        trader_config: TraderCopyConfig
    ) -> VirtualTradeResult:
        """
        Execute a copy trade in virtual portfolio.

        Process:
        1. Calculate position size based on trader config
        2. Create virtual order
        3. Execute order through virtual executor
        4. Update portfolio state
        5. Record trade for analytics
        """
        # Step 1: Calculate Position Size
        position_size = self._calculate_position_size(
            source_trade=source_trade,
            trader_config=trader_config
        )

        # Step 2: Validate Against Constraints
        validation = self._validate_trade_constraints(position_size)
        if not validation.is_valid:
            return VirtualTradeResult(
                status="REJECTED",
                reason=validation.reason,
                source_trade_id=source_trade.trade_id,
                position_size=0
            )

        # Step 3: Create Virtual Order
        virtual_order = VirtualOrder(
            order_id=f"sim_{source_trade.trade_id}_{uuid.uuid4().hex[:8]}",
            market_id=source_trade.market_id,
            side="buy" if source_trade.outcome == "YES" else "sell",
            quantity=position_size,
            order_type="market",  # Copy trades use market for immediate execution
            limit_price=None,
            timestamp=datetime.utcnow(),
            source_trade_id=source_trade.trade_id,
            source_trader=source_trade.trader_address
        )

        # Step 4: Execute Order
        market_data = await self._get_market_data(source_trade.market_id)
        executor = VirtualOrderExecutor(self.config)
        exec_result = await executor.execute_virtual_order(virtual_order, market_data)

        # Step 5: Update Portfolio
        if exec_result.status == "FILLED":
            self._update_portfolio(exec_result, source_trade)
            self._record_trade(exec_result, source_trade)

        return VirtualTradeResult(
            status=exec_result.status,
            reason=exec_result.rejection_reason if exec_result.status != "FILLED" else None,
            source_trade_id=source_trade.trade_id,
            position_size=exec_result.filled_quantity,
            execution_price=exec_result.average_price,
            fees=exec_result.total_fees,
            slippage=exec_result.slippage,
            timestamp=exec_result.timestamp
        )

    def _calculate_position_size(
        self,
        source_trade: TradeRecord,
        trader_config: TraderCopyConfig
    ) -> float:
        """
        Calculate virtual position size for copy trade.

        Methods:
        - Fixed Amount: Fixed $ amount per trade
        - Percentage of Portfolio: % of total portfolio value
        - Scaled by Trader Confidence: Size scales with trader score
        - Kelly Criterion: Optimal size based on edge and odds
        """
        if trader_config.position_sizing_method == "fixed_amount":
            return trader_config.base_position_size
        elif trader_config.position_sizing_method == "percentage":
            portfolio_value = self._calculate_portfolio_value()
            return portfolio_value * trader_config.position_size_pct
        elif trader_config.position_sizing_method == "scaled":
            portfolio_value = self._calculate_portfolio_value()
            base_size = portfolio_value * trader_config.position_size_pct
            # Scale by trader confidence (0.5 to 1.0)
            confidence_multiplier = 0.5 + (trader_config.trader_score / 2)
            return base_size * confidence_multiplier
        elif trader_config.position_sizing_method == "kelly":
            # Kelly Criterion: f* = (bp - q) / b
            # Where: b = odds, p = win probability, q = loss probability
            win_rate = trader_config.estimated_win_rate
            profit_loss_ratio = trader_config.estimated_profit_loss_ratio
            kelly_fraction = trader_config.kelly_fraction  # Typically 0.25-0.5 for safety

            if profit_loss_ratio <= 0:
                return 0

            kelly = ((profit_loss_ratio * win_rate) - (1 - win_rate)) / profit_loss_ratio
            kelly = max(0, kelly) * kelly_fraction  # Apply fractional Kelly

            portfolio_value = self._calculate_portfolio_value()
            return portfolio_value * kelly
        else:
            return trader_config.base_position_size

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value (balance + positions)."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.balance + positions_value

    def _validate_trade_constraints(
        self,
        position_size: float
    ) -> ConstraintValidation:
        """Validate trade against sandbox constraints."""
        portfolio_value = self._calculate_portfolio_value()

        # Check minimum order size
        if position_size < self.config.min_order_size:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Order size ${position_size:.2f} below minimum ${self.config.min_order_size:.2f}"
            )

        # Check max position size
        max_position_value = portfolio_value * self.config.max_position_size_pct
        if position_size > max_position_value:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Position size ${position_size:.2f} exceeds max ${max_position_value:.2f} ({self.config.max_position_size_pct:.0%})"
            )

        # Check total exposure
        current_exposure = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        total_exposure_after = current_exposure + position_size
        max_exposure = portfolio_value * self.config.max_total_exposure_pct
        if total_exposure_after > max_exposure:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Total exposure ${total_exposure_after:.2f} would exceed max ${max_exposure:.2f} ({self.config.max_total_exposure_pct:.0%})"
            )

        # Check balance
        if position_size > self.balance:
            return ConstraintValidation(
                is_valid=False,
                reason=f"Insufficient balance: ${self.balance:.2f} < ${position_size:.2f}"
            )

        return ConstraintValidation(is_valid=True)

    def _update_portfolio(
        self,
        exec_result: VirtualOrderResult,
        source_trade: TradeRecord
    ):
        """Update portfolio with executed trade."""
        market_id = source_trade.market_id

        # Deduct balance for buy or add for sell
        total_cost = exec_result.filled_quantity * exec_result.average_price + exec_result.total_fees

        if source_trade.outcome == "YES":
            # Buying YES outcome
            self.balance -= total_cost

            if market_id in self.positions:
                position = self.positions[market_id]
                # Average in
                total_qty = position.quantity + exec_result.filled_quantity
                avg_price = (
                    (position.quantity * position.avg_price) +
                    (exec_result.filled_quantity * exec_result.average_price)
                ) / total_qty
                position.quantity = total_qty
                position.avg_price = avg_price
            else:
                self.positions[market_id] = VirtualPosition(
                    market_id=market_id,
                    outcome="YES",
                    quantity=exec_result.filled_quantity,
                    avg_price=exec_result.average_price,
                    current_price=exec_result.average_price,
                    unrealized_pnl=0,
                    timestamp=datetime.utcnow()
                )
        else:
            # Selling YES (buying NO)
            # For simplicity, track YES positions only; NO is inverse
            self.balance += total_cost  # Closing or reducing YES position

        # Update position prices
        self._update_position_prices()

    def update_market_prices(self, market_updates: Dict[str, float]):
        """Update current prices for all positions."""
        for market_id, new_price in market_updates.items():
            if market_id in self.positions:
                position = self.positions[market_id]
                position.current_price = new_price
                position.unrealized_pnl = (
                    (new_price - position.avg_price) * position.quantity
                    if position.outcome == "YES"
                    else ((position.avg_price - new_price) * position.quantity)
                )

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Generate portfolio summary with all metrics."""
        positions_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl
            for pos in self.positions.values()
        )

        return PortfolioSummary(
            balance=self.balance,
            positions_value=positions_value,
            total_value=self.balance + positions_value,
            unrealized_pnl=unrealized_pnl,
            position_count=len(self.positions),
            exposure_pct=positions_value / (self.balance + positions_value) if (self.balance + positions_value) > 0 else 0
        )
```

---

### 12.4 Performance Tracking & Analytics

#### 12.4.1 Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for sandbox simulation."""

    # Time Period
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Portfolio Performance
    starting_balance: float
    ending_balance: float
    total_pnl: float
    total_pnl_pct: float
    annualized_return: float

    # Win/Loss Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    profit_factor: float  # Gross profits / Gross losses

    # Risk Metrics
    volatility: float  # Standard deviation of returns
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float

    # Position Statistics
    avg_position_size: float
    max_position_size: float
    avg_hold_time_hours: float
    max_hold_time_hours: float
    min_hold_time_hours: float

    # Trader Copy Performance
    traders_copied: int
    profitable_traders: int
    top_performing_trader: str
    worst_performing_trader: str
    trader_specific_returns: Dict[str, float]

    # Execution Quality
    avg_slippage: float
    total_fees_paid: float
    fill_rate: float
    partial_fill_rate: float


class PerformanceTracker:
    """
    Tracks and calculates performance metrics for sandbox simulation.
    """

    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.trades: List[VirtualTrade] = []
        self.daily_returns: List[DailyReturn] = []
        self.equity_curve: List[EquityPoint] = []

    def record_trade(self, trade: VirtualTrade):
        """Record a completed trade."""
        self.trades.append(trade)
        self._update_equity_curve()

    def _update_equity_curve(self):
        """Update equity curve with current portfolio value."""
        current_value = self._calculate_current_value()
        self.equity_curve.append(EquityPoint(
            timestamp=datetime.utcnow(),
            value=current_value,
            return_pct=(current_value - self.initial_balance) / self.initial_balance
        ))

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        returns = [d.return_pct for d in self.daily_returns]
        winning_trades = [t for t in self.trades if t.profit > 0]
        losing_trades = [t for t in self.trades if t.profit <= 0]

        # Time calculations
        if self.equity_curve:
            start_date = self.equity_curve[0].timestamp
            end_date = self.equity_curve[-1].timestamp
            trading_days = (end_date - start_date).days
        else:
            trading_days = 0

        # Basic returns
        final_balance = self._calculate_current_value()
        total_pnl = final_balance - self.initial_balance
        total_pnl_pct = total_pnl / self.initial_balance if self.initial_balance > 0 else 0

        # Annualized return
        if trading_days > 0 and total_pnl_pct > -1:
            years = trading_days / 365.25
            annualized_return = ((1 + total_pnl_pct) ** (1 / years)) - 1 if years > 0 else 0
        else:
            annualized_return = 0

        # Win/Loss statistics
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t.profit for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.profit for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # Profit factor
        gross_profits = sum(t.profit for t in winning_trades)
        gross_losses = abs(sum(t.profit for t in losing_trades))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        # Risk metrics
        volatility = self._calculate_volatility(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns, annualized_return)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_dd_duration = self._calculate_max_drawdown()

        return PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            starting_balance=self.initial_balance,
            ending_balance=final_balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            annualized_return=annualized_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            profit_factor=profit_factor,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_dd_duration,
            calmar_ratio=annualized_return / max_drawdown if max_drawdown > 0 else float('inf'),
            avg_position_size=self._avg_position_size(),
            max_position_size=self._max_position_size(),
            avg_hold_time_hours=self._avg_hold_time(),
            max_hold_time_hours=self._max_hold_time(),
            min_hold_time_hours=self._min_hold_time(),
            traders_copied=self._unique_traders(),
            profitable_traders=self._profitable_traders(),
            top_performing_trader=self._top_trader(),
            worst_performing_trader=self._worst_trader(),
            trader_specific_returns=self._trader_returns(),
            avg_slippage=self._avg_slippage(),
            total_fees_paid=self._total_fees(),
            fill_rate=self._fill_rate(),
            partial_fill_rate=self._partial_fill_rate()
        )
```

#### 12.4.2 Performance Reports

```python
class PerformanceReporter:
    """
    Generates comprehensive performance reports for sandbox simulation.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path

    def generate_report(
        self,
        metrics: PerformanceMetrics,
        trades: List[VirtualTrade],
        equity_curve: List[EquityPoint]
    ) -> str:
        """
        Generate HTML/PDF performance report.

        Sections:
        1. Executive Summary
        2. Performance Overview (charts)
        3. Risk Analysis
        4. Trade Analysis
        5. Trader Copy Performance
        6. Execution Quality
        7. Recommendations
        """
        report = f"""
# Sandbox Simulation Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Simulation Period:** {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}
**Duration:** {metrics.trading_days} days

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Starting Balance | ${metrics.starting_balance:,.2f} |
| Ending Balance | ${metrics.ending_balance:,.2f} |
| **Total P&L** | **${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:.2%})** |
| Annualized Return | {metrics.annualized_return:.2%} |
| Win Rate | {metrics.win_rate:.2%} |
| Profit Factor | {metrics.profit_factor:.2f} |
| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |
| Max Drawdown | {metrics.max_drawdown:.2%} |

---

## 2. Performance Overview

### Equity Curve

```
{self._generate_equity_chart(equity_curve)}
```

### Monthly Returns

| Month | Return |
|-------|--------|
{self._generate_monthly_returns_table(metrics, trades)}

---

## 3. Risk Analysis

| Metric | Value | Assessment |
|--------|-------|------------|
| Volatility | {metrics.volatility:.2%} | {"High" if metrics.volatility > 0.05 else "Medium" if metrics.volatility > 0.02 else "Low"} |
| Sharpe Ratio | {metrics.sharpe_ratio:.2f} | {"Excellent" if metrics.sharpe_ratio > 2 else "Good" if metrics.sharpe_ratio > 1 else "Fair" if metrics.sharpe_ratio > 0.5 else "Poor"} |
| Sortino Ratio | {metrics.sortino_ratio:.2f} | {"Excellent" if metrics.sortino_ratio > 3 else "Good" if metrics.sortino_ratio > 2 else "Fair" if metrics.sortino_ratio > 1 else "Poor"} |
| Max Drawdown | {metrics.max_drawdown:.2%} | {"Severe" if metrics.max_drawdown > 0.3 else "Moderate" if metrics.max_drawdown > 0.15 else "Acceptable"} |
| Calmar Ratio | {metrics.calmar_ratio:.2f} | {"Excellent" if metrics.calmar_ratio > 3 else "Good" if metrics.calmar_ratio > 1 else "Fair" if metrics.calmar_ratio > 0.5 else "Poor"} |

---

## 4. Trade Analysis

| Metric | Value |
|--------|-------|
| Total Trades | {metrics.total_trades} |
| Winning Trades | {metrics.winning_trades} |
| Losing Trades | {metrics.losing_trades} |
| Average Win | ${metrics.avg_win:,.2f} |
| Average Loss | ${metrics.avg_loss:,.2f} |
| Win/Loss Ratio | {metrics.win_loss_ratio:.2f} |
| Average Position Size | ${metrics.avg_position_size:,.2f} |
| Maximum Position Size | ${metrics.max_position_size:,.2f} |
| Average Hold Time | {metrics.avg_hold_time_hours:.1f} hours |

---

## 5. Trader Copy Performance

| Metric | Value |
|--------|-------|
| Traders Copied | {metrics.traders_copied} |
| Profitable Traders | {metrics.profitable_traders} |
| Success Rate | {metrics.profitable_traders/metrics.traders_copied:.2%} if metrics.traders_copied > 0 else 0 |

**Top Performer:** {metrics.top_performing_trader}
**Worst Performer:** {metrics.worst_performing_trader}

### Individual Trader Returns

| Trader | Return | Trades | Win Rate |
|--------|--------|--------|----------|
{self._generate_trader_table(metrics)}

---

## 6. Execution Quality

| Metric | Value |
|--------|-------|
| Average Slippage | {metrics.avg_slippage:.4f} |
| Total Fees Paid | ${metrics.total_fees_paid:,.2f} |
| Fill Rate | {metrics.fill_rate:.2%} |
| Partial Fill Rate | {metrics.partial_fill_rate:.2%} |

---

## 7. Recommendations

{self._generate_recommendations(metrics)}

---

*Report generated by CopyCat Sandbox Simulation Engine*
"""
        return report

    def save_report(self, report: str, filename: str = None):
        """Save report to file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"sandbox_report_{timestamp}.md"

        filepath = os.path.join(self.output_path, filename)
        os.makedirs(self.output_path, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(report)

        return filepath
```

---

### 12.5 Backtesting Module

#### 12.5.1 Historical Data Simulation

```python
class BacktestEngine:
    """
    Runs historical backtests using saved market data.
    Validates strategies against past market conditions.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.historical_data_loader = HistoricalDataLoader()
        self.simulation_engine = VirtualOrderExecutor(SandboxConfig())

    async def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        trader_selection_config: TraderSelectionConfig,
        copy_trading_config: TraderCopyConfig
    ) -> BacktestResult:
        """
        Run comprehensive backtest.

        Process:
        1. Load historical market data
        2. Simulate day-by-day trading
        3. Track trader performance in historical context
        4. Calculate performance metrics
        5. Generate backtest report
        """
        # Step 1: Load Historical Data
        markets = await self.historical_data_loader.load_markets(
            start_date=start_date,
            end_date=end_date
        )

        # Step 2: Initialize Simulation State
        portfolio_manager = VirtualPortfolioManager(
            SandboxConfig(initial_balance=self.config.starting_balance)
        )
        performance_tracker = PerformanceTracker(self.config.starting_balance)

        # Step 3: Run Day-by-Day Simulation
        current_date = start_date
        while current_date <= end_date:
            # Get historical market state for this date
            market_state = await self.historical_data_loader.get_market_state(
                date=current_date,
                markets=markets
            )

            # Get historical trades that occurred on this date
            historical_trades = await self.historical_data_loader.get_trades(
                date=current_date,
                markets=markets
            )

            # Process each historical trade
            for trade in historical_trades:
                # Check if we should copy this trader
                should_copy = self._evaluate_trader(
                    trade=trade,
                    config=trader_selection_config,
                    historical_context=market_state
                )

                if should_copy:
                    # Execute virtual copy trade
                    result = await portfolio_manager.execute_copy_trade(
                        source_trade=trade,
                        trader_config=copy_trading_config
                    )

                    if result.status == "FILLED":
                        # Record trade for performance tracking
                        performance_tracker.record_trade(VirtualTrade(
                            trade_id=result.execution_id,
                            market_id=trade.market_id,
                            outcome=trade.outcome,
                            quantity=result.position_size,
                            entry_price=result.execution_price,
                            exit_price=None,
                            profit=result.profit if hasattr(result, 'profit') else 0,
                            roi=result.roi if hasattr(result, 'roi') else 0,
                            timestamp=current_date,
                            source_trader=trade.trader_address
                        ))

            # Update position prices for end-of-day valuation
            portfolio_manager.update_market_prices(market_state.current_prices)

            # Advance to next day
            current_date += timedelta(days=1)

        # Step 4: Calculate Final Metrics
        metrics = performance_tracker.calculate_metrics()

        # Step 5: Generate Report
        reporter = PerformanceReporter(self.config.output_path)
        report = reporter.generate_report(
            metrics=metrics,
            trades=performance_tracker.trades,
            equity_curve=performance_tracker.equity_curve
        )

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            starting_balance=self.config.starting_balance,
            ending_balance=portfolio_manager.get_portfolio_summary().total_value,
            total_pnl=metrics.total_pnl,
            total_pnl_pct=metrics.total_pnl_pct,
            metrics=metrics,
            trades=performance_tracker.trades,
            report=report
        )
```

---

### 12.6 Sandbox CLI & Dashboard

#### 12.6.1 Command-Line Interface

```python
class SandboxCLI:
    """
    Command-line interface for sandbox simulation management.
    """

    def __init__(self):
        self.sandbox_manager = SandboxManager()

    def run(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description='CopyCat Sandbox - Paper Trading Simulation'
        )
        parser.add_argument(
            'command',
            choices=['start', 'stop', 'status', 'report', 'backtest'],
            help='Command to execute'
        )
        parser.add_argument(
            '--mode',
            choices=['realtime', 'accelerated'],
            default='realtime',
            help='Simulation mode'
        )
        parser.add_argument(
            '--duration',
            type=int,
            default=30,
            help='Simulation duration in days (for accelerated mode)'
        )
        parser.add_argument(
            '--balance',
            type=float,
            default=10000,
            help='Starting virtual balance'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='./sandbox_results',
            help='Output directory for results'
        )

        args = parser.parse_args()

        if args.command == 'start':
            self._cmd_start(args)
        elif args.command == 'stop':
            self._cmd_stop()
        elif args.command == 'status':
            self._cmd_status()
        elif args.command == 'report':
            self._cmd_report(args)
        elif args.command == 'backtest':
            self._cmd_backtest(args)

    def _cmd_start(self, args):
        """Start sandbox simulation."""
        config = SandboxConfig(
            mode="SANDBOX",
            initial_balance=args.balance,
            results_storage_path=args.output,
            use_historical_data=False,
            replay_speed=1.0 if args.mode == 'realtime' else 100.0
        )

        self.sandbox_manager.start_simulation(config)
        print(f"Sandbox simulation started with ${args.balance} balance")
        print(f"Results will be saved to: {args.output}")

    def _cmd_stop(self):
        """Stop sandbox simulation."""
        self.sandbox_manager.stop_simulation()
        print("Sandbox simulation stopped")

    def _cmd_status(self):
        """Show sandbox status."""
        status = self.sandbox_manager.get_status()
        print(f"Status: {status.state}")
        print(f"Uptime: {status.uptime}")
        print(f"Portfolio Value: ${status.portfolio_value:,.2f}")
        print(f"Total P&L: ${status.total_pnl:,.2f} ({status.total_pnl_pct:.2%})")
        print(f"Open Positions: {status.open_positions}")
        print(f"Completed Trades: {status.completed_trades}")

    def _cmd_report(self, args):
        """Generate performance report."""
        reporter = PerformanceReporter(args.output)
        metrics = self.sandbox_manager.get_performance_metrics()
        report = reporter.generate_report(
            metrics=metrics,
            trades=self.sandbox_manager.get_trades(),
            equity_curve=self.sandbox_manager.get_equity_curve()
        )
        filepath = reporter.save_report(report)
        print(f"Report saved to: {filepath}")

    def _cmd_backtest(self, args):
        """Run backtest."""
        engine = BacktestEngine(BacktestConfig(
            starting_balance=args.balance,
            output_path=args.output
        ))

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.duration)

        result = asyncio.run(engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            trader_selection_config=TraderSelectionConfig(),
            copy_trading_config=TraderCopyConfig()
        ))

        print(f"Backtest completed:")
        print(f"  Starting Balance: ${result.starting_balance:,.2f}")
        print(f"  Ending Balance: ${result.ending_balance:,.2f}")
        print(f"  Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2%})")
        print(f"  Win Rate: {result.metrics.win_rate:.2%}")
        print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
```

#### 12.6.2 Real-Time Dashboard

```python
class SandboxDashboard:
    """
    Real-time dashboard for monitoring sandbox simulation.
    """

    def __init__(self, sandbox_manager):
        self.manager = sandbox_manager
        self.last_update = None

    def render(self):
        """Render dashboard to terminal."""
        while True:
            status = self.manager.get_status()
            metrics = self.manager.get_performance_metrics()

            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')

            # Header
            print("=" * 80)
            print(f"  CopyCat Sandbox Simulation - Real-Time Dashboard")
            print("=" * 80)
            print()

            # Portfolio Summary
            print("PORTFOLIO SUMMARY")
            print("-" * 40)
            print(f"  Balance:        ${status.balance:>15,.2f}")
            print(f"  Positions:      ${status.positions_value:>15,.2f}")
            print(f"  Total Value:    ${status.total_value:>15,.2f}")
            print(f"  Unrealized P&L: ${status.unrealized_pnl:>15,.2f}")
            print()

            # Performance Metrics
            print("PERFORMANCE")
            print("-" * 40)
            print(f"  Total P&L:      ${metrics.total_pnl:>15,.2f} ({metrics.total_pnl_pct:>8.2%})")
            print(f"  Win Rate:       {metrics.win_rate:>15.2%}")
            print(f"  Profit Factor:  {metrics.profit_factor:>15.2f}")
            print(f"  Sharpe Ratio:   {metrics.sharpe_ratio:>15.2f}")
            print(f"  Max Drawdown:   {metrics.max_drawdown:>15.2%}")
            print()

            # Trade Statistics
            print("TRADE STATISTICS")
            print("-" * 40)
            print(f"  Total Trades:   {metrics.total_trades:>15}")
            print(f"  Winning:        {metrics.winning_trades:>15}")
            print(f"  Losing:         {metrics.losing_trades:>15}")
            print(f"  Fill Rate:      {metrics.fill_rate:>15.2%}")
            print(f"  Avg Slippage:   {metrics.avg_slippage:>15.4f}")
            print()

            # Top Positions
            print("TOP POSITIONS")
            print("-" * 80)
            print(f"  {'Market':<50} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P&L':>10}")
            print("-" * 80)
            for position in status.top_positions[:5]:
                print(f"  {position.market_name[:48]:<50} {position.quantity:>8.2f} "
                      f"${position.avg_price:>9.2f} ${position.current_price:>9.2f} "
                      f"${position.unrealized_pnl:>9.2f}")
            print()

            # Recent Trades
            print("RECENT TRADES")
            print("-" * 80)
            print(f"  {'Time':<20} {'Market':<30} {'Side':>6} {'Qty':>8} {'Price':>10} {'P&L':>10}")
            print("-" * 80)
            for trade in status.recent_trades[:5]:
                print(f"  {trade.timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{trade.market_name[:28]:<30} {trade.side:>6} "
                      f"{trade.quantity:>8.2f} ${trade.price:>9.2f} "
                      f"${trade.profit:>9.2f}")
            print()

            # Footer
            print("=" * 80)
            print(f"  Status: {status.state.upper()} | "
                  f"Uptime: {status.uptime} | "
                  f"Traders Copied: {status.traders_copied} | "
                  f"Press Ctrl+C to exit")
            print("=" * 80)

            # Sleep before refresh
            time.sleep(5)
```

---

### 12.7 Sandbox Integration Points

#### 12.7.1 Configuration Switch

```python
# Main configuration file (config.py)

# Sandbox Configuration
SANDBOX_MODE = True  # Set to False for live trading
SANDBOX_CONFIG = SandboxConfig(
    mode="SANDBOX",
    initial_balance=50000,  # $50K starting balance for testing
    simulate_slippage=True,
    simulate_fees=True,
    persist_results=True,
    results_storage_path="./sandbox_results"
)

# Live Trading Configuration (when SANDBOX_MODE = False)
LIVE_CONFIG = LiveTradingConfig(
    api_keys={
        "polymarket": os.getenv("POLYMARKET_API_KEY"),
        "kalshi": os.getenv("KALSHI_API_KEY")
    },
    wallet_address=os.getenv("WALLET_ADDRESS"),
    max_position_size_pct=0.10,
    max_total_exposure_pct=0.50,
    require_manual_approval_above=1000.0,  # $1K+ trades need manual approval
    alerts_enabled=True
)

# Unified entry point
def get_trading_config():
    if SANDBOX_MODE:
        return SANDBOX_CONFIG
    else:
        return LIVE_CONFIG
```

#### 12.7.2 Module Dependencies

```python
# sandbox/dependencies.py

from .simulation_engine import VirtualOrderExecutor
from .portfolio_manager import VirtualPortfolioManager
from .performance_tracker import PerformanceTracker, PerformanceMetrics
from .backtest_engine import BacktestEngine
from .dashboard import SandboxDashboard
from .cli import SandboxCLI

# Export for main application
__all__ = [
    'VirtualOrderExecutor',
    'VirtualPortfolioManager',
    'PerformanceTracker',
    'PerformanceMetrics',
    'BacktestEngine',
    'SandboxDashboard',
    'SandboxCLI'
]
```

---

### 12.8 Testing Protocol

#### 12.8.1 Validation Checklist

Before transitioning from sandbox to live trading, the bot must pass:

```markdown
## Sandbox Validation Checklist

### Performance Requirements
- [ ] Average monthly return > 5%
- [ ] Win rate > 55%
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 20%
- [ ] Profit factor > 1.5
- [ ] Consistent performance across 3+ months of simulation

### Stability Requirements
- [ ] Zero unhandled exceptions during 30-day simulation
- [ ] No memory leaks (memory usage < 500MB sustained)
- [ ] No network timeout cascades
- [ ] Graceful handling of API rate limits

### Accuracy Requirements
- [ ] Trader selection matches expected profitable traders
- [ ] Copy trade execution matches paper trading expectations
- [ ] P&L calculation matches manual verification
- [ ] Fee calculation matches platform actual fees

### Risk Management
- [ ] Position sizing limits enforced correctly
- [ ] Total exposure limits enforced correctly
- [ ] Stop-loss triggers when expected
- [ ] No over-trading (max 50 orders/day)

### Execution Quality
- [ ] Fill rate > 95%
- [ ] Average slippage < 0.5%
- [ ] No phantom orders or double executions
```

#### 12.8.2 Sandbox-to-Live Transition Checklist

```python
TRANSITION_CHECKLIST = {
    "pre_deployment": [
        "Run 90-day continuous sandbox simulation",
        "Achieve positive ROI in at least 4 of 6 months",
        "No single month loss > 10%",
        "Sharpe ratio > 1.0 sustained",
        "Document all parameters and configurations",
        "Create runbook for live trading operations"
    ],
    "security": [
        "API keys stored in environment variables",
        "2FA enabled on all exchange accounts",
        "Withdrawal whitelisting configured",
        "Alert thresholds configured for all major events",
        "Kill switch tested and functional"
    ],
    "monitoring": [
        "Real-time dashboard operational",
        "Performance alerts configured",
        "Error alerting configured",
        "Daily performance reports scheduled",
        "Emergency contact list documented"
    ],
    "operational": [
        "Backup and recovery procedures tested",
        "Rollback plan documented",
        "Communication channels established",
        "Decision tree for trading pauses documented"
    ]
}
```

---

### 12.9 File Structure

```
sandbox/
├── __init__.py
├── config.py                    # Sandbox configuration
├── simulation_engine.py         # VirtualOrderExecutor
├── portfolio_manager.py         # VirtualPortfolioManager
├── performance_tracker.py       # PerformanceTracker, PerformanceMetrics
├── performance_reporter.py      # PerformanceReporter
├── backtest_engine.py           # BacktestEngine
├── dashboard.py                 # SandboxDashboard
├── cli.py                       # SandboxCLI
├── data/
│   ├── __init__.py
│   ├── historical_loader.py     # HistoricalDataLoader
│   └── market_data_cache.py     # MarketDataCache
├── tests/
│   ├── __init__.py
│   ├── test_simulation_engine.py
│   ├── test_portfolio_manager.py
│   ├── test_performance_tracker.py
│   └── test_backtest_engine.py
└── results/
    ├── .gitkeep
    └── reports/                 # Generated performance reports
```

---

### 12.10 Success Criteria Summary

**Minimum Requirements for Live Trading Approval:**

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Monthly Return | > 3% | > 5% | > 8% |
| Win Rate | > 52% | > 58% | > 65% |
| Sharpe Ratio | > 0.8 | > 1.2 | > 1.5 |
| Max Drawdown | < 25% | < 15% | < 10% |
| Profit Factor | > 1.2 | > 1.5 | > 2.0 |
| Simulation Duration | 90 days | 180 days | 365 days |
| Consistency | 4/6 months positive | 5/6 months positive | All months positive |

**Critical Success Factors:**
1. **Consistency > Speed**: Bot must generate steady returns, not volatile swings
2. **Risk Control**: Never lose more than 20% of portfolio in worst case
3. **Reproducibility**: Results must be consistent across multiple simulation runs
4. **Transparency**: Every trade must be logged with full context
5. **Learning**: Bot should improve over time as it "learns" from more trader data

---

*End of Section 12*