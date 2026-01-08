"""
TUI Dashboard for Sandbox Simulation.

Provides a terminal-based real-time dashboard for monitoring sandbox
simulations with portfolio tracking, performance metrics, and trade history.
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.live import Live
from rich.prompt import Prompt
from rich.style import Style
from rich.color import Color
from rich.tree import Tree
from rich import box

from sandbox.runner import SandboxRunner
from sandbox.config import PortfolioSummary, VirtualTrade, VirtualPosition
from sandbox.analytics import PerformanceMetrics, EquityPoint


# Try to import for ASCII charts, fallback to simple implementation
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class DashboardConfig:
    """Configuration for TUI dashboard."""
    refresh_interval: float = 1.0  # seconds
    max_positions_shown: int = 10
    max_trades_shown: int = 20
    chart_height: int = 10
    theme: str = "default"
    show_charts: bool = True


class TUIDashboard:
    """
    Terminal UI dashboard for sandbox simulation monitoring.
    
    Features:
    - Real-time portfolio tracking
    - Performance metrics display
    - Position management view
    - Trade history log
    - Equity curve visualization
    - Keyboard navigation
    """

    def __init__(self, runner: SandboxRunner, config: DashboardConfig = None):
        """Initialize dashboard with sandbox runner."""
        self.runner = runner
        self.config = config or DashboardConfig()
        self.console = Console()
        self.running = False
        self.current_view = "overview"
        self.views = ["overview", "positions", "trades", "performance", "charts"]
        self.start_time = datetime.utcnow()
        self.trade_counter = 0
        self.last_summary: Optional[PortfolioSummary] = None
        self.last_metrics: Optional[PerformanceMetrics] = None
        
        # Register callbacks with runner
        self.runner.set_order_callback(self._on_order_executed)

    def _on_order_executed(self, order):
        """Callback when an order is executed."""
        self.trade_counter += 1

    def _get_elapsed_time(self) -> str:
        """Get elapsed time since dashboard started."""
        elapsed = datetime.utcnow() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _render_header(self) -> Panel:
        """Render dashboard header."""
        elapsed = self._get_elapsed_time()
        
        header_content = f"""
[bold cyan]CopyCat[/bold cyan] [yellow]Sandbox Dashboard[/yellow]
[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]
[bold]Time Running:[/bold] {elapsed}    [bold]Trades Executed:[/bold] {self.trade_counter}
[bold]Mode:[/bold] {self.runner.config.mode}    [bold]Initial Balance:[/bold] ${self.runner.config.initial_balance:,.2f}
"""
        
        return Panel(
            Text.from_markup(header_content),
            title="[bold]Sandbox Simulation[/bold]",
            subtitle=f"View: [bold]{self.current_view.upper()}[/bold] | Press 'q' to quit, '1-5' to switch views",
            box=box.ROUNDED,
            style="cyan"
        )

    def _render_overview(self) -> Panel:
        """Render overview panel with key metrics."""
        summary = self.runner.get_portfolio_summary()
        metrics = self.runner.get_performance_metrics()
        self.last_summary = summary
        self.last_metrics = metrics
        
        # Calculate key metrics
        pnl_emoji = "🟢" if summary.unrealized_pnl >= 0 else "🔴"
        pnl_sign = "+" if summary.unrealized_pnl >= 0 else ""
        
        # Create metrics grid
        metrics_table = Table.grid(padding=1)
        metrics_table.add_column(ratio=1)
        metrics_table.add_column(ratio=1)
        
        metrics_table.add_row(
            f"[bold]Total Value[/bold]",
            f"[bold]{pnl_emoji} ${summary.total_value:,.2f}[/bold]"
        )
        metrics_table.add_row(
            "Cash Balance",
            f"${summary.balance:,.2f}"
        )
        metrics_table.add_row(
            "Positions Value",
            f"${summary.positions_value:,.2f}"
        )
        metrics_table.add_row(
            "Unrealized P&L",
            f"{pnl_sign}${summary.unrealized_pnl:,.2f} ({pnl_sign}{summary.unrealized_pnl/summary.positions_value*100:.2f}%)"
        )
        metrics_table.add_row(
            "",
            ""
        )
        metrics_table.add_row(
            "[bold]Performance[/bold]",
            ""
        )
        metrics_table.add_row(
            "Total P&L",
            f"{pnl_sign}${metrics.total_pnl:,.2f} ({pnl_sign}{metrics.total_pnl_pct:.2%})"
        )
        metrics_table.add_row(
            "Win Rate",
            f"{metrics.win_rate:.2%}"
        )
        metrics_table.add_row(
            "Sharpe Ratio",
            f"{metrics.sharpe_ratio:.2f}"
        )
        metrics_table.add_row(
            "Max Drawdown",
            f"{metrics.max_drawdown:.2%}"
        )
        
        content = f"""
[bold]Portfolio Summary[/bold]

{metrics_table}

[dim]Last Updated: {datetime.utcnow().strftime('%H:%M:%S')}[/dim]
"""
        
        return Panel(
            Text.from_markup(content),
            title="[bold]📊 Overview[/bold]",
            box=box.ROUNDED,
            style="green"
        )

    def _render_positions(self) -> Panel:
        """Render positions panel."""
        summary = self.runner.get_portfolio_summary()
        positions = summary.__dict__ if hasattr(summary, '__dict__') else {}
        
        # Get positions from portfolio manager
        if hasattr(self.runner, 'portfolio_manager'):
            positions_dict = self.runner.portfolio_manager.positions
        else:
            positions_dict = {}
        
        if not positions_dict:
            return Panel(
                Text.from_markup("[yellow]No open positions[/yellow]"),
                title="[bold]📈 Positions[/bold]",
                box=box.ROUNDED,
                style="blue"
            )
        
        # Create positions table
        table = Table(
            title="Open Positions",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        table.add_column("Market", style="cyan", no_wrap=True)
        table.add_column("Qty", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("ROI", justify="right")
        
        # Sort positions by value (descending)
        sorted_positions = sorted(
            positions_dict.items(),
            key=lambda x: x[1].quantity * x[1].current_price,
            reverse=True
        )[:self.config.max_positions_shown]
        
        for market_id, position in sorted_positions:
            pnl = position.unrealized_pnl
            roi = (position.current_price - position.avg_price) / position.avg_price if position.avg_price > 0 else 0
            
            pnl_style = "green" if pnl >= 0 else "red"
            roi_style = "green" if roi >= 0 else "red"
            roi_sign = "+" if roi >= 0 else ""
            pnl_sign = "+" if pnl >= 0 else ""
            
            table.add_row(
                market_id,
                f"{position.quantity:.4f}",
                f"${position.avg_price:.4f}",
                f"${position.current_price:.4f}",
                f"[{pnl_style}]{pnl_sign}${pnl:.2f}[/]",
                f"[{roi_style}]{roi_sign}{roi:.2%}[/]"
            )
        
        return Panel(
            table,
            title="[bold]📈 Positions[/bold]",
            box=box.ROUNDED,
            style="blue"
        )

    def _render_trades(self) -> Panel:
        """Render trade history panel."""
        # Get trades from portfolio manager
        if hasattr(self.runner, 'portfolio_manager'):
            trades = self.runner.portfolio_manager.completed_trades
        else:
            trades = []
        
        if not trades:
            return Panel(
                Text.from_markup("[yellow]No completed trades yet[/yellow]"),
                title="[bold]📜 Trade History[/bold]",
                box=box.ROUNDED,
                style="yellow"
            )
        
        # Create trades table
        table = Table(
            title="Recent Trades",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        table.add_column("Time", style="dim", width=10)
        table.add_column("Market", style="cyan", no_wrap=True)
        table.add_column("Side", justify="center")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("P&L", justify="right")
        
        # Show most recent trades first
        recent_trades = sorted(trades, key=lambda t: t.timestamp, reverse=True)[:self.config.max_trades_shown]
        
        for trade in recent_trades:
            pnl = trade.profit
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else ""
            
            time_str = trade.timestamp.strftime('%H:%M:%S') if trade.timestamp else "N/A"
            
            table.add_row(
                time_str,
                trade.market_id,
                trade.outcome,
                f"{trade.quantity:.4f}",
                f"${trade.entry_price:.4f}",
                f"[{pnl_style}]{pnl_sign}${pnl:.2f}[/]"
            )
        
        return Panel(
            table,
            title="[bold]📜 Trade History[/bold]",
            box=box.ROUNDED,
            style="yellow"
        )

    def _render_performance(self) -> Panel:
        """Render performance metrics panel."""
        metrics = self.runner.get_performance_metrics()
        self.last_metrics = metrics
        
        # Create performance table
        table = Table(
            title="Performance Metrics",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")
        
        # Define metrics to display with thresholds
        metrics_display = [
            ("Total P&L", f"${metrics.total_pnl:,.2f}", metrics.total_pnl >= 0),
            ("Return", f"{metrics.total_pnl_pct:.2%}", metrics.total_pnl_pct > 0),
            ("Win Rate", f"{metrics.win_rate:.2%}", metrics.win_rate >= 0.5),
            ("Profit Factor", f"{metrics.profit_factor:.2f}", metrics.profit_factor >= 1.0),
            ("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}", metrics.sharpe_ratio >= 1.0),
            ("Max Drawdown", f"{metrics.max_drawdown:.2%}", metrics.max_drawdown <= 0.2),
            ("Total Trades", f"{metrics.total_trades}", metrics.total_trades > 0),
            ("Winning Trades", f"{metrics.winning_trades}", metrics.winning_trades > 0),
            ("Losing Trades", f"{metrics.losing_trades}", True),
            ("Avg Win", f"${metrics.avg_win:,.2f}", metrics.avg_win > 0),
            ("Avg Loss", f"${metrics.avg_loss:,.2f}", metrics.avg_loss <= 0),
            ("Avg Position", f"${metrics.avg_position_size:,.2f}", True),
            ("Traders Copied", f"{metrics.traders_copied}", True),
            ("Profitable Traders", f"{metrics.profitable_traders}", metrics.profitable_traders > 0),
        ]
        
        for metric, value, is_good in metrics_display:
            status = "✅" if is_good else "⚠️"
            table.add_row(metric, value, status)
        
        return Panel(
            table,
            title="[bold]📊 Performance[/bold]",
            box=box.ROUNDED,
            style="green"
        )

    def _render_charts(self) -> Panel:
        """Render equity curve chart."""
        if not self.config.show_charts:
            return Panel(
                Text.from_markup("[yellow]Charts disabled in config[/yellow]"),
                title="[bold]📈 Equity Curve[/bold]",
                box=box.ROUNDED,
                style="magenta"
            )
        
        # Get equity curve from performance tracker
        if hasattr(self.runner, 'tracker'):
            equity_curve = self.runner.tracker.equity_curve
        else:
            equity_curve = []
        
        if not equity_curve or len(equity_curve) < 2:
            return Panel(
                Text.from_markup("[yellow]Not enough data for chart yet...[/yellow]\n[dim]Keep trading to see equity curve![/dim]"),
                title="[bold]📈 Equity Curve[/bold]",
                box=box.ROUNDED,
                style="magenta"
            )
        
        # Generate ASCII chart
        chart = self._generate_ascii_chart(equity_curve)
        
        content = f"""
[bold]Equity Curve Over Time[/bold]

{chart}

[dim]Starting: ${equity_curve[0].value:,.2f} | Current: ${equity_curve[-1].value:,.2f} | Change: {((equity_curve[-1].value - equity_curve[0].value) / equity_curve[0].value * 100):+.2f}%[/dim]
"""
        
        return Panel(
            Text.from_markup(content),
            title="[bold]📈 Equity Curve[/bold]",
            box=box.ROUNDED,
            style="magenta"
        )

    def _generate_ascii_chart(self, equity_curve: List[EquityPoint]) -> str:
        """Generate ASCII equity curve chart."""
        if not equity_curve:
            return "No data available"
        
        # Limit data points for display
        max_points = 50
        if len(equity_curve) > max_points:
            # Sample evenly
            step = len(equity_curve) // max_points
            sampled = equity_curve[::step]
            if len(equity_curve) % max_points:
                sampled.append(equity_curve[-1])
        else:
            sampled = equity_curve
        
        # Extract values
        values = [point.value for point in sampled]
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val > min_val else 1
        
        # Chart dimensions
        width = 60
        height = self.config.chart_height
        
        # Create chart lines
        chart_lines = []
        for i in range(height):
            # Calculate threshold for this line
            threshold = max_val - (i + 1) * (value_range / height)
            threshold_next = max_val - i * (value_range / height)
            
            line_chars = []
            for val in values:
                if val >= threshold_next:
                    line_chars.append("█")
                elif val >= threshold:
                    line_chars.append("▄")
                else:
                    line_chars.append(" ")
            
            chart_lines.append("".join(line_chars))
        
        # Reverse so highest value is at top
        chart_lines.reverse()
        
        # Add Y-axis labels
        chart_with_labels = []
        for i, line in enumerate(chart_lines):
            # Calculate Y-axis label position
            label_idx = height - 1 - i
            if label_idx % 2 == 0:  # Show label every other line
                y_val = max_val - label_idx * (value_range / height)
                label = f"${y_val:,.0f}"
                if len(label) > 10:
                    label = label[:10]
                chart_with_labels.append(f"{label:>10} │ {line}")
            else:
                chart_with_labels.append(f"            │ {line}")
        
        # Add X-axis
        x_axis = "            " + "─" * (width + 2)
        chart_with_labels.append(x_axis)
        
        # Add time labels
        if sampled:
            start_time = sampled[0].timestamp.strftime('%H:%M') if sampled[0].timestamp else ""
            end_time = sampled[-1].timestamp.strftime('%H:%M') if sampled[-1].timestamp else ""
            chart_with_labels.append(f"            │ {start_time:^{width}} {end_time}")
        
        return "\n".join(chart_lines)

    def _render_layout(self) -> Layout:
        """Render the complete dashboard layout."""
        layout = Layout()
        
        layout.split(
            Layout(Panel(Text("CopyCat Sandbox Dashboard"), size=3, style="cyan"), name="header"),
            Layout(name="main")
        )
        
        layout["main"].split(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Different layouts based on current view
        if self.current_view == "overview":
            layout["main"].split_column(
                Layout(self._render_overview(), size=20),
                Layout(self._render_charts())
            )
        elif self.current_view == "positions":
            layout["main"].replace(
                Layout(self._render_positions(), size=30)
            )
        elif self.current_view == "trades":
            layout["main"].replace(
                Layout(self._render_trades(), size=30)
            )
        elif self.current_view == "performance":
            layout["main"].replace(
                Layout(self._render_performance(), size=30)
            )
        elif self.current_view == "charts":
            layout["main"].replace(
                Layout(self._render_charts())
            )
        
        return layout

    def _handle_input(self) -> bool:
        """Handle keyboard input. Returns False if should quit."""
        try:
            if self.console.is_terminal:
                # Check for keypress (simplified - in real implementation would use stdin)
                import sys
                import select
                
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    char = sys.stdin.read(1)
                    if char == 'q' or char == 'Q':
                        return False
                    elif char == '1':
                        self.current_view = "overview"
                    elif char == '2':
                        self.current_view = "positions"
                    elif char == '3':
                        self.current_view = "trades"
                    elif char == '4':
                        self.current_view = "performance"
                    elif char == '5':
                        self.current_view = "charts"
        except:
            pass
        
        return True

    async def run(self):
        """Run the dashboard with live updates."""
        self.running = True
        self.start_time = datetime.utcnow()
        
        self.console.clear()
        
        with Live(
            self._render_layout(),
            console=self.console,
            refresh_per_second=1.0 / self.config.refresh_interval,
            transient=False
        ) as live:
            while self.running:
                try:
                    # Check for exit condition
                    if not self._handle_input():
                        break
                    
                    # Update display
                    live.update(self._render_layout())
                    
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.1)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    # Log error but continue
                    self.console.print(f"[red]Error: {e}[/red]")
                    await asyncio.sleep(1)
        
        self.console.clear()
        self.console.print("[bold green]Dashboard closed. Goodbye![/bold green]")

    def stop(self):
        """Stop the dashboard."""
        self.running = False


class SimpleDashboard:
    """
    Simplified dashboard for basic terminal output.
    For use in environments where rich library is not available.
    """
    
    def __init__(self, runner: SandboxRunner):
        self.runner = runner
        self.start_time = datetime.utcnow()
    
    def _get_elapsed_time(self) -> str:
        elapsed = datetime.utcnow() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def display_summary(self):
        """Print summary to console."""
        summary = self.runner.get_portfolio_summary()
        metrics = self.runner.get_performance_metrics()
        
        print("\n" + "="*60)
        print(f"CopyCat Sandbox - {self._get_elapsed_time()}")
        print("="*60)
        print(f"Total Value:  ${summary.total_value:,.2f}")
        print(f"Cash Balance: ${summary.balance:,.2f}")
        print(f"Positions:    ${summary.positions_value:,.2f}")
        print(f"Unrealized:   ${summary.unrealized_pnl:,.2f}")
        print("-"*60)
        print(f"Total P&L:    ${metrics.total_pnl:,.2f} ({metrics.total_pnl_pct:+.2%})")
        print(f"Win Rate:     {metrics.win_rate:.2%}")
        print(f"Sharpe:       {metrics.sharpe_ratio:.2f}")
        print(f"Drawdown:     {metrics.max_drawdown:.2%}")
        print("="*60 + "\n")


async def run_dashboard(
    runner: SandboxRunner,
    mode: str = "rich",
    refresh_interval: float = 1.0
):
    """
    Run the sandbox dashboard.
    
    Args:
        runner: SandboxRunner instance
        mode: "rich" for full TUI, "simple" for basic output
        refresh_interval: Dashboard refresh interval in seconds
    """
    if mode == "rich":
        try:
            config = DashboardConfig(refresh_interval=refresh_interval)
            dashboard = TUIDashboard(runner, config)
            await dashboard.run()
        except ImportError:
            print("[yellow]Rich library not available, falling back to simple mode[/yellow]")
            SimpleDashboard(runner).display_summary()
    else:
        SimpleDashboard(runner).display_summary()


# Main execution
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        """Demo the dashboard with sample data."""
        from sandbox.config import SandboxConfig, VirtualOrder
        from datetime import datetime
        
        print("Initializing Sandbox Runner...")
        config = SandboxConfig(initial_balance=10000)
        runner = SandboxRunner(config)
        
        # Set up fallback market data
        def get_market_data(market_id: str):
            return {
                "market_id": market_id,
                "current_price": 0.5 + hash(market_id) % 50 / 100,
                "previous_price": 0.5,
                "volatility": 0.02,
            }
        runner.set_market_data_callback(get_market_data)
        
        print("Starting dashboard demo...")
        print("Press '1-5' to switch views, 'q' to quit\n")
        
        # Run dashboard
        await run_dashboard(runner, mode="rich", refresh_interval=1.0)
    
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
