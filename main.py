"""
main.py (visually improved)
- Uses 'rich' for colored banners, prompts, tables, and spinners if available.
- Falls back to plain text when 'rich' is not installed.
- Preserves all menu/logic/undo/encoding features.
"""
import argparse
import sys
import traceback
from typing import Optional, List

from data_input import read_file
from data_analysis import print_summary
from imputation import impute_dataframe
from feature_scaling import scale_dataframe
from visualization import DataVisualization  # improved visualization module

# Try to import rich; provide safe fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.text import Text
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None


UNDO_STACK_MAX = 5


def _echo(msg: str, style: Optional[str] = None):
    if RICH:
        if style:
            console.print(msg, style=style)
        else:
            console.print(msg)
    else:
        print(msg)


def _print_banner():
    if RICH:
        txt = Text("ML-Preprocessor-CLI", style="bold white on blue")
        console.print(Panel(txt, expand=False))
        console.print("A friendly CLI to preprocess and visualize tabular data\n", style="dim")
    else:
        print("=" * 40)
        print("ML-Preprocessor-CLI")
        print("=" * 40)


def _print_menu():
    if RICH:
        table = Table(show_header=False, box=None)
        table.add_row("[bold cyan]1.[/]", "Data Description")
        table.add_row("[bold cyan]2.[/]", "Handling NULL Values (Impute)")
        table.add_row("[bold cyan]3.[/]", "Encoding Categorical Data")
        table.add_row("[bold cyan]4.[/]", "Feature Scaling of the Dataset")
        table.add_row("[bold cyan]5.[/]", "Download the modified dataset")
        table.add_row("[bold cyan]6.[/]", "Visualize the Dataset")
        table.add_row("[bold cyan]7.[/]", "Undo last mutation (encoding/imputation/scaling)")
        table.add_row("[bold red]-1.[/]", "Exit")
        console.print(Panel(table, title="Tasks (Preprocessing)👇", expand=False))
    else:
        print("\nTasks (Preprocessing)👇\n")
        print("1. Data Description")
        print("2. Handling NULL Values")
        print("3. Encoding Categorical Data")
        print("4. Feature Scaling of the Dataset")
        print("5. Download the modified dataset")
        print("6. Visualize the Dataset")
        print("7. Undo last mutation (encoding/imputation/scaling)")
        print("Press -1 to exit\n")


def _select_columns_interactive(available: List[str]) -> List[str]:
    """
    Same selection helper but with rich-friendly prompts.
    """
    if RICH:
        console.print("\nAvailable columns:", style="bold")
        for i, c in enumerate(available, 1):
            console.print(f"[cyan]{i}.[/] {c}")
        raw = Prompt.ask("Enter column numbers or names (comma-separated). Leave blank to select all", default="")
    else:
        print("Available columns:")
        for i, c in enumerate(available, 1):
            print(f"{i}. {c}")
        raw = input("Enter column numbers or names (comma-separated). Leave blank to select all: ").strip()

    if not raw:
        return available[:]

    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    selected = []
    for t in tokens:
        if t.isdigit():
            idx = int(t) - 1
            if 0 <= idx < len(available):
                selected.append(available[idx])
            else:
                _echo(f"Ignoring out-of-range index: {t}", style="yellow" if RICH else None)
        else:
            matches = [c for c in available if c.lower() == t.lower()]
            if matches:
                selected.append(matches[0])
            else:
                _echo(f"Ignoring unknown column name: {t}", style="yellow" if RICH else None)
    # dedupe preserving order
    final = []
    seen = set()
    for c in selected:
        if c not in seen:
            final.append(c)
            seen.add(c)
    return final


def interactive_menu_loop(df: Optional[object] = None):
    undo_stack = []

    def push_undo(snapshot):
        if snapshot is None:
            return
        undo_stack.append(snapshot.copy())
        if len(undo_stack) > UNDO_STACK_MAX:
            undo_stack.pop(0)

    def do_undo():
        nonlocal df
        if not undo_stack:
            _echo("Nothing to undo.", style="yellow" if RICH else None)
            return df
        df = undo_stack.pop()
        _echo("Reverted to previous dataset state.", style="green" if RICH else None)
        return df

    while True:
        _print_menu()
        if RICH:
            choice = Prompt.ask("What do you want to do? (Press -1 to exit)", default="")
        else:
            choice = input("What do you want to do? (Press -1 to exit): ").strip()

        if choice == "-1":
            _echo("Exiting. Goodbye 👋", style="bold" if RICH else None)
            break

        if df is None and choice not in ("-1",):
            if RICH:
                path = Prompt.ask("No dataset loaded. Enter path to dataset file")
            else:
                path = input("No dataset loaded. Enter path to dataset file: ").strip()
            try:
                df = read_file(path)
                _echo("Dataset loaded.", style="green" if RICH else None)
            except Exception as e:
                _echo(f"Failed to load file: {e}", style="red" if RICH else None)
                df = None
                continue

        if choice == "1":
            _echo("Data summary:", style="bold")
            print_summary(df, sample_rows=100)
        elif choice == "2":
            push_undo(df)
            _echo("Handling NULL values (imputation) — this may take a moment...", style="bold")
            if RICH:
                with console.status("[bold green]Imputing...[/]"):
                    df = impute_dataframe(df)
            else:
                df = impute_dataframe(df)
            _echo("Imputation completed.", style="green")
        elif choice == "3":
            from encoding import label_encode, one_hot_encode, detect_categorical_columns

            cat_cols = detect_categorical_columns(df)
            if not cat_cols:
                _echo("No categorical columns found to encode.", style="yellow")
                continue

            if RICH:
                console.print("Categorical columns detected:", style="bold")
                for i, c in enumerate(cat_cols, 1):
                    console.print(f"[cyan]{i}.[/] {c}")
            else:
                print("Categorical columns detected:", cat_cols)

            selected = _select_columns_interactive(cat_cols)
            if not selected:
                _echo("No columns selected. Aborting encoding.", style="yellow")
                continue

            if RICH:
                enc_choice = Prompt.ask("Choose encoding: 1) Label Encoding  2) One-Hot Encoding [default=1]", default="1")
            else:
                enc_choice = input("Choose encoding: 1) Label Encoding  2) One-Hot Encoding [default=1]: ").strip()

            push_undo(df)
            if enc_choice == "2":
                _echo("Applying one-hot encoding...", style="bold")
                if RICH:
                    with console.status("[bold green]Encoding...[/]"):
                        df = one_hot_encode(df, selected)
                else:
                    df = one_hot_encode(df, selected)
                _echo(f"One-hot encoding applied to: {selected}", style="green")
            else:
                _echo("Applying label encoding...", style="bold")
                if RICH:
                    with console.status("[bold green]Encoding...[/]"):
                        df = label_encode(df, selected)
                else:
                    df = label_encode(df, selected)
                _echo(f"Label encoding applied to: {selected}", style="green")
        elif choice == "4":
            numeric_cols = [c for c in df.columns if df[c].dtype.kind in "bifc"]
            if not numeric_cols:
                _echo("No numeric columns found to scale.", style="yellow")
            else:
                _echo(f"Numeric columns detected: {numeric_cols}", style="bold")
                if RICH:
                    col_choice = Prompt.ask("Scale all numeric columns? (y/n) [default=y]", default="y")
                else:
                    col_choice = input("Scale all numeric columns? (y/n) [default=y]: ").strip().lower()
                if str(col_choice).lower() in ("n", "no"):
                    selected = _select_columns_interactive(numeric_cols)
                else:
                    selected = numeric_cols
                if not selected:
                    _echo("No numeric columns selected.", style="yellow")
                    continue
                if RICH:
                    method = Prompt.ask("Choose scaler: 1) standard 2) minmax 3) robust [default=1]", default="1")
                else:
                    method = input("Choose scaler: 1) standard 2) minmax 3) robust [default=1]: ").strip()
                method_map = {"1": "standard", "2": "minmax", "3": "robust"}
                method_choice = method_map.get(method, "standard")
                push_undo(df)
                _echo("Applying scaling...", style="bold")
                if RICH:
                    with console.status("[bold green]Scaling...[/]"):
                        df, _ = scale_dataframe(df, selected, method=method_choice)
                else:
                    df, _ = scale_dataframe(df, selected, method=method_choice)
                _echo(f"Scaling applied to: {selected}", style="green")
        elif choice == "5":
            if RICH:
                out_path = Prompt.ask("Enter output CSV filename to save (example: out.csv)", default="output.csv")
            else:
                out_path = input("Enter output CSV filename to save (example: out.csv): ").strip() or "output.csv"
            try:
                df.to_csv(out_path, index=False)
                _echo(f"Saved to {out_path}", style="green")
            except Exception as e:
                _echo(f"Failed to save: {e}", style="red")
        elif choice == "6":
            vis = DataVisualization(df)
            vis.run_visualization()
        elif choice == "7":
            df = do_undo()
        else:
            _echo("Invalid choice. Try again.", style="yellow")


def quick_run_with_file(file_path: str, sample: int, no_visual: bool):
    _echo("Loading file...", style="bold")
    df = read_file(file_path, sample_rows=sample if sample > 0 else None)
    _echo("Data loaded. Basic summary:", style="bold")
    print_summary(df, sample_rows=100)

    if df.isna().sum().sum() > 0:
        _echo("Missing values detected. Applying default imputation...", style="bold")
        if RICH:
            with console.status("[bold green]Imputing...[/]"):
                df = impute_dataframe(df)
        else:
            df = impute_dataframe(df)

    if RICH:
        choice = Prompt.ask("Would you like to scale numeric columns? (y/n)", default="n")
    else:
        choice = input("\nWould you like to scale numeric columns? (y/n): ").strip().lower()

    if choice and choice.lower() in ("y", "yes"):
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in "bifc"]
        if not numeric_cols:
            _echo("No numeric columns found to scale.", style="yellow")
        else:
            _echo(f"Numeric columns detected: {numeric_cols}", style="bold")
            if RICH:
                method = Prompt.ask("Choose scaler: 1) standard 2) minmax 3) robust [default=1]", default="1")
            else:
                method = input("Choose scaler: 1) standard 2) minmax 3) robust [default=1]: ").strip()
            method_map = {"1": "standard", "2": "minmax", "3": "robust"}
            method_choice = method_map.get(method, "standard")
            _echo("Applying scaling...", style="bold")
            if RICH:
                with console.status("[bold green]Scaling...[/]"):
                    df, _scaler = scale_dataframe(df, numeric_cols, method=method_choice)
            else:
                df, _scaler = scale_dataframe(df, numeric_cols, method=method_choice)
            _echo("Scaling applied.", style="green")

    if not no_visual:
        vis = DataVisualization(df)
        vis.run_visualization()
    else:
        _echo("Skipping visualization (--no-visual).", style="dim")


def parse_args():
    p = argparse.ArgumentParser(description="ML Preprocessor CLI")
    p.add_argument("pos_file", nargs="?", help="Dataset file path (positional)")
    p.add_argument("--file", "-f", help="Dataset file path (named)")
    p.add_argument("--sample", type=int, default=0, help="If >0, sample N rows for quick run")
    p.add_argument("--no-visual", action="store_true", help="Do not open visualization UI/plots")
    p.add_argument("--debug", action="store_true", help="Show full traceback on error")
    p.add_argument("--menu", "--interactive", dest="menu", action="store_true",
                   help="Show interactive menu (even if a file is provided)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        _print_banner()
        file_path = args.pos_file or args.file
        df = None

        if args.menu and file_path:
            df = read_file(file_path, sample_rows=args.sample if args.sample > 0 else None)
            interactive_menu_loop(df)
            return
        elif args.menu and not file_path:
            interactive_menu_loop(None)
            return
        elif file_path:
            df = read_file(file_path, sample_rows=args.sample if args.sample > 0 else None)
            interactive_menu_loop(df)
            return
        else:
            interactive_menu_loop(None)
            return

    except Exception as e:
        _echo(f"Error: {e}", style="red")
        if args.debug:
            traceback.print_exc()
        else:
            _echo("Run with --debug to see full traceback.", style="dim")
        sys.exit(1)


if __name__ == "__main__":
    main()
