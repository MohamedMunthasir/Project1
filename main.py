# """
# main.py (visually improved)
# - Uses 'rich' for colored banners, prompts, tables, and spinners if available.
# - Falls back to plain text when 'rich' is not installed.
# - Preserves all menu/logic/undo/encoding features.
# """
# import argparse
# import sys
# import traceback
# from typing import Optional, List

# from data_input import read_file
# from data_analysis import print_summary
# from imputation import impute_dataframe
# from feature_scaling import scale_dataframe
# from visualization import DataVisualization  # improved visualization module

# # Try to import rich; provide safe fallback
# try:
#     from rich.console import Console
#     from rich.table import Table
#     from rich.prompt import Prompt
#     from rich.panel import Panel
#     from rich.spinner import Spinner
#     from rich.text import Text
#     RICH = True
#     console = Console()
# except Exception:
#     RICH = False
#     console = None


# UNDO_STACK_MAX = 5


# def _echo(msg: str, style: Optional[str] = None):
#     if RICH:
#         if style:
#             console.print(msg, style=style)
#         else:
#             console.print(msg)
#     else:
#         print(msg)


# def _print_banner():
#     if RICH:
#         txt = Text("ML-Preprocessor-CLI", style="bold white on blue")
#         console.print(Panel(txt, expand=False))
#         console.print("A friendly CLI to preprocess and visualize tabular data\n", style="dim")
#     else:
#         print("=" * 40)
#         print("ML-Preprocessor-CLI")
#         print("=" * 40)


# def _print_menu():
#     if RICH:
#         table = Table(show_header=False, box=None)
#         table.add_row("[bold cyan]1.[/]", "Data Description")
#         table.add_row("[bold cyan]2.[/]", "Handling NULL Values (Impute)")
#         table.add_row("[bold cyan]3.[/]", "Encoding Categorical Data")
#         table.add_row("[bold cyan]4.[/]", "Feature Scaling of the Dataset")
#         table.add_row("[bold cyan]5.[/]", "Download the modified dataset")
#         table.add_row("[bold cyan]6.[/]", "Visualize the Dataset")
#         table.add_row("[bold cyan]7.[/]", "Undo last mutation (encoding/imputation/scaling)")
#         table.add_row("[bold red]-1.[/]", "Exit")
#         console.print(Panel(table, title="Tasks (Preprocessing)👇", expand=False))
#     else:
#         print("\nTasks (Preprocessing)👇\n")
#         print("1. Data Description")
#         print("2. Handling NULL Values")
#         print("3. Encoding Categorical Data")
#         print("4. Feature Scaling of the Dataset")
#         print("5. Download the modified dataset")
#         print("6. Visualize the Dataset")
#         print("7. Undo last mutation (encoding/imputation/scaling)")
#         print("Press -1 to exit\n")


# def _select_columns_interactive(available: List[str]) -> List[str]:
#     """
#     Same selection helper but with rich-friendly prompts.
#     """
#     if RICH:
#         console.print("\nAvailable columns:", style="bold")
#         for i, c in enumerate(available, 1):
#             console.print(f"[cyan]{i}.[/] {c}")
#         raw = Prompt.ask("Enter column numbers or names (comma-separated). Leave blank to select all", default="")
#     else:
#         print("Available columns:")
#         for i, c in enumerate(available, 1):
#             print(f"{i}. {c}")
#         raw = input("Enter column numbers or names (comma-separated). Leave blank to select all: ").strip()

#     if not raw:
#         return available[:]

#     tokens = [t.strip() for t in raw.split(",") if t.strip()]
#     selected = []
#     for t in tokens:
#         if t.isdigit():
#             idx = int(t) - 1
#             if 0 <= idx < len(available):
#                 selected.append(available[idx])
#             else:
#                 _echo(f"Ignoring out-of-range index: {t}", style="yellow" if RICH else None)
#         else:
#             matches = [c for c in available if c.lower() == t.lower()]
#             if matches:
#                 selected.append(matches[0])
#             else:
#                 _echo(f"Ignoring unknown column name: {t}", style="yellow" if RICH else None)
#     # dedupe preserving order
#     final = []
#     seen = set()
#     for c in selected:
#         if c not in seen:
#             final.append(c)
#             seen.add(c)
#     return final


# def interactive_menu_loop(df: Optional[object] = None):
#     undo_stack = []

#     def push_undo(snapshot):
#         if snapshot is None:
#             return
#         undo_stack.append(snapshot.copy())
#         if len(undo_stack) > UNDO_STACK_MAX:
#             undo_stack.pop(0)

#     def do_undo():
#         nonlocal df
#         if not undo_stack:
#             _echo("Nothing to undo.", style="yellow" if RICH else None)
#             return df
#         df = undo_stack.pop()
#         _echo("Reverted to previous dataset state.", style="green" if RICH else None)
#         return df

#     while True:
#         _print_menu()
#         if RICH:
#             choice = Prompt.ask("What do you want to do? (Press -1 to exit)", default="")
#         else:
#             choice = input("What do you want to do? (Press -1 to exit): ").strip()

#         if choice == "-1":
#             _echo("Exiting. Goodbye 👋", style="bold" if RICH else None)
#             break

#         if df is None and choice not in ("-1",):
#             if RICH:
#                 path = Prompt.ask("No dataset loaded. Enter path to dataset file")
#             else:
#                 path = input("No dataset loaded. Enter path to dataset file: ").strip()
#             try:
#                 df = read_file(path)
#                 _echo("Dataset loaded.", style="green" if RICH else None)
#             except Exception as e:
#                 _echo(f"Failed to load file: {e}", style="red" if RICH else None)
#                 df = None
#                 continue

#         if choice == "1":
#             _echo("Data summary:", style="bold")
#             print_summary(df, sample_rows=100)
#         elif choice == "2":
#             push_undo(df)
#             _echo("Handling NULL values (imputation) — this may take a moment...", style="bold")
#             if RICH:
#                 with console.status("[bold green]Imputing...[/]"):
#                     df = impute_dataframe(df)
#             else:
#                 df = impute_dataframe(df)
#             _echo("Imputation completed.", style="green")
#         elif choice == "3":
#             from encoding import label_encode, one_hot_encode, detect_categorical_columns

#             cat_cols = detect_categorical_columns(df)
#             if not cat_cols:
#                 _echo("No categorical columns found to encode.", style="yellow")
#                 continue

#             if RICH:
#                 console.print("Categorical columns detected:", style="bold")
#                 for i, c in enumerate(cat_cols, 1):
#                     console.print(f"[cyan]{i}.[/] {c}")
#             else:
#                 print("Categorical columns detected:", cat_cols)

#             selected = _select_columns_interactive(cat_cols)
#             if not selected:
#                 _echo("No columns selected. Aborting encoding.", style="yellow")
#                 continue

#             if RICH:
#                 enc_choice = Prompt.ask("Choose encoding: 1) Label Encoding  2) One-Hot Encoding [default=1]", default="1")
#             else:
#                 enc_choice = input("Choose encoding: 1) Label Encoding  2) One-Hot Encoding [default=1]: ").strip()

#             push_undo(df)
#             if enc_choice == "2":
#                 _echo("Applying one-hot encoding...", style="bold")
#                 if RICH:
#                     with console.status("[bold green]Encoding...[/]"):
#                         df = one_hot_encode(df, selected)
#                 else:
#                     df = one_hot_encode(df, selected)
#                 _echo(f"One-hot encoding applied to: {selected}", style="green")
#             else:
#                 _echo("Applying label encoding...", style="bold")
#                 if RICH:
#                     with console.status("[bold green]Encoding...[/]"):
#                         df = label_encode(df, selected)
#                 else:
#                     df = label_encode(df, selected)
#                 _echo(f"Label encoding applied to: {selected}", style="green")
#         elif choice == "4":
#             numeric_cols = [c for c in df.columns if df[c].dtype.kind in "bifc"]
#             if not numeric_cols:
#                 _echo("No numeric columns found to scale.", style="yellow")
#             else:
#                 _echo(f"Numeric columns detected: {numeric_cols}", style="bold")
#                 if RICH:
#                     col_choice = Prompt.ask("Scale all numeric columns? (y/n) [default=y]", default="y")
#                 else:
#                     col_choice = input("Scale all numeric columns? (y/n) [default=y]: ").strip().lower()
#                 if str(col_choice).lower() in ("n", "no"):
#                     selected = _select_columns_interactive(numeric_cols)
#                 else:
#                     selected = numeric_cols
#                 if not selected:
#                     _echo("No numeric columns selected.", style="yellow")
#                     continue
#                 if RICH:
#                     method = Prompt.ask("Choose scaler: 1) standard 2) minmax 3) robust [default=1]", default="1")
#                 else:
#                     method = input("Choose scaler: 1) standard 2) minmax 3) robust [default=1]: ").strip()
#                 method_map = {"1": "standard", "2": "minmax", "3": "robust"}
#                 method_choice = method_map.get(method, "standard")
#                 push_undo(df)
#                 _echo("Applying scaling...", style="bold")
#                 if RICH:
#                     with console.status("[bold green]Scaling...[/]"):
#                         df, _ = scale_dataframe(df, selected, method=method_choice)
#                 else:
#                     df, _ = scale_dataframe(df, selected, method=method_choice)
#                 _echo(f"Scaling applied to: {selected}", style="green")
#         elif choice == "5":
#             if RICH:
#                 out_path = Prompt.ask("Enter output CSV filename to save (example: out.csv)", default="output.csv")
#             else:
#                 out_path = input("Enter output CSV filename to save (example: out.csv): ").strip() or "output.csv"
#             try:
#                 df.to_csv(out_path, index=False)
#                 _echo(f"Saved to {out_path}", style="green")
#             except Exception as e:
#                 _echo(f"Failed to save: {e}", style="red")
#         elif choice == "6":
#             vis = DataVisualization(df)
#             vis.run_visualization()
#         elif choice == "7":
#             df = do_undo()
#         else:
#             _echo("Invalid choice. Try again.", style="yellow")


# def quick_run_with_file(file_path: str, sample: int, no_visual: bool):
#     _echo("Loading file...", style="bold")
#     df = read_file(file_path, sample_rows=sample if sample > 0 else None)
#     _echo("Data loaded. Basic summary:", style="bold")
#     print_summary(df, sample_rows=100)

#     if df.isna().sum().sum() > 0:
#         _echo("Missing values detected. Applying default imputation...", style="bold")
#         if RICH:
#             with console.status("[bold green]Imputing...[/]"):
#                 df = impute_dataframe(df)
#         else:
#             df = impute_dataframe(df)

#     if RICH:
#         choice = Prompt.ask("Would you like to scale numeric columns? (y/n)", default="n")
#     else:
#         choice = input("\nWould you like to scale numeric columns? (y/n): ").strip().lower()

#     if choice and choice.lower() in ("y", "yes"):
#         numeric_cols = [c for c in df.columns if df[c].dtype.kind in "bifc"]
#         if not numeric_cols:
#             _echo("No numeric columns found to scale.", style="yellow")
#         else:
#             _echo(f"Numeric columns detected: {numeric_cols}", style="bold")
#             if RICH:
#                 method = Prompt.ask("Choose scaler: 1) standard 2) minmax 3) robust [default=1]", default="1")
#             else:
#                 method = input("Choose scaler: 1) standard 2) minmax 3) robust [default=1]: ").strip()
#             method_map = {"1": "standard", "2": "minmax", "3": "robust"}
#             method_choice = method_map.get(method, "standard")
#             _echo("Applying scaling...", style="bold")
#             if RICH:
#                 with console.status("[bold green]Scaling...[/]"):
#                     df, _scaler = scale_dataframe(df, numeric_cols, method=method_choice)
#             else:
#                 df, _scaler = scale_dataframe(df, numeric_cols, method=method_choice)
#             _echo("Scaling applied.", style="green")

#     if not no_visual:
#         vis = DataVisualization(df)
#         vis.run_visualization()
#     else:
#         _echo("Skipping visualization (--no-visual).", style="dim")


# def parse_args():
#     p = argparse.ArgumentParser(description="ML Preprocessor CLI")
#     p.add_argument("pos_file", nargs="?", help="Dataset file path (positional)")
#     p.add_argument("--file", "-f", help="Dataset file path (named)")
#     p.add_argument("--sample", type=int, default=0, help="If >0, sample N rows for quick run")
#     p.add_argument("--no-visual", action="store_true", help="Do not open visualization UI/plots")
#     p.add_argument("--debug", action="store_true", help="Show full traceback on error")
#     p.add_argument("--menu", "--interactive", dest="menu", action="store_true",
#                    help="Show interactive menu (even if a file is provided)")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     try:
#         _print_banner()
#         file_path = args.pos_file or args.file
#         df = None

#         if args.menu and file_path:
#             df = read_file(file_path, sample_rows=args.sample if args.sample > 0 else None)
#             interactive_menu_loop(df)
#             return
#         elif args.menu and not file_path:
#             interactive_menu_loop(None)
#             return
#         elif file_path:
#             df = read_file(file_path, sample_rows=args.sample if args.sample > 0 else None)
#             interactive_menu_loop(df)
#             return
#         else:
#             interactive_menu_loop(None)
#             return

#     except Exception as e:
#         _echo(f"Error: {e}", style="red")
#         if args.debug:
#             traceback.print_exc()
#         else:
#             _echo("Run with --debug to see full traceback.", style="dim")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


"""
main.py (IMPROVED VERSION with guided workflow)
Enhanced with preprocessing wizard, quality checks, and better UX
"""
import argparse
import sys
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from data_input import read_file
from data_analysis import print_summary
from imputation import impute_dataframe
from feature_scaling import scale_dataframe
from visualization import DataVisualization
from outlier_detection import OutlierDetector
from data_quality import DataQualityAnalyzer

try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

UNDO_STACK_MAX = 10


class PreprocessingState:
    """Track preprocessing state and history"""
    def __init__(self, df: pd.DataFrame):
        self.current_df = df
        self.original_df = df.copy()
        self.history = []
        self.undo_stack = []
    
    def save_state(self, action: str):
        """Save current state before modification"""
        if len(self.undo_stack) >= UNDO_STACK_MAX:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.current_df.copy())
        self.history.append({
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'shape': self.current_df.shape
        })
    
    def undo(self):
        """Revert to previous state"""
        if not self.undo_stack:
            return False
        self.current_df = self.undo_stack.pop()
        if self.history:
            self.history.pop()
        return True
    
    def get_changes_summary(self):
        """Get summary of all changes made"""
        return {
            'original_shape': self.original_df.shape,
            'current_shape': self.current_df.shape,
            'rows_removed': self.original_df.shape[0] - self.current_df.shape[0],
            'columns_added': self.current_df.shape[1] - self.original_df.shape[1],
            'actions_performed': len(self.history),
            'history': self.history
        }


def _echo(msg: str, style: Optional[str] = None):
    if RICH:
        console.print(msg, style=style) if style else console.print(msg)
    else:
        print(msg)


def _print_banner():
    if RICH:
        txt = Text("🚀 ML-Preprocessor-CLI v2.0", style="bold white on blue")
        console.print(Panel(txt, expand=False))
        console.print("Production-grade data preprocessing for machine learning\n", style="dim")
    else:
        print("="*50)
        print("🚀 ML-Preprocessor-CLI v2.0")
        print("="*50)


def _print_main_menu():
    if RICH:
        table = Table(show_header=False, box=None)
        table.add_row("[bold cyan]1.[/]", "🔍 Data Quality Report")
        table.add_row("[bold cyan]2.[/]", "📊 Data Description & Summary")
        table.add_row("[bold cyan]3.[/]", "🧹 Handle Missing Values")
        table.add_row("[bold cyan]4.[/]", "🎯 Detect & Handle Outliers")
        table.add_row("[bold cyan]5.[/]", "🔄 Remove Duplicate Rows")
        table.add_row("[bold cyan]6.[/]", "🏷️  Encode Categorical Features")
        table.add_row("[bold cyan]7.[/]", "⚖️  Scale Numerical Features")
        table.add_row("[bold cyan]8.[/]", "📈 Visualize Dataset")
        table.add_row("[bold cyan]9.[/]", "💾 Save Processed Dataset")
        table.add_row("[bold cyan]10.[/]", "↩️  Undo Last Action")
        table.add_row("[bold cyan]11.[/]", "📝 View Processing History")
        table.add_row("[bold cyan]12.[/]", "🧙 Guided Preprocessing Wizard")
        table.add_row("[bold red]0.[/]", "Exit")
        console.print(Panel(table, title="Main Menu", expand=False))
    else:
        print("\n=== Main Menu ===")
        print("1. 🔍 Data Quality Report")
        print("2. 📊 Data Description & Summary")
        print("3. 🧹 Handle Missing Values")
        print("4. 🎯 Detect & Handle Outliers")
        print("5. 🔄 Remove Duplicate Rows")
        print("6. 🏷️  Encode Categorical Features")
        print("7. ⚖️  Scale Numerical Features")
        print("8. 📈 Visualize Dataset")
        print("9. 💾 Save Processed Dataset")
        print("10. ↩️  Undo Last Action")
        print("11. 📝 View Processing History")
        print("12. 🧙 Guided Preprocessing Wizard")
        print("0. Exit")


def guided_preprocessing_wizard(state: PreprocessingState):
    """Step-by-step guided preprocessing workflow"""
    _echo("\n🧙 GUIDED PREPROCESSING WIZARD", style="bold magenta")
    _echo("I'll guide you through essential preprocessing steps.\n")
    
    df = state.current_df
    
    # Step 1: Quality Assessment
    _echo("Step 1/6: Data Quality Assessment", style="bold")
    qa = DataQualityAnalyzer(df)
    qa.generate_full_report()
    qa.print_summary()
    
    # Step 2: Handle Duplicates
    _echo("\nStep 2/6: Duplicate Detection", style="bold")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        _echo(f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.2f}%)", style="yellow")
        if Confirm.ask("Remove duplicates?", default=True) if RICH else input("Remove duplicates? (y/n): ").lower() == 'y':
            state.save_state("Remove duplicates")
            state.current_df = df.drop_duplicates().reset_index(drop=True)
            _echo(f"✓ Removed {dup_count} duplicates", style="green")
    else:
        _echo("✓ No duplicates found", style="green")
    
    df = state.current_df
    
    # Step 3: Handle Missing Values
    _echo("\nStep 3/6: Missing Value Treatment", style="bold")
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        _echo(f"Found {missing_count} missing values", style="yellow")
        if Confirm.ask("Apply automatic imputation?", default=True) if RICH else input("Apply imputation? (y/n): ").lower() == 'y':
            state.save_state("Impute missing values")
            state.current_df = impute_dataframe(df)
            _echo("✓ Missing values imputed", style="green")
    else:
        _echo("✓ No missing values", style="green")
    
    df = state.current_df
    
    # Step 4: Outlier Detection
    _echo("\nStep 4/6: Outlier Detection", style="bold")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        detector = OutlierDetector(df)
        outliers = detector.detect_iqr(numeric_cols)
        total_outliers = sum(info['count'] for info in outliers.values())
        
        if total_outliers > 0:
            _echo(f"Found {total_outliers} outlier values across {len(outliers)} columns", style="yellow")
            action = Prompt.ask("How to handle outliers?", choices=["remove", "cap", "skip"], default="cap") if RICH else input("Handle outliers (remove/cap/skip): ")
            
            if action == "remove":
                state.save_state("Remove outliers")
                state.current_df = detector.remove_outliers()
                _echo("✓ Outliers removed", style="green")
            elif action == "cap":
                state.save_state("Cap outliers")
                state.current_df = detector.cap_outliers()
                _echo("✓ Outliers capped", style="green")
        else:
            _echo("✓ No significant outliers detected", style="green")
    
    df = state.current_df
    
    # Step 5: Encoding
    _echo("\nStep 5/6: Categorical Encoding", style="bold")
    from encoding import detect_categorical_columns, one_hot_encode
    cat_cols = detect_categorical_columns(df)
    if cat_cols:
        _echo(f"Found {len(cat_cols)} categorical columns: {cat_cols}", style="yellow")
        if Confirm.ask("Apply one-hot encoding?", default=True) if RICH else input("Apply encoding? (y/n): ").lower() == 'y':
            state.save_state("One-hot encode categorical")
            state.current_df = one_hot_encode(df, cat_cols)
            _echo(f"✓ Encoded {len(cat_cols)} columns", style="green")
    else:
        _echo("✓ No categorical columns to encode", style="green")
    
    df = state.current_df
    
    # Step 6: Scaling
    _echo("\nStep 6/6: Feature Scaling", style="bold")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        if Confirm.ask("Apply standard scaling to numeric features?", default=True) if RICH else input("Apply scaling? (y/n): ").lower() == 'y':
            state.save_state("Standard scaling")
            state.current_df, _ = scale_dataframe(df, numeric_cols, method="standard")
            _echo(f"✓ Scaled {len(numeric_cols)} numeric columns", style="green")
    
    _echo("\n✨ Preprocessing Complete!", style="bold green")
    _echo(f"Original shape: {state.original_df.shape} → Current shape: {state.current_df.shape}")
    
    return state


def interactive_menu_loop(initial_df: Optional[pd.DataFrame] = None):
    """Main interactive menu loop"""
    
    state = PreprocessingState(initial_df) if initial_df is not None else None
    
    while True:
        _print_main_menu()
        
        choice = Prompt.ask("Select option", default="0") if RICH else input("Select option: ").strip()
        
        if choice == "0":
            if state:
                changes = state.get_changes_summary()
                _echo(f"\nSummary: {changes['actions_performed']} actions performed", style="bold")
                if changes['rows_removed'] != 0 or changes['columns_added'] != 0:
                    _echo(f"Shape: {changes['original_shape']} → {changes['current_shape']}")
            _echo("Goodbye! 👋", style="bold")
            break
        
        # Load dataset if not loaded
        if state is None and choice not in ("0",):
            path = Prompt.ask("Enter path to dataset file") if RICH else input("Enter dataset path: ").strip()
            try:
                df = read_file(path)
                state = PreprocessingState(df)
                _echo(f"✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns", style="green")
            except Exception as e:
                _echo(f"Error loading file: {e}", style="red")
                continue
        
        df = state.current_df if state else None
        
        if choice == "1":  # Quality Report
            qa = DataQualityAnalyzer(df)
            qa.generate_full_report()
            qa.print_summary()
        
        elif choice == "2":  # Data Description
            print_summary(df)
        
        elif choice == "3":  # Missing Values
            missing = df.isna().sum().sum()
            if missing > 0:
                _echo(f"Found {missing} missing values", style="yellow")
                if Confirm.ask("Apply imputation?") if RICH else input("Apply imputation? (y/n): ").lower() == 'y':
                    state.save_state("Imputation")
                    state.current_df = impute_dataframe(df)
                    _echo("✓ Imputation completed", style="green")
            else:
                _echo("No missing values found", style="green")
        
        elif choice == "4":  # Outliers
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                _echo("No numeric columns found", style="yellow")
                continue
            
            detector = OutlierDetector(df)
            method = Prompt.ask("Detection method", choices=["iqr", "zscore"], default="iqr") if RICH else "iqr"
            
            if method == "iqr":
                outliers = detector.detect_iqr(numeric_cols)
            else:
                outliers = detector.detect_zscore(numeric_cols)
            
            detector.print_summary(method)
            
            action = Prompt.ask("Action", choices=["remove", "cap", "skip"], default="skip") if RICH else input("Action (remove/cap/skip): ")
            if action == "remove":
                state.save_state("Remove outliers")
                state.current_df = detector.remove_outliers(method)
            elif action == "cap":
                state.save_state("Cap outliers")
                state.current_df = detector.cap_outliers()
        
        elif choice == "5":  # Duplicates
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                _echo(f"Found {dup_count} duplicate rows", style="yellow")
                if Confirm.ask("Remove?") if RICH else input("Remove? (y/n): ").lower() == 'y':
                    state.save_state("Remove duplicates")
                    state.current_df = df.drop_duplicates().reset_index(drop=True)
                    _echo(f"✓ Removed {dup_count} rows", style="green")
            else:
                _echo("No duplicates", style="green")
        
        elif choice == "6":  # Encoding
            from encoding import detect_categorical_columns, label_encode, one_hot_encode
            cat_cols = detect_categorical_columns(df)
            if not cat_cols:
                _echo("No categorical columns", style="yellow")
                continue
            
            _echo(f"Categorical columns: {cat_cols}")
            enc_type = Prompt.ask("Encoding", choices=["label", "onehot"], default="onehot") if RICH else input("Type (label/onehot): ")
            
            state.save_state(f"{enc_type} encoding")
            if enc_type == "label":
                state.current_df = label_encode(df, cat_cols)
            else:
                state.current_df = one_hot_encode(df, cat_cols)
            _echo("✓ Encoding completed", style="green")
        
        elif choice == "7":  # Scaling
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                _echo("No numeric columns", style="yellow")
                continue
            
            method = Prompt.ask("Scaler", choices=["standard", "minmax", "robust"], default="standard") if RICH else "standard"
            state.save_state(f"{method} scaling")
            state.current_df, _ = scale_dataframe(df, numeric_cols, method=method)
            _echo(f"✓ Scaled {len(numeric_cols)} columns", style="green")
        
        elif choice == "8":  # Visualization
            vis = DataVisualization(df)
            vis.run_visualization()
        
        elif choice == "9":  # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"preprocessed_data_{timestamp}.csv"
            out_path = Prompt.ask("Output filename", default=default_name) if RICH else input(f"Output filename [{default_name}]: ") or default_name
            
            try:
                df.to_csv(out_path, index=False)
                _echo(f"✓ Saved to {out_path} ({df.shape[0]} rows × {df.shape[1]} cols)", style="green")
            except Exception as e:
                _echo(f"Error saving: {e}", style="red")
        
        elif choice == "10":  # Undo
            if state.undo():
                _echo("✓ Reverted to previous state", style="green")
            else:
                _echo("Nothing to undo", style="yellow")
        
        elif choice == "11":  # History
            changes = state.get_changes_summary()
            _echo(f"\n{'='*60}", style="bold")
            _echo("PREPROCESSING HISTORY", style="bold")
            _echo(f"{'='*60}")
            _echo(f"Original Shape: {changes['original_shape']}")
            _echo(f"Current Shape: {changes['current_shape']}")
            _echo(f"Actions Performed: {changes['actions_performed']}\n")
            for i, action in enumerate(changes['history'], 1):
                _echo(f"{i}. {action['action']} - Shape: {action['shape']}")
        
        elif choice == "12":  # Wizard
            state = guided_preprocessing_wizard(state)
        
        else:
            _echo("Invalid choice", style="yellow")


def main():
    parser = argparse.ArgumentParser(description="ML-Preprocessor-CLI v2.0")
    parser.add_argument("file", nargs="?", help="Dataset file path")
    parser.add_argument("--wizard", "-w", action="store_true", help="Start with guided wizard")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()
    
    try:
        _print_banner()
        
        df = None
        if args.file:
            df = read_file(args.file)
            _echo(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns", style="green")
        
        if args.wizard and df is not None:
            state = PreprocessingState(df)
            state = guided_preprocessing_wizard(state)
            if args.output:
                state.current_df.to_csv(args.output, index=False)
                _echo(f"✓ Saved to {args.output}", style="green")
        else:
            interactive_menu_loop(df)
    
    except Exception as e:
        _echo(f"Error: {e}", style="red")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
