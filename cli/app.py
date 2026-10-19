import argparse
import os
from typing import Dict, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.session import InferenceSession


PRESETS: Dict[str, str] = {
    "en": "Photosynthesis is the biological process by which green plants",
    "ko": "질문: 머신러닝에서 과적합(Overfitting)이 발생하는 원인과 해결 방법은 무엇인가요?\n답변:",
    "code": "def binary_search(arr: list[int], target: int) -> int:\n    low, high = 0, len(arr) - 1\n",
}

SLASH_COMMANDS: Dict[str, str] = {
    "/mode": "Switch prompt mode: /mode base | instruct | chat",
    "/model": "Hot-swap checkpoint or HF repo: /model <path_or_repo>",
    "/temp": "Set sampling temperature (0 = greedy): /temp <float>",
    "/top_p": "Set nucleus sampling top_p: /top_p <float>",
    "/max_tokens": "Set max generated tokens: /max_tokens <int>",
    "/rep": "Set repetition penalty: /rep <float>",
    "/system": "Update system prompt (instruct/chat modes): /system <text>",
    "/preset": "Run quick test prompt: /preset en | ko | code",
    "/clear": "Clear multi-turn conversation history",
    "/status": "Show current session configuration and metrics",
    "/help": "Show available slash commands",
    "/exit": "Exit Jerboa CLI",
}


def build_completer() -> NestedCompleter:
    return NestedCompleter.from_nested_dict(
        {
            "/mode": {"base": None, "instruct": None, "chat": None},
            "/model": {
                "checkpoints/pretrain/model": None,
                "checkpoints/sft/model": None,
                "ztor2/jerboa-base": None,
            },
            "/temp": {"0.0": None, "0.3": None, "0.7": None, "1.0": None},
            "/top_p": {"0.9": None, "0.95": None},
            "/max_tokens": {"128": None, "256": None, "512": None},
            "/rep": {"1.0": None, "1.15": None, "1.2": None},
            "/preset": {"en": None, "ko": None, "code": None},
            "/system": None,
            "/clear": None,
            "/status": None,
            "/help": None,
            "/exit": None,
        }
    )


def render_header(console: Console, session: InferenceSession) -> None:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(
        "[bold cyan]Jerboa CLI[/bold cyan]  [dim]Type /help for commands, /exit to quit[/dim]",
        f"[bold green]{session.num_params_m:.1f}M[/bold green] on [bold yellow]{session.device}[/bold yellow]",
    )
    grid.add_row(
        f"Model: [bold white]{session.model_path}[/bold white]",
        f"Mode: [bold magenta]{session.mode}[/bold magenta] | temp={session.temperature} | max_tokens={session.max_tokens}",
    )
    console.print(Panel(grid, border_style="cyan", padding=(0, 1)))


def render_status_table(console: Console, session: InferenceSession) -> None:
    table = Table(title="Session Status", border_style="dim", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="white")
    table.add_row("Model", session.model_path)
    table.add_row("Parameters", f"{session.num_params_m:.1f}M")
    table.add_row("Device", str(session.device))
    table.add_row("Mode", f"{session.mode} ({'raw continuation' if session.mode == 'base' else 'ChatML template'})")
    table.add_row("Sampling", f"temp={session.temperature}, top_p={session.top_p}, rep_penalty={session.repetition_penalty}")
    table.add_row("Max New Tokens", str(session.max_tokens))
    table.add_row("Chat Turns", str(len(session.history)))
    if session.last_stats.generated_tokens > 0:
        s = session.last_stats
        table.add_row(
            "Last Generation",
            f"{s.generated_tokens} tok in {s.elapsed_sec:.2f}s ({s.tokens_per_sec:.1f} tok/s, TTFT {s.ttft_ms:.0f}ms)",
        )
    console.print(table)


def render_help_table(console: Console) -> None:
    table = Table(title="Slash Commands", border_style="dim", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold green")
    table.add_column("Description", style="white")
    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)


def stream_and_display(console: Console, session: InferenceSession, user_input: str) -> None:
    buffer = ""
    title = f"[bold cyan]Jerboa ({session.mode})[/bold cyan]"
    with Live(
        Panel(Text("...", style="dim"), title=title, border_style="cyan"),
        console=console,
        refresh_per_second=16,
    ) as live:
        for chunk in session.stream_generate(user_input):
            buffer += chunk
            if session.mode == "base":
                display_text = Text()
                display_text.append(user_input, style="bold yellow")
                display_text.append(buffer)
                live.update(Panel(display_text, title=title, border_style="cyan"))
            else:
                renderable = Markdown(buffer) if "```" in buffer else Text(buffer)
                live.update(Panel(renderable, title=title, border_style="cyan"))

    stats = session.last_stats
    console.print(
        f"  [dim]prompt: {stats.prompt_tokens} tok | output: {stats.generated_tokens} tok | "
        f"TTFT: {stats.ttft_ms:.0f}ms | speed: {stats.tokens_per_sec:.1f} tok/s[/dim]\n"
    )


def handle_slash_command(console: Console, session: InferenceSession, raw: str) -> Optional[str]:
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        raise EOFError

    if cmd == "/help":
        render_help_table(console)
        return None

    if cmd == "/status":
        render_status_table(console, session)
        return None

    if cmd == "/clear":
        session.clear_history()
        console.print("[dim]Cleared conversation history.[/dim]")
        return None

    if cmd == "/mode":
        if arg not in ("base", "instruct", "chat"):
            console.print("[red]Usage: /mode base | instruct | chat[/red]")
            return None
        session.mode = arg
        console.print(f"[green]Mode set to:[/green] [bold]{session.mode}[/bold]")
        return None

    if cmd == "/model":
        if not arg:
            console.print("[red]Usage: /model <checkpoint_path_or_hf_repo>[/red]")
            return None
        try:
            console.print(f"[dim]Loading model '{arg}'...[/dim]")
            session.mode = "auto"
            session.load_model(arg)
            console.print(
                f"[green]Loaded:[/green] [bold]{session.model_path}[/bold] "
                f"(mode auto-set to [bold]{session.mode}[/bold])"
            )
        except Exception as exc:
            console.print(f"[red]Failed to load model: {exc}[/red]")
        return None

    if cmd == "/temp":
        try:
            session.temperature = max(0.0, float(arg))
            console.print(f"[green]Temperature set to:[/green] {session.temperature}")
        except ValueError:
            console.print("[red]Usage: /temp <float>[/red]")
        return None

    if cmd == "/top_p":
        try:
            session.top_p = min(1.0, max(0.01, float(arg)))
            console.print(f"[green]top_p set to:[/green] {session.top_p}")
        except ValueError:
            console.print("[red]Usage: /top_p <float>[/red]")
        return None

    if cmd == "/max_tokens":
        try:
            session.max_tokens = max(1, int(arg))
            console.print(f"[green]max_tokens set to:[/green] {session.max_tokens}")
        except ValueError:
            console.print("[red]Usage: /max_tokens <int>[/red]")
        return None

    if cmd == "/rep":
        try:
            session.repetition_penalty = max(1.0, float(arg))
            console.print(f"[green]repetition_penalty set to:[/green] {session.repetition_penalty}")
        except ValueError:
            console.print("[red]Usage: /rep <float>[/red]")
        return None

    if cmd == "/system":
        if not arg:
            console.print(f"[dim]Current system prompt:[/dim] {session.system_prompt}")
            return None
        session.system_prompt = arg
        console.print("[green]Updated system prompt.[/green]")
        return None

    if cmd == "/preset":
        if arg not in PRESETS:
            console.print(f"[red]Available presets: {', '.join(PRESETS.keys())}[/red]")
            return None
        preset_prompt = PRESETS[arg]
        console.print(f"[dim]Preset ({arg}):[/dim] {preset_prompt}")
        return preset_prompt

    console.print(f"[red]Unknown command '{cmd}'. Type /help for available commands.[/red]")
    return None


def run_cli(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Jerboa Interactive Terminal CLI")
    default_model = "checkpoints/pretrain/model" if os.path.exists("checkpoints/pretrain/model") else "ztor2/jerboa-base"
    parser.add_argument("--model", type=str, default=default_model, help="Model checkpoint path or HF Hub repo ID")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "base", "instruct", "chat"], help="Prompt mode")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt and exit")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top_p")
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--device", type=str, default=None, help="Device override (mps, cuda, cpu)")
    args = parser.parse_args(argv)

    console = Console()
    console.print(f"[dim]Initializing Jerboa session ({args.model})...[/dim]")
    session = InferenceSession(
        model_path=args.model,
        mode=args.mode,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        device_name=args.device,
    )

    if args.prompt is not None:
        stream_and_display(console, session, args.prompt)
        return

    render_header(console, session)

    pt_style = Style.from_dict(
        {
            "bottom-toolbar": "bg:#1e293b #e2e8f0",
            "prompt": "bold ansicyan",
        }
    )

    def bottom_toolbar():
        speed_str = f"{session.last_stats.tokens_per_sec:.1f} tok/s" if session.last_stats.generated_tokens else "ready"
        return HTML(
            f" <b>Model:</b> {session.model_path} | "
            f"<b>Mode:</b> {session.mode} | "
            f"<b>Temp:</b> {session.temperature} | "
            f"<b>Turns:</b> {len(session.history)} | "
            f"<b>Speed:</b> {speed_str} "
        )

    prompt_session = PromptSession(
        history=InMemoryHistory(),
        completer=build_completer(),
        bottom_toolbar=bottom_toolbar,
        style=pt_style,
    )

    while True:
        try:
            user_input = prompt_session.prompt([("class:prompt", f"[{session.mode}] > ")]).strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                maybe_prompt = handle_slash_command(console, session, user_input)
                if maybe_prompt is None:
                    continue
                user_input = maybe_prompt

            stream_and_display(console, session, user_input)
        except KeyboardInterrupt:
            console.print("[dim]Interrupted. Type /exit or Ctrl+D to quit.[/dim]")
        except EOFError:
            console.print("[dim]Exiting Jerboa CLI.[/dim]")
            break
