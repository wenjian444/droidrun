"""
DroidRun CLI - Command line interface for controlling Android devices through LLM agents.
"""

import asyncio
import click
import os
import logging
import warnings
from contextlib import nullcontext
from rich.console import Console
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm
from droidrun.adb import DeviceManager
from droidrun.tools import AdbTools, IOSTools
from functools import wraps
from droidrun.cli.logs import LogHandler
from droidrun.telemetry import print_telemetry_message
from droidrun.portal import (
    download_portal_apk,
    enable_portal_accessibility,
    PORTAL_PACKAGE_NAME,
    ping_portal,
)

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

console = Console()
device_manager = DeviceManager()


def configure_logging(goal: str, debug: bool):
    logger = logging.getLogger("droidrun")
    logger.handlers = []

    handler = LogHandler(goal)
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(message)s", "%H:%M:%S")
        if debug
        else logging.Formatter("%(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    return handler


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@coro
async def run_command(
    command: str,
    device: str | None,
    provider: str,
    model: str,
    steps: int,
    base_url: str,
    api_base: str,
    vision: bool,
    reasoning: bool,
    reflection: bool,
    tracing: bool,
    debug: bool,
    save_trajectory: bool = False,
    ios: bool = False,
    **kwargs,
):
    """Run a command on your Android device using natural language."""
    log_handler = configure_logging(command, debug)
    logger = logging.getLogger("droidrun")

    log_handler.update_step("Initializing...")

    with log_handler.render() as live:
        try:
            logger.info(f"🚀 Starting: {command}")
            print_telemetry_message()

            if not kwargs.get("temperature"):
                kwargs["temperature"] = 0

            log_handler.update_step("Setting up tools...")

            # Device setup
            if device is None and not ios:
                logger.info("🔍 Finding connected device...")
                device_manager = DeviceManager()
                devices = await device_manager.list_devices()
                if not devices:
                    raise ValueError("No connected devices found.")
                device = devices[0].serial
                logger.info(f"📱 Using device: {device}")
            elif device is None and ios:
                raise ValueError(
                    "iOS device not specified. Please specify the device base url (http://device-ip:6643) via --device"
                )
            else:
                logger.info(f"📱 Using device: {device}")

            tools = AdbTools(serial=device) if not ios else IOSTools(url=device)

            # LLM setup
            log_handler.update_step("Initializing LLM...")
            llm = load_llm(
                provider_name=provider,
                model=model,
                base_url=base_url,
                api_base=api_base,
                **kwargs,
            )
            logger.info(f"🧠 LLM ready: {provider}/{model}")

            # Agent setup
            log_handler.update_step("Initializing DroidAgent...")

            mode = "planning with reasoning" if reasoning else "direct execution"
            logger.info(f"🤖 Agent mode: {mode}")

            if tracing:
                logger.info("🔍 Tracing enabled")

            droid_agent = DroidAgent(
                goal=command,
                llm=llm,
                tools=tools,
                max_steps=steps,
                timeout=1000,
                vision=vision,
                reasoning=reasoning,
                reflection=reflection,
                enable_tracing=tracing,
                debug=debug,
                save_trajectories=save_trajectory,
            )

            logger.info("▶️  Starting agent execution...")
            logger.info("Press Ctrl+C to stop")
            log_handler.update_step("Running agent...")

            try:
                handler = droid_agent.run()

                async for event in handler.stream_events():
                    log_handler.handle_event(event)
                result = await handler

            except KeyboardInterrupt:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = "Stopped by user"
                logger.info("⏹️ Stopped by user")

            except Exception as e:
                log_handler.is_completed = True
                log_handler.is_success = False
                log_handler.current_step = f"Error: {e}"
                logger.error(f"💥 Error: {e}")
                if debug:
                    import traceback

                    logger.debug(traceback.format_exc())

        except Exception as e:
            log_handler.current_step = f"Error: {e}"
            logger.error(f"💥 Setup error: {e}")
            if debug:
                import traceback

                logger.debug(traceback.format_exc())


class DroidRunCLI(click.Group):
    def parse_args(self, ctx, args):
        # If the first arg is not an option and not a known command, treat as 'run'
        if args and """not args[0].startswith("-")""" and args[0] not in self.commands:
            args.insert(0, "run")

        return super().parse_args(ctx, args)


@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--provider",
    "-p",
    help="LLM provider (OpenAI, Ollama, Anthropic, GoogleGenAI, DeepSeek)",
    default="GoogleGenAI",
)
@click.option(
    "--model",
    "-m",
    help="LLM model name",
    default="models/gemini-2.5-flash",
)
@click.option("--temperature", type=float, help="Temperature for LLM", default=0.2)
@click.option("--steps", type=int, help="Maximum number of steps", default=15)
@click.option(
    "--base_url",
    "-u",
    help="Base URL for API (e.g., OpenRouter or Ollama)",
    default=None,
)
@click.option(
    "--api_base",
    help="Base URL for API (e.g., OpenAI, OpenAI-Like)",
    default=None,
)
@click.option(
    "--vision",
    is_flag=True,
    help="Enable vision capabilites by using screenshots",
    default=False,
)
@click.option(
    "--reasoning", is_flag=True, help="Enable planning with reasoning", default=False
)
@click.option(
    "--reflection",
    is_flag=True,
    help="Enable reflection step for higher reasoning",
    default=False,
)
@click.option(
    "--tracing", is_flag=True, help="Enable Arize Phoenix tracing", default=False
)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@click.option(
    "--save-trajectory",
    is_flag=True,
    help="Save agent trajectory to file",
    default=False,
)
@click.group(cls=DroidRunCLI)
def cli(
    device: str | None,
    provider: str,
    model: str,
    steps: int,
    base_url: str,
    api_base: str,
    temperature: float,
    vision: bool,
    reasoning: bool,
    reflection: bool,
    tracing: bool,
    debug: bool,
    save_trajectory: bool,
):
    """DroidRun - Control your Android device through LLM agents."""
    pass


@cli.command()
@click.argument("command", type=str)
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--provider",
    "-p",
    help="LLM provider (OpenAI, Ollama, Anthropic, GoogleGenAI, DeepSeek)",
    default="GoogleGenAI",
)
@click.option(
    "--model",
    "-m",
    help="LLM model name",
    default="models/gemini-2.5-flash",
)
@click.option("--temperature", type=float, help="Temperature for LLM", default=0.2)
@click.option("--steps", type=int, help="Maximum number of steps", default=15)
@click.option(
    "--base_url",
    "-u",
    help="Base URL for API (e.g., OpenRouter or Ollama)",
    default=None,
)
@click.option(
    "--api_base",
    help="Base URL for API (e.g., OpenAI or OpenAI-Like)",
    default=None,
)
@click.option(
    "--vision",
    is_flag=True,
    help="Enable vision capabilites by using screenshots",
    default=False,
)
@click.option(
    "--reasoning", is_flag=True, help="Enable planning with reasoning", default=False
)
@click.option(
    "--reflection",
    is_flag=True,
    help="Enable reflection step for higher reasoning",
    default=False,
)
@click.option(
    "--tracing", is_flag=True, help="Enable Arize Phoenix tracing", default=False
)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@click.option(
    "--save-trajectory",
    is_flag=True,
    help="Save agent trajectory to file",
    default=False,
)
@click.option("--ios", is_flag=True, help="Run on iOS device", default=False)
def run(
    command: str,
    device: str | None,
    provider: str,
    model: str,
    steps: int,
    base_url: str,
    api_base: str,
    temperature: float,
    vision: bool,
    reasoning: bool,
    reflection: bool,
    tracing: bool,
    debug: bool,
    save_trajectory: bool,
    ios: bool,
):
    """Run a command on your Android device using natural language."""
    # Call our standalone function
    return run_command(
        command,
        device,
        provider,
        model,
        steps,
        base_url,
        api_base,
        vision,
        reasoning,
        reflection,
        tracing,
        debug,
        temperature=temperature,
        save_trajectory=save_trajectory,
        ios=ios,
    )


@cli.command()
@coro
async def devices():
    """List connected Android devices."""
    try:
        devices = await device_manager.list_devices()
        if not devices:
            console.print("[yellow]No devices connected.[/]")
            return

        console.print(f"[green]Found {len(devices)} connected device(s):[/]")
        for device in devices:
            console.print(f"  • [bold]{device.serial}[/]")
    except Exception as e:
        console.print(f"[red]Error listing devices: {e}[/]")


@cli.command()
@click.argument("serial")
@click.option("--port", "-p", default=5555, help="ADB port (default: 5555)")
@coro
async def connect(serial: str, port: int):
    """Connect to a device over TCP/IP."""
    try:
        device = await device_manager.connect(serial, port)
        if device:
            console.print(f"[green]Successfully connected to {serial}:{port}[/]")
        else:
            console.print(f"[red]Failed to connect to {serial}:{port}[/]")
    except Exception as e:
        console.print(f"[red]Error connecting to device: {e}[/]")


@cli.command()
@click.argument("serial")
@coro
async def disconnect(serial: str):
    """Disconnect from a device."""
    try:
        success = await device_manager.disconnect(serial)
        if success:
            console.print(f"[green]Successfully disconnected from {serial}[/]")
        else:
            console.print(f"[yellow]Device {serial} was not connected[/]")
    except Exception as e:
        console.print(f"[red]Error disconnecting from device: {e}[/]")


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option("--path", help="Path to the Droidrun Portal APK to install on the device. If not provided, the latest portal apk version will be downloaded and installed.", default=None)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@coro
async def setup(path: str | None, device: str | None, debug: bool):
    """Install and enable the DroidRun Portal on a device."""
    try:
        if not device:
            devices = await device_manager.list_devices()
            if not devices:
                console.print("[yellow]No devices connected.[/]")
                return

            device = devices[0].serial
            console.print(f"[blue]Using device:[/] {device}")

        device_obj = await device_manager.get_device(device)
        if not device_obj:
            console.print(
                f"[bold red]Error:[/] Could not get device object for {device}"
            )
            return

        if not path:
            console.print("[bold blue]Downloading DroidRun Portal APK...[/]")
            apk_context = download_portal_apk(debug)
        else:
            console.print(f"[bold blue]Using provided APK:[/] {path}")
            apk_context = nullcontext(path)

        with apk_context as apk_path:
            if not os.path.exists(apk_path):
                console.print(f"[bold red]Error:[/] APK file not found at {apk_path}")
                return

            console.print(f"[bold blue]Step 1/2: Installing APK:[/] {apk_path}")
            result = await device_obj.install_app(apk_path, True, True)

            if "Error" in result:
                console.print(f"[bold red]Installation failed:[/] {result}")
                return
            else:
                console.print(f"[bold green]Installation successful![/]")

            console.print(f"[bold blue]Step 2/2: Enabling accessibility service[/]")

            try:
                await enable_portal_accessibility(device_obj)

                console.print("[green]Accessibility service enabled successfully![/]")
                console.print(
                    "\n[bold green]Setup complete![/] The DroidRun Portal is now installed and ready to use."
                )

            except Exception as e:
                console.print(
                    f"[yellow]Could not automatically enable accessibility service: {e}[/]"
                )
                console.print(
                    "[yellow]Opening accessibility settings for manual configuration...[/]"
                )

                await device_obj.shell(
                    "am start -a android.settings.ACCESSIBILITY_SETTINGS"
                )

                console.print(
                    "\n[yellow]Please complete the following steps on your device:[/]"
                )
                console.print(
                    f"1. Find [bold]{PORTAL_PACKAGE_NAME}[/] in the accessibility services list"
                )
                console.print("2. Tap on the service name")
                console.print(
                    "3. Toggle the switch to [bold]ON[/] to enable the service"
                )
                console.print("4. Accept any permission dialogs that appear")

                console.print(
                    "\n[bold green]APK installation complete![/] Please manually enable the accessibility service using the steps above."
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")

        if debug:
            import traceback

            traceback.print_exc()


@cli.command()
@click.option("--device", "-d", help="Device serial number or IP address", default=None)
@click.option(
    "--debug", is_flag=True, help="Enable verbose debug logging", default=False
)
@coro
async def ping(device: str | None, debug: bool):
    """Ping a device to check if it is ready and accessible."""
    try:
        device_obj = await device_manager.get_device(device)
        if not device_obj:
            console.print(f"[bold red]Error:[/] Could not find device {device}")
            return

        await ping_portal(device_obj, debug)
        console.print(
            "[bold green]Portal is installed and accessible. You're good to go![/]"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    command = "Open the settings app"
    device = None
    provider = "GoogleGenAI"
    model = "models/gemini-2.5-flash"
    temperature = 0
    api_key = os.getenv("GOOGLE_API_KEY")
    steps = 15
    vision = True
    reasoning = True
    reflection = False
    tracing = True
    debug = True
    base_url = None
    api_base = None
    ios = False
    run_command(
        command=command,
        device=device,
        provider=provider,
        model=model,
        steps=steps,
        temperature=temperature,
        vision=vision,
        reasoning=reasoning,
        reflection=reflection,
        tracing=tracing,
        debug=debug,
        base_url=base_url,
        api_base=api_base,
        api_key=api_key,
        ios=ios,
    )
