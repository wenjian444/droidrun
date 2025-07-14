"""
DroidRun - A framework for controlling Android devices through LLM agents.
"""

__version__ = "0.3.0"

# Import main classes for easier access
from droidrun.agent.utils.llm_picker import load_llm
from droidrun.adb.manager import DeviceManager
from droidrun.tools import Tools, AdbTools, IOSTools
from droidrun.agent.droid import DroidAgent


# Make main components available at package level
__all__ = [
    "DroidAgent",
    "DeviceManager",
    "load_llm",
    "Tools",
    "AdbTools",
    "IOSTools",
]
