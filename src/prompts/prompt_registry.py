"""
PromptRegistry: loads the markdown prompt templates and serves them by name.
"""
from typing import Dict
from pathlib import Path
from src.prompts.prompt_file import PromptFile
from src.prompts.prompt_loader import PromptLoader


class PromptRegistry:
    """Central registry for the loaded prompt-template files (`get(name)`, `versions()`)."""

    def __init__(self, prompts_dir: str):
        self.prompts_dir = Path(prompts_dir)
        self.loader = PromptLoader(str(prompts_dir))
        self._templates: Dict[str, PromptFile] = self.loader.load_all()

    def get(self, name: str) -> PromptFile:
        """Get a loaded template by name. Raises KeyError if not found."""
        if name not in self._templates:
            raise KeyError(f"Prompt template not found: {name}")
        return self._templates[name]

    def versions(self) -> Dict[str, str]:
        """Return {name: version_hash} for all loaded templates."""
        return {name: template.version_hash for name, template in self._templates.items()}
