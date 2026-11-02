"""
PromptLoader: Discovers and loads all .md prompt files from a directory.
"""
from pathlib import Path
from typing import Dict
from src.prompts.prompt_file import PromptFile


class PromptLoader:
    """
    Discovers and loads markdown prompt files from a directory.

    Usage:
        loader = PromptLoader("prompts/")
        templates = loader.load_all()  # -> {'memory_update': PromptFile, ...}
    """

    def __init__(self, prompts_dir: str):
        """
        Args:
            prompts_dir: path to the prompts directory
        """
        self.prompts_dir = Path(prompts_dir)

    def load_all(self) -> Dict[str, PromptFile]:
        """
        Load all .md files in the prompts directory.

        Returns:
            dict mapping template name -> PromptFile

        Raises:
            FileNotFoundError: if prompts_dir doesn't exist
        """
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        templates = {}
        for md_file in self.prompts_dir.glob("*.md"):
            if "_meta" in str(md_file):
                continue

            template = PromptFile.from_file(str(md_file))
            templates[template.name] = template

        return templates
