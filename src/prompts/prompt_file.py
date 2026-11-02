"""
PromptFile: Parses a single markdown prompt file into structured sections.
"""
import re
import hashlib
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PromptSection:
    """A single ## section from a markdown prompt file."""
    name: str
    raw_content: str
    placeholders: List[str]
    conditionals: List[str]


@dataclass
class PromptFile:
    """
    Parsed representation of a markdown prompt file.

    Attributes:
        path: absolute path to the .md file
        name: stem name (e.g., "user", "discovery", "soul")
        sections: dict mapping section name -> PromptSection
        raw_text: original full text
        version_hash: SHA-256 of raw_text (for change detection)
    """
    path: Path
    name: str
    sections: Dict[str, PromptSection] = field(default_factory=dict)
    raw_text: str = ""
    version_hash: str = ""

    @classmethod
    def from_file(cls, filepath: str) -> 'PromptFile':
        """
        Parse a markdown file into a PromptFile.

        Args:
            filepath: absolute path to .md file

        Returns:
            PromptFile instance with sections parsed

        Raises:
            FileNotFoundError: if file doesn't exist
            ValueError: if required sections are missing
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        text = path.read_text(encoding='utf-8')
        sections = cls._parse_sections(text)
        version_hash = hashlib.sha256(text.encode()).hexdigest()

        return cls(
            path=path,
            name=path.stem,
            sections=sections,
            raw_text=text,
            version_hash=version_hash
        )

    @staticmethod
    def _parse_sections(text: str) -> Dict[str, PromptSection]:
        """
        Split markdown text on ## headings.
        Uses re.split(r'(?=^## )', text, flags=re.MULTILINE)

        Returns:
            dict mapping section name -> PromptSection
        """
        # Split on ## headings using lookahead (keeps the heading in each chunk)
        parts = re.split(r'(?=^## )', text, flags=re.MULTILINE)
        sections = {}

        for part in parts:
            part = part.strip()
            if not part or not part.startswith('## '):
                continue

            lines = part.split('\n', 1)
            name = lines[0].replace('## ', '').strip()
            content = lines[1] if len(lines) > 1 else ''

            sections[name] = PromptSection(
                name=name,
                raw_content=content,
                placeholders=PromptFile._extract_placeholders(content),
                conditionals=PromptFile._extract_conditionals(content)
            )

        return sections

    @staticmethod
    def _extract_placeholders(content: str) -> List[str]:
        """
        Extract all {{placeholder}} names from content.
        Pattern: re.findall(r'\{\{(\w+)\}\}', content)

        Returns:
            list of placeholder names (e.g., ['user_id', 'n_facets'])
        """
        return re.findall(r'\{\{(\w+)\}\}', content)

    @staticmethod
    def _extract_conditionals(content: str) -> List[str]:
        """
        Extract all {{#if name}}...{{/if}} conditional block names.
        Pattern: re.findall(r'\{\{#if (\w+)\}\}', content)
        """
        return re.findall(r'\{\{#if (\w+)\}\}', content)
