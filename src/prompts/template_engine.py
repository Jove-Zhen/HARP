"""
TemplateEngine: Fills {{placeholders}} and processes {{#if}} conditionals.
"""
import re
from typing import Dict


class TemplateEngine:
    """
    Minimal template engine for the markdown prompt files.
    Supports:
      - {{variable}} substitution
      - {{#if variable}}...{{/if}} conditional blocks
    """

    @staticmethod
    def render(template: str, variables: Dict[str, str]) -> str:
        """
        Render template with variable substitution.

        Args:
            template: raw markdown text with {{placeholders}}
            variables: mapping of name -> value

        Returns:
            rendered text

        Processing order:
        1. Resolve {{#if var}}...{{/if}} blocks (include if var is truthy, remove otherwise)
        2. Replace all {{var}} with corresponding values
        3. Replace any remaining {{var}} with empty string (fail-open)
        """
        result = template

        def replace_conditional(match):
            var_name = match.group(1)
            block_content = match.group(2)
            return block_content if variables.get(var_name) else ''

        result = re.sub(
            r'\{\{#if (\w+)\}\}(.*?)\{\{/if\}\}',
            replace_conditional,
            result,
            flags=re.DOTALL
        )

        for key, value in variables.items():
            result = result.replace(f'{{{{{key}}}}}', str(value))

        result = re.sub(r'\{\{\w+\}\}', '', result)

        return result
