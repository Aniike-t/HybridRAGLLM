# retrieval/prompt_builder.py

from typing import List, Tuple

class PromptBuilder:
    def __init__(self, prompt_template: str = None):
        """
        Initializes the PromptBuilder.

        Args:
            prompt_template:  A string template for the prompt.
                Placeholders:
                    {context}:  Replaced with the retrieved context.
                    {question}: Replaced with the user's question.
                Example:
                   "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        """
        if prompt_template is None:
            self.prompt_template = (
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        else:
            self.prompt_template = prompt_template

    def build_prompt(self, context: str, question: str) -> str:
        """Builds the prompt for the LLM."""
        return self.prompt_template.format(context=context, question=question)