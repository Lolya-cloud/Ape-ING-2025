# protegi meta prompts
GRADIENT_DESCENT_PROMPT = """\
I'm trying to write a zero-shot prompt for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
But this prompt gets the following examples wrong:
{error_str}
give {num_gradients} reasons why the prompt could
have gotten these examples wrong. Only give solid reasons, do not consider formating of the result. DO NOT GIVE MORE REASONS THAN {num_gradients}.
Wrap each reason with <START> and <END>. WRAP EACH REASON WITH <START> and <END>, DO NOT MODIFY THE TAG.
"""

INCORPORATING_GRADIENT_FEEDBACK_PROMPT = """\
I'm trying to write a zero-shot prompt for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
But it gets the following examples wrong:
{error_str}
Based on these examples the problem with this
prompt is that {gradient}
Based on the above information, Write
{steps_per_gradient} different improved prompts.
Wrap each prompt with <START> and <END>.
WRAP EACH PROMPT WITH <START> and <END>, DO NOT MODIFY THE TAG.
The {steps_per_gradient} new prompts are:
"""

PARAPHRASE_PROMPT = """\
I'm trying to write a zero-shot prompt for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
Based on the above information, Write
{num_paraphrases} different prompts paraphrasing the current one.
Wrap each prompt with <START> and <END>. DO NOT MODIFY THE TAG.
The {num_paraphrases} new prompts are:
"""
