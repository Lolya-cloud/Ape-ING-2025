GRADIENT_DESCENT_PROMPT_SCHEMA = """\
I'm trying to write a zero-shot model for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
But with this prompt, the model gets following examples wrong:
{error_str}.
Analyze reasons for the mistake and list {num_gradients} fixes based on the prompt and the documents attached. 
Formulate each fix as a short instruction fixing the error, THAT WILL LATER BE APPENDED TO THE MAIN PROMPT. 
SO DIRECT THE INSTRUCTION TO THE MODEL THAT WILL PERFORM INFORMATION EXTRACTION (AS ADDITONAL INSTRUCTIONS).
Address the instruction to the model performing the task, not to the person asking this.
Do not overfit to the document, the fix should be generalizable to other documents. 
Do not repeat the main prompt in your fixes, the fixes should be additive, not standalone.
For context, some of the fixes will later be appended to the main prompt, so formulate them as short instructions additive to the prompt.
Only give solid fixes, do not consider specific formating of the result. DO NOT GIVE MORE FIXES THAN {num_gradients}.
USE THE DOCUMENT I ATTACH TO THIS PROMPT FOR THE FEEDBACK, DO NOT GUESS. Use the schema attached to format your response (one variable in the schema = 1 fix)
The {num_gradients} fixes are:
"""

GROUP_PROMPTS_LLM = """
I will give you a list of brief instructions for an information extraction llm-backed system.
The format of the instructions will be:
Instruction 1: "text"
Instruction 2: "text"
and so on.
The list will have some similar instructions, some different and some contradicting instructions.
Your goal is to identify similar/identical instructions and
combine similar/identical without touching different and contradicting. If the instructions are related
and talk about same concept differently, combine them. Only return the combined instructions, unique, different
and contradicting (do not return both new instructions combied from previous and also previous.)
Also combine prompts which talk about closely related stuff but in different words (and do not contradict).
Identify similar/paraphrased/identical instructions in the given list and combine them.
If the instructions are unique or have significant diferences, or are contradicting, 
return them as is. If the instruction is "dirty", e.g. has some leftover tags in it, clean it up.
Wrap each output insturction with <START> and <END>. WRAP EACH ONE WITH <START> and <END>, DO NOT MODIFY THE TAG. 
ONLY USE THE TAG FOR WRAPPING THE INSTRUCTIONS, Do not use it anywhere else.
"""

IDENTIFY_CONTRADICTIONS_PROMPT = """\
You are a heuristic filter for short task instructions. I will give you two instructions and you
will identify whether they are directly contradicting (e.g. you cannot follow both at the same time).
In other words, if following the first instruction makes it impossible to follow the second, or vice-versa,
return True. If the instructions do not appear to contradict and can be executed together (by aggregating them),
return False. ONLY RETURN TRUE IF THERE IS A REAL AND SERIOUS CONTRADICTION.
The two instructions: {instruction_pair}
"""

ASSESS_SIMILARITY_PROMPT = """\
You are a binary similarity classifier between two short instructions.
I will give you a pair of short instructions for an information extraction llm-backed system.
Identify whether instructions are similar/identical/related, or not. Instructions are considered similar
if they talk about the same idea in different words and they do not contradict. 
The instructions are also similar if they are paraphrases of each other. Otherwise, they are not similar.
Instruction 1: {instruction_1};
Instruction 2: {instruction_2};

Return True if similar, False if not.

For context, the instructions are used as a part of a larger information extraction workflow,
where I extract information from long documents using llms and a prompt. The prompt contains the starting
part (basic task description, unchanged), while the short instructions are additves aimed at supplementing/improving/strengthening the starting prompt.
Here is the startig prompt: {starting_prompt}. While analyzing similarity between the two instructions,
keep this starting prompt in mind (to more effectively distinguish between similar and dissimilar based
on the context knowledge, which is formulated as a starting prompt).
"""

COMBINE_SIMILAR_INSTRUCTIONS_PROMT = """\
Here is a list of short instructions. Combine them into one (do not repeat similar stuff).
"""
