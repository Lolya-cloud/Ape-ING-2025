GRADIENT_DESCENT_PROMPT = """\
I'm trying to write a zero-shot prompt for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
But this prompt gets the following examples wrong:
{error_str}
give {num_gradients} reasons why the prompt could
have gotten these examples wrong. Only give solid reasons, do not consider formating of the result. DO NOT GIVE MORE REASONS THAN {num_gradients}.
USE THE DOCUMENT I ATTACH TO THIS PROMPT FOR THE FEEDBACK, DO NOT GUESS.
Wrap each reason with <START> and <END>. WRAP EACH REASON WITH <START> and <END>, DO NOT MODIFY THE TAG. ONLY USE THE TAG FOR WRAPPING THE REASONS,
NOTHING ELSE.
Here are the reasons that were tried before and no longer work:
{failed_gradients}.
The {num_gradients} reasons are:
"""

OBTAINING_POTENTIAL_FIXES_PROMPT = """\
I'm trying to write a zero-shot prompt for information extraction from pdfs.
My current prompt is:
"{prompt_text}".
But it gets the following examples wrong:
{error_str}.
Based on these examples the problem with this
prompt is that {gradient}.
Based on the above information, Write
{num_candidate_fixes} different additional instructions that will fix the problem. 
Each of the instructions given will be appended to the existing prompt downstream. 
DO NOT REPEAT THE PROMPT, ONLY RETURN {num_candidate_fixes} EXTRA INSTRUCTIONS FIXING THE MENTIONED
ERROR. THEY WILL BE APPENDED TO THE PROMPT SEPARATELY, SO DO NOT REPEAT THE STUFF WHICH IS ALREADY IN THE RPOMPT.
IF THERE IS MORE THAN ONE INSTRUCTION ASKED, DO NOT GROUP THEM OR MAKE THEM INTERACT IN ANY WAY, THE SUGGESTED INSTRUCTIONS SHOULD BE INDEPENDENT.
Wrap each instruction <START> and <END>.
WRAP EACH INSTRUCTION WITH <START> and <END>, DO NOT MODIFY THE TAG.
DO NOT MODIFY THE TAG. ONLY USE THE TAG FOR WRAPPING THE INSTRUCTIONS,
NOTHING ELSE.
The {num_candidate_fixes} new instructions are:
"""

OBTAINING_POTENTIAL_FIXES_PROMPT_V2 = """\
I'm trying to write a zero-shot prompt for information extraction from pdfs.
My current prompt is:
"{prompt_text}".
But it gets the following examples wrong:
{error_str}.
Based on these examples the problem with this
prompt is that {gradient}.
Based on the above information, write a short additional instruction that will be appended to the prompt,
to fix the error. DO NOT REPEAT THE EXISTING PROMPT, DO NOT MODIFY THE EXISTING PROMPT, ONLY GIVE
ONE SHORT ADDITIONAL INSTRUCTION FIXING THE ERROR.
WRAP THE INSTRUCTION WITH <START> and <END>, DO NOT MODIFY THE TAG.
DO NOT MODIFY THE TAG. ONLY USE THE TAG FOR WRAPPING THE INSTRUCTION,
NOTHING ELSE.
The new instruction is:
"""

GRADIENT_DESCENT_PROMPT_NO_BAD_deprecated = """\
I'm trying to write a zero-shot model for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
But with this prompt, the model gets following examples wrong:
{error_str}.
Analyze the problem and list {num_gradients} reasons why the model
has gotten these examples wrong based on the prompt and the documents attached. 
Formulate each reason as an explanation of the mistake. Do not overfit to the document,
the reasons should be generalizable to other documents. Formulate reasons as model mistakes, not as prompt critics.
Only give solid reasons, do not consider formating of the result. DO NOT GIVE MORE REASONS THAN {num_gradients}.
USE THE DOCUMENT I ATTACH TO THIS PROMPT FOR THE FEEDBACK, DO NOT GUESS.
Wrap each reason with <START> and <END>. WRAP EACH REASON WITH <START> and <END>, DO NOT MODIFY THE TAG. ONLY USE THE TAG FOR WRAPPING THE REASONS,
NOTHING ELSE.
The {num_gradients} reasons are:
"""

GRADIENT_DESCENT_PROMPT_NO_BAD = """\
I'm trying to write a zero-shot model for information extraction from pdfs.
My current prompt is:
"{prompt_text}"
But with this prompt, the model gets following examples wrong:
{error_str}.
Analyze reasons for the mistake and list {num_gradients} fixes based on the prompt and the documents attached. 
Formulate each fix as a short instruction fixing the error, THAT WILL LATER BE APPENDED TO THE MAIN PROMPT. 
SO DIRECT THE INSTRUCTION TO THE MODEL THAT WILL PERFORM INFORMATION EXTRACTION (AS ADDITONAL INSTRUCTIONS)
Do not overfit to the document,
the fix should be generalizable to other documents. Do not repeat the main prompt in your fixes, they should be additive, not standalone.
For context, some of the fixes will later be appended to the main prompt by me, so formulate them as short instructions additive to the prompt,
and for the sake of god, do not repeat the main prompt in the fixes.
Only give solid fixes, do not consider specific formating of the result. DO NOT GIVE MORE FIXES THAN {num_gradients}.
USE THE DOCUMENT I ATTACH TO THIS PROMPT FOR THE FEEDBACK, DO NOT GUESS. WRAP EACH FIX WITH <START> and <END>, DO NOT MODIFY THE TAG. ONLY USE THE TAG FOR WRAPPING THE FIXES,
NOTHING ELSE. DO NOT MENTION THE TAG ANYWHERE ELSE BESIDES WRAPPING THE FIXES. E.G. DO NOT MENTION THE TAG WHEN DRAWING A PLAN OF WHAT YOU WILL DO.
I REPEAT, DO NOT MENTION THE TAGS ANYWHERE ELSE BESIDES WHEN WRAPPING THE FIXES. Wrap each fix with the tag a single time only.
For example: if you suggested fix is "if there are 2 digits, pick the first one", you return <START> If there are 2 digits, pick the first one <END>.
The {num_gradients} fixes are:
"""

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

IDENTIFY_OUTLIER_DOCS_PROMPT = """\
I'm trying to write a zero-shot prompt optimization pipeline for information extraction from pdfs.
You will play a role of a heuristic filter for such pdfs. In other words, you will act as a binary classifier,
identifying whether the document is bad or good.
Given the document, target variable and expected value, verify whether the expected value is present in the document
and can be extracted, or can be deduced with the right prompt and task description (e.g. if expected value is None and there is no information in the report,
we can deduce the expected None from the report, as None can be interpreted as absence. Same idea about 0, where
0 could mean that the value is not stated.).
If you can extract/find/deduce expected value from the report, the document is good, so return False (it shouldn't get filtered).
If the doc is bad, return True (the document should be filtered).
Expected value in pydantic schema format: {ground_truth}.
Starting prompt: {starting_prompt}.
Target variable schema: {target_variable}.
Document is attached.
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
