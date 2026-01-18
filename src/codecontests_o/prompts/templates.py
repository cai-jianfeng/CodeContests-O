"""
Prompt Templates

Contains all prompt templates for interacting with LLM
"""


SYSTEM_PROMPT = """You are a helpful assistant. You must strictly follow the user's instructions."""


TESTLIB_HEADER_COMMENT = """
/*
 * It is strictly recommended to include "testlib.h" before any other include
 * in your code. In this case testlib overrides compiler specific "random()".
 *
 * If you can't compile your code and the compiler outputs something about
 * ambiguous calls of "random_shuffle", "rand" or "srand", it means that
 * you shouldn't use them. Use "shuffle", and "rnd.next()" instead because
 * these calls produce stable results for any C++ compiler. Read
 * sample generator sources for clarification.
 *
 * Please read the documentation for class "random_t" and use the "rnd" instance in
 * generators. These sample calls might be useful for you:
 *              rnd.next(); rnd.next(100); rnd.next(1, 2);
 *              rnd.next(3.14); rnd.next("[a-z]{{1,100}}").
 *
 * Also read about wnext() to generate off-center random distributions.
 *
 * See https://github.com/MikeMirzayanov/testlib/ to get the latest version or bug tracker.
 */
"""
"""
从问题描述直接生成 Generator 的 Prompt Template

输入: testlib_content (完整的 testlib.h), problem_statement
输出: input_constraints_summary, generator, command_list
"""

INIT_PROMPT_TEMPLATE_WITH_GENERATOR = """You are an expert in generating command-line arguments for corner case generation programs for programming problems.

Given the following problem statement and a C++ generation program, your tasks are:
1. Carefully read and understand the problem statement.
2. Carefully read and understand the provided generation program, which is designed to generate corner case inputs for this problem.
3. Identify and summarize the constraints of the input data.
4. Analyze the problem and the generation program to anticipate common mistakes or edge cases that contestants might overlook.
5. If the provided generator is incomplete or insufficient to produce high-quality adversarial cases (e.g., missing modes/flags/branches or has buggy logic), propose minimal, concrete generator code improvements using search-replace blocks. Each block must strictly follow the pattern:
    <<<<<<< SEARCH
    <original code fragment to search for>
    =======
    <replacement fragment (the improved code)>
    >>>>>>> REPLACE

    Notes:
    - Provide only the smallest necessary surrounding context to uniquely match; avoid large blocks.
    - Prefer multiple small, focused replacements over a single massive one.
    - Do not add explanations around the blocks; return only the blocks themselves as strings.
    - Pay close attention to code indentation, spaces, and line breaks; do not omit or alter them in the search/replace fragments.
    - For each SEARCH block, you must strictly copy the exact content from the provided generator. Do NOT add or modify any characters, such as adding "-" or "+" at the beginning of lines. The SEARCH block must be an exact substring of the generator.
    - For each REPLACE block, strictly follow the code format and ensure that after replacing the SEARCH content with the REPLACE content, the generator can be compiled and run directly.
    - In the REPLACE blocks you add, if you need to introduce new functions or variables, ensure that these functions or variables are already defined or imported in the generator. Do not introduce non-existent functions or variables, and carefully check whether the parameters of the called functions are correct. For example, the common `rnd.shuffle()` function may cause an error: 'class random_t' has no member named 'shuffle'; `rnd.next` requires two arguments of the same type; the `ensure` function only accepts one argument, etc. Below is the header comment from the testlib.h used by the code:
{testlib_header}
    - In the REPLACE blocks you add, if you need to reference variables from other parts of the code, carefully check their scope to ensure that the referenced variables are visible in the generator.
6. Based on your analysis and the improved generator, design and output a diverse set of command-line commands ("command_list") that, when executed, will use the generation program to generate corner case inputs that cover as many special and adversarial cases as possible. Note that the format and arguments of the command line must comply with the requirements of the generation program. For example, ensure that --seed may be an invalid argument, and when --n usually expects a numeric value, do not pass a string.

Problem Statement:
{problem_statement}

Generation Program (C++):
{generator}

**Strictly follow these output requirements:**
- Your response must be in JSON format matching this structure:
    {{
        "input_constraints_summary": "string describing input constraints from the problem statement",
        "search_replace_generator_blocks": [
            "<<<<<<< SEARCH\\n<original>\\n=======\\n<replacement>\\n>>>>>>> REPLACE",
            ...
        ],
        "command_list": ["./gen --arg1 value1 ...", "./gen --arg2 value2 ...", ...]
    }}
- The "input_constraints_summary" field should contain a clear and concise summary of all input constraints, including both explicit constraints mentioned in the problem statement (such as input size limits, value ranges, format requirements, etc.) and any implicit constraints that can be inferred from the problem description (such as properties, invariants, or hidden requirements implied by the problem context).
- `search_replace_generator_blocks` is optional—include it only when the generator needs improvements. Each item must strictly follow the search–replace block format shown above. If no changes are needed, return an empty list ([]). If changes are proposed, ensure that `command_list` is generated against the updated generator (i.e., after applying the edits).
- The "command_list" field must contain a list of shell commands, each starting with './gen' and followed by the appropriate arguments for the generation program. Each command should be designed to generate one corner case input. All corner case inputs generated by these commands should be as diverse and adversarial as possible, covering a wide range of edge cases and adversarial scenarios.
- Do not generate the corner case inputs directly; only generate the command lines to run the generation program.
- The commands should be ready to execute in a Linux shell and should use proper argument formatting as required by the generation program.
"""

INIT_PROMPT_TEMPLATE_WITHOUT_GENERATOR = """You are an expert in competitive programming and test case generation. Your task is to create a high-quality C++ test case generator for the given programming problem.

Given the following problem statement and the complete testlib.h library reference, your tasks are:
1. Carefully read and understand the problem statement.
2. Identify and summarize all input constraints (explicit and implicit).
3. Analyze the problem to anticipate common mistakes, edge cases, and adversarial scenarios that contestants might encounter.
4. Write a complete, well-structured C++ generator program using testlib.h that can produce diverse and adversarial test cases.
5. Design a set of command-line commands that will use your generator to cover various corner cases.

## Problem Statement:
{problem_statement}

## testlib.h Library Reference:
The following is the complete testlib.h library that your generator will use. Study its API carefully to write correct generator code.

```cpp
{testlib_content}
```

## Generator Requirements:

### Code Structure:
Your generator must:
1. Include `#include "testlib.h"` as the first include
2. Use `registerGen(argc, argv, 1);` at the beginning of main()
3. Use the `rnd` object from testlib for all random generation
4. Support command-line arguments to control test case generation (e.g., size, type, edge cases)
5. Output the generated test case to stdout

### Key testlib.h APIs to use:
- `rnd.next(int n)` - returns random int in [0, n-1]
- `rnd.next(int l, int r)` - returns random int in [l, r]
- `rnd.next(long long l, long long r)` - returns random long long in [l, r]
- `rnd.next(double l, double r)` - returns random double in [l, r]
- `rnd.next("[a-z]{{1,10}}")` - returns random string matching regex
- `rnd.wnext(int n, int type)` - returns weighted random (type>0: biased toward n-1, type<0: biased toward 0)
- `rnd.any(container)` - returns random element from container
- `shuffle(container.begin(), container.end())` - shuffles container (use this, NOT rnd.shuffle)
- `opt<T>(name)` - gets command-line option value (e.g., `opt<int>("n")` for `-n=100` or `--n=100`)
- `has_opt(name)` - checks if option exists

### Important Notes:
- Do NOT use `rnd.shuffle()` - it doesn't exist. Use `shuffle(vec.begin(), vec.end())` instead.
- `rnd.next(a, b)` requires a and b to be the same type
- For strings, use `rnd.next("[a-z]{{n}}")` format where n is the length
- Use `println()` or `cout` for output
- Design flexible command-line arguments to control:
  - Test size (n, m, etc.)
  - Value ranges
  - Special case modes (e.g., all same values, strictly increasing, etc.)
  - Edge case triggers

### Generator Design Principles:
1. **Modular**: Support different generation modes via command-line flags
2. **Flexible**: Allow size and value ranges to be specified
3. **Adversarial**: Include modes for edge cases that might break naive solutions
4. **Diverse**: Cover minimum, maximum, and random cases

## Output Format:

**Strictly follow these output requirements:**
- Your response must be in JSON format matching this structure:
    {{
        "input_constraints_summary": "string describing all input constraints",
        "generator": "complete C++ generator code as a single string",
        "command_list": ["./gen arg1 arg2 ...", "./gen arg1 arg2 ...", ...]
    }}

### Field Specifications:

1. **input_constraints_summary**: A clear and concise summary of all input constraints, including:
   - Explicit constraints from the problem (size limits, value ranges, format requirements)
   - Implicit constraints inferred from the problem context (properties, invariants, special conditions)

2. **generator**: A complete, compilable C++ program that:
   - Includes proper headers (testlib.h first)
   - Uses `registerGen(argc, argv, 1);`
   - Parses command-line arguments using `opt<T>()` and `has_opt()`
   - Generates valid test cases according to the problem's input format
   - Handles various modes/options for different types of test cases
   - The code should be a single string with proper newlines (\\n)

3. **command_list**: A list of shell commands where each command:
   - Starts with `./gen`
   - Uses appropriate arguments for your generator
   - Is designed to produce a specific type of corner case
   - Together, the commands should cover:
     * Minimum size cases
     * Maximum size cases
     * Edge cases specific to the problem
     * Random cases of various sizes
     * Adversarial cases that might break common incorrect solutions

## Example Generator Structure:

```cpp
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {{
    registerGen(argc, argv, 1);
    
    // Parse command-line arguments
    int n = opt<int>("n", 10);  // default value 10
    int maxVal = opt<int>("maxval", 1000000);
    string mode = opt("mode", "random");
    
    // Generate based on mode
    if (mode == "random") {{
        // random generation
    }} else if (mode == "max") {{
        // maximum values
    }} else if (mode == "edge") {{
        // edge cases
    }}
    
    // Output
    println(n);
    // ... rest of output
    
    return 0;
}}
```

Now, analyze the problem and create the generator:
"""


REFINE_PROMPT_TEMPLATE = """Now you need to refine the previously generated command list for the corner case generation program based on evaluation feedback.

You previously generated a set of commands for the given programming problem. The process is as follows:
1. The generated `search_replace_generator_blocks` have already been applied to the generator. Any blocks whose SEARCH fragments did not match exactly were skipped.
2. Each command is executed to generate one or more corner case inputs.
3. For each generated corner case input, the canonical solution is executed to obtain the corresponding output, thus forming a complete corner case (input + output).
4. These corner cases are then used to evaluate both correct solutions and incorrect solutions.

Current improved Generation Program (C++) (Note: The edits from the previously returned `search_replace_generator_blocks` have already been applied to the generator below. Any blocks that are not reflected were skipped because their SEARCH fragments did not match exactly. If the previous `search_replace_generator_blocks` was empty or none of the blocks were applied, then the generator shown here is the same as the originally provided generator and will appear as an empty string):
{improved_generator}

Current command list: {current_command_list}

For each command, here are the corresponding generated corner case input(s) (for some commands that generate very long inputs, the input for that command is replaced by `[input]`):
{command_to_input_map}

If any command failed to execute or produced errors when generating input, here are the error messages (if any):
{command_run_errors} (ideally, this should be empty)

The evaluation results are as follows (ideally, all three should be empty):
- Outputs from correct solutions: These are cases where the generated corner cases incorrectly cause correct solutions to fail (i.e., the correct solution is judged as wrong on these cases). {correct_results}
- Outputs from incorrect solutions: These are cases where the generated corner cases fail to expose bugs in incorrect solutions (i.e., the incorrect solution is judged as correct on these cases). {incorrect_results}
- Outputs from the canonical solution (only includes results for cases that failed when run with the canonical solution): These are cases where the canonical solution itself fails or produces errors on the generated corner cases. {outputs}

Please note:
- For some commands that generate very long inputs, the `stdin` field in `correct_results`/`incorrect_results`/`outputs` may be replaced by the corresponding command string, and the field will have a trailing ` [command]` tag to indicate this substitution. When you see such a `stdin` value, you should use the provided mapping between commands and generated inputs to implicitly convert the command back to its actual `stdin` content for any reasoning, comparison, or decision-making tasks.
- For some cases where the output (`stdout`/`expected_output`) is very long, the `stdout`/`expected_output` field may be replaced by `[output]`/`[expected output]`. When you see `[output]`/`[expected output]`, in this case, if the solution's `passed` field is False, you should rely only on the given solution content for reasoning.

Here is a clear and concise summary of the input constraints mentioned in the problem statement (e.g., input size limits, value ranges, format requirements, etc.): {input_constraints_summary}

Your tasks are:
1. Based on the above canonical solution results, identify any commands that generate invalid or unhelpful corner cases (i.e., those that fail when run with the canonical solution) and mark them for replacement.
2. Based on the correct solutions results, identify commands that generate corner cases which incorrectly classify correct solutions as wrong, and mark them for replacement.
3. Analyze the above results to determine which commands fail to effectively distinguish between correct and incorrect solutions.
4. If the provided generator is incomplete/insufficient to produce high-quality adversarial cases (e.g., missing modes/flags/branches or has buggy logic), propose minimal, concrete generator code improvements using search-replace blocks. 
5. Generate new additional commands that can better expose bugs in incorrect solutions and improve differentiation between correct and incorrect solutions.

**Strictly follow these output requirements:**
- Your response must be in JSON format matching this structure:
    {{
        "search_replace_generator_blocks": [
            "<<<<<<< SEARCH\\n<original>\\n=======\\n<replacement>\\n>>>>>>> REPLACE",
            ...
        ],
        "replace_command_list": ["old_command_1", "old_command_2", ...],
        "add_command_list": ["new_command_1", "new_command_2", ...]
    }}
- `search_replace_generator_blocks` is optional—include it only when the generator needs improvements. Each item must strictly follow the search–replace block format shown above. If no changes are needed, return an empty list ([]). If changes are proposed, ensure that both `replace_command_list` and `add_command_list` are generated against the updated generator (i.e., after applying the edits).
- `replace_command_list` contains commands from the original list that should be removed/replaced due to generating invalid or unhelpful corner cases, or incorrectly classifying correct solutions as wrong.
- `add_command_list` contains new commands to be added to better distinguish correct and incorrect solutions, including improved versions of replaced commands and completely new adversarial commands.
- Each command should be a shell command starting with './gen' and followed by the appropriate arguments for the generation program.
- Do not generate the corner case inputs directly; only generate the command lines to run the generation program.
- The commands should be ready to execute in a Linux shell and should use proper argument formatting as required by the generation program.

Please focus on maximizing the adversarial value of the generated corner cases based on the feedback above.
"""


# Response templates
INIT_RESPONSE_TEMPLATE_WGEN = """{{
    "input_constraints_summary": {input_constraints_summary},
    "search_replace_generator_blocks": {search_replace_generator_blocks},
    "command_list": {command_list}
}}"""


INIT_RESPONSE_TEMPLATE_WOGEN = """{{
    "input_constraints_summary": {input_constraints_summary},
    "generator": {generator},
    "command_list": {command_list}
}}"""

REFINE_RESPONSE_TEMPLATE = """{{
    "search_replace_generator_blocks": {search_replace_generator_blocks},
    "replace_command_list": {replace_command_list},
    "add_command_list": {add_command_list}
}}"""


# Result formatting templates
SOLUTION_RESULT_TEMPLATE = """
language: {language},
solution: {solution},
output: {output}
"""

TEST_CASE_RESULT_TEMPLATE = "passed: {passed}; stdin: {stdin}; stdout: {stdout}; expected_output: {expected_output}; error_info: {error_info}"

CANONICAL_SOLUTION_TEMPLATE = """
canonical_solution_language: {language}, 
canonical_solution: {solution},
stdin: {stdin},
output: {output}
"""

SOLUTION_CODE_TEMPLATE = """
Here is a {language} solution to the problem: 
```{language}
{solution}
```
"""
