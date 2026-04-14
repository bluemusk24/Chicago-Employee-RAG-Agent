def valid_reasoning(inputs: dict, outputs: dict) -> bool:
    """Use an LLM to judge if the reasoning and the answer are consistent."""
    # Define the evaluation criteria
    instructions = """
Given the following question, answer, and reasoning, determine if the reasoning
for the answer is logically valid and consistent with the question and the answer."""

    # Use structured output to get a boolean score
    class Response(BaseModel):
        reasoning_is_valid: bool

    # Construct the prompt with the actual inputs and outputs
    msg = f"Question: {inputs['question']}\nAnswer: {outputs['answer']}\nReasoning: {outputs['reasoning']}"

    # Call the LLM to judge the output
    response = ollama_client.beta.chat.completions.parse(
        model="qwen3-coder-next:cloud",
        messages=[{"role": "system", "content": instructions}, {"role": "user", "content": msg}],
        response_format=Response
    )

    # Return the boolean score
    return response.choices[0].message.parsed.reasoning_is_valid