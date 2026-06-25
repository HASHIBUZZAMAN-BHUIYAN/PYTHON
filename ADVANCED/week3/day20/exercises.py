# Advanced Day 20 — Exercises

# Exercise 1 — Add a new tool: unit converter
# Implement tool_convert_units(value_and_units) that handles:
#   "100 km to miles" → 62.14 miles
#   "32 fahrenheit to celsius" → 0.0 celsius
#   "1 kg to pounds" → 2.205 pounds
# Register it in TOOLS and add it to TOOL_DESCRIPTIONS.
# Test with agent.run("Convert 50 km to miles").
# TODO

# Exercise 2 — Agent memory / context
# The basic agent above does not remember previous conversations.
# Implement a PersistentAgent that stores all past Q&A pairs in a list
# and includes the last 3 exchanges in each new prompt as "context".
# Show that it can answer "What did I ask before?" correctly.
# TODO

# Exercise 3 — Multi-step planning
# Create a query that requires 3+ tool calls:
#   "What is the current temperature in London, and what is (temperature^2 + 10) / 3?"
# The agent must: get_weather → extract temp → calculator
# Add a debug flag that prints each message in the messages list.
# TODO

# Exercise 4 — Tool error handling
# Modify the agent to gracefully handle a tool that raises an exception.
# Add a tool_divide(a_and_b) that evaluates "a / b".
# If b == 0, raise ZeroDivisionError.
# The agent should catch the error, include it in the OBSERVE message,
# and attempt an alternative approach (e.g., report the error).
# TODO

# Exercise 5 — Streaming output mock
# Real LLM APIs support streaming (tokens arrive one by one).
# Simulate streaming: split the mock LLM response into characters,
# print each character with a 0.01s delay (like typewriter output).
# Wrap this in a StreamingReActAgent class.
# TODO
