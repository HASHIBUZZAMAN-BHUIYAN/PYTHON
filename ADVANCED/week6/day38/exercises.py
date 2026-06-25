# Advanced Day 38 — Exercises

# Exercise 1 — Tool timeout
# Add a timeout wrapper: if a tool takes longer than N seconds, raise ToolError.
# Implement using threading.Timer or signal (threading is cross-platform).
# Test with a tool that sleeps for 2 seconds; timeout set to 0.5 seconds.
# TODO

# Exercise 2 — Rate limiter
# Add a rate limiter decorator: max N calls per minute per tool.
# If the limit is exceeded, raise ToolError("Rate limit exceeded").
# Test with limit=3 and calling a tool 5 times in a loop.
# TODO

# Exercise 3 — Tool composition
# Create a ComposedTool that chains two tools: output of tool A is
# passed as the first argument to tool B.
# Example: clean_text | summarize_text
# Implement compose(tool_a, tool_b) and test it.
# TODO

# Exercise 4 — Parallel tool calls
# Implement run_parallel(tools_with_args) that runs multiple independent
# tools concurrently using concurrent.futures.ThreadPoolExecutor.
# Collect and return all results (or errors) in a dict.
# TODO

# Exercise 5 — Tool version compatibility
# Add a "version" field to Tool. When calling a tool, check if the
# requested version matches. If not, check if a fallback version exists.
# Implement register_version and call_versioned in ToolRegistry.
# TODO
