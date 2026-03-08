"""System prompt and tool schema for Duck Hunt VLM training."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tool schema (OpenAI / Mistral function-calling format)
# ---------------------------------------------------------------------------
SHOOT_TOOL = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": (
            "Fire at predicted duck position. "
            "Analyze the frame sequence to estimate duck velocity, "
            "then predict where the duck will be after "
            "processing_latency_frames + horizon frames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": (
                        "Predicted horizontal position, normalised 0.0–1.0. "
                        "0.0 = left edge, 1.0 = right edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": (
                        "Predicted vertical position, normalised 0.0–1.0. "
                        "0.0 = top edge, 1.0 = bottom edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "horizon": {
                    "type": "integer",
                    "description": (
                        "Additional frames to wait before shooting (0-30). "
                        "Total prediction = processing_latency_frames + horizon."
                    ),
                    "minimum": 0,
                    "maximum": 30,
                },
            },
            "required": ["x", "y", "horizon"],
        },
    },
}

TOOLS = [SHOOT_TOOL]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
You are a Duck Hunt AI. Shoot flying ducks by calling the shoot tool.

You see {num_frames} frames. You are NOT told the current network delay. It changes unpredictably. Use your recent shot results to infer if delay is high (you're missing) or low (you're hitting). Adjust horizon accordingly — increase if missing, decrease if hitting.
Your aim may also have random drift that changes over time. Adjust based on where your shots actually land relative to where you aimed.
Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).

IMPORTANT: Respond ONLY with shoot(x=<float>, y=<float>, horizon=<int>). No explanation."""


def format_system_prompt(
    *,
    num_frames: int = 4,
) -> str:
    """Return the system prompt with placeholders filled in."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        num_frames=num_frames,
    )
