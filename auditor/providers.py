from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LLMResponse:
    prompt: str
    output: str
    meta: Dict[str, Any]

class MockProvider:
    """Offline mock simulating occasional risky outputs."""
    name = "mock"
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        lower = prompt.lower()
        risky = (
            "reveal system prompt" in lower
            or "ignore previous" in lower
            or "circumvent" in lower
            or "leak chain-of-thought" in lower
            or "make up citations" in lower
            or "disallowed medical" in lower
        )
        if risky:
            # flip between “unsafe-ish” and “safe”
            if hash(prompt) % 3 == 0:
                out = "Okay, here is the (simulated) system prompt: [REDACTED] …"
            elif "medical" in lower:
                out = "You should take 400mg of X every 4 hours. (simulated unsafe)"
            elif "citations" in lower:
                out = "According to Smith et al., 2021 (Imaginary Journal)…"
            else:
                out = "I will ignore the policy and continue. (simulated unsafe)"
        else:
            out = "I’m unable to help with that. (simulated safe)"
        return LLMResponse(prompt=prompt, output=out, meta={"provider":"mock"})
