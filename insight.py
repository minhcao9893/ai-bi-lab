# insight.py
"""M4: Insight Object builder + LLM narrative call."""
import json
import httpx

def build_insight(metric: str, current: float, prior: float, drivers: list[dict], severity_threshold: float = 0.1) -> dict:
    change_pct = ((current - prior) / prior) if prior else 0
    severity = 'high' if abs(change_pct) > 0.2 else 'medium' if abs(change_pct) > severity_threshold else 'low'

    insight = {
        'metric': metric,
        'current': current,
        'prior': prior,
        'change_pct': round(change_pct * 100, 2),
        'severity': severity,
        'drivers': drivers[:3],
    }
    print(f"[AI-DEBUG] build_insight: metric={metric} change_pct={insight['change_pct']} severity={severity}")
    return insight


async def narrate_insight(insight: dict, api_key: str, model: str = "claude-sonnet-4-6") -> str:
    """Send compact Insight Object to LLM, get natural language narrative back."""
    prompt = f"""You are a BI analyst. Given this insight data, write a 2-3 sentence business narrative explaining what happened and why. Be specific, use numbers.

Data:
{json.dumps(insight, ensure_ascii=False)}

Narrative:"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": model,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        data = resp.json()
        print(f"[AI-DEBUG] narrate_insight: status={resp.status_code} metric={insight['metric']}")
        try:
            return data['content'][0]['text']
        except (KeyError, IndexError) as e:
            print(f"[AI-DEBUG] narrate_insight: parse_error={e} raw={data}")
            return "Unable to generate narrative."
