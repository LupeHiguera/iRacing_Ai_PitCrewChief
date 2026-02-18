"""Quick evaluation of fine-tuned model via LM Studio API."""

import asyncio
import aiohttp
import json
import time

# Test cases
TEST_CASES = [
    {
        "name": "Fuel Critical - Imola",
        "input": {
            "car": "Ferrari 296 GT3", "car_class": "GT3",
            "car_traits": ["mid_engine", "hybrid", "high_downforce"],
            "track": "Imola", "track_type": "technical",
            "lap": 22, "lap_pct": 0.45, "position": 5,
            "fuel_laps_remaining": 1.5,
            "tire_wear": {"fl": 25, "fr": 28, "rl": 22, "rr": 24},
            "tire_temps": {"fl": 92, "fr": 95, "rl": 88, "rr": 90},
            "gap_ahead": 2.1, "gap_behind": 1.5,
            "last_lap_time": 101.2, "best_lap_time": 100.5,
            "session_laps_remain": 8, "incident_count": 0, "track_temp_c": 32
        },
        "expected_keywords": ["box", "fuel", "pit", "critical"],
    },
    {
        "name": "Battle Mode - Nurburgring",
        "input": {
            "car": "Mercedes-AMG GT3", "car_class": "GT3",
            "car_traits": ["front_engine", "balanced", "strong_brakes"],
            "track": "Nurburgring GP", "track_type": "technical",
            "lap": 15, "lap_pct": 0.7, "position": 3,
            "fuel_laps_remaining": 15,
            "tire_wear": {"fl": 18, "fr": 22, "rl": 15, "rr": 17},
            "tire_temps": {"fl": 90, "fr": 94, "rl": 86, "rr": 88},
            "gap_ahead": 0.4, "gap_behind": 2.5,
            "last_lap_time": 115.8, "best_lap_time": 115.2,
            "session_laps_remain": 20, "incident_count": 0, "track_temp_c": 28
        },
        "expected_keywords": ["gap", "push", "attack", "ahead", "p3"],
    },
    {
        "name": "Tire Warning - Barcelona",
        "input": {
            "car": "Porsche 911 GT3 R", "car_class": "GT3",
            "car_traits": ["rear_engine", "technical", "trail_brake_critical"],
            "track": "Barcelona", "track_type": "technical",
            "lap": 18, "lap_pct": 0.35, "position": 7,
            "fuel_laps_remaining": 10,
            "tire_wear": {"fl": 35, "fr": 45, "rl": 30, "rr": 38},
            "tire_temps": {"fl": 98, "fr": 108, "rl": 92, "rr": 96},
            "gap_ahead": 3.2, "gap_behind": 1.8,
            "last_lap_time": 103.5, "best_lap_time": 102.8,
            "session_laps_remain": 12, "incident_count": 1, "track_temp_c": 38
        },
        "expected_keywords": ["tire", "front", "wear", "manage", "hot"],
    },
    {
        "name": "Cold Tires - Spa",
        "input": {
            "car": "BMW M4 GT3", "car_class": "GT3",
            "car_traits": ["front_engine", "balanced", "forgiving"],
            "track": "Spa", "track_type": "high_speed",
            "lap": 1, "lap_pct": 0.1, "position": 12,
            "fuel_laps_remaining": 25,
            "tire_wear": {"fl": 0, "fr": 0, "rl": 0, "rr": 0},
            "tire_temps": {"fl": 45, "fr": 48, "rl": 42, "rr": 44},
            "gap_ahead": 1.5, "gap_behind": 0.8,
            "last_lap_time": None, "best_lap_time": None,
            "session_laps_remain": 30, "incident_count": 0, "track_temp_c": 18
        },
        "expected_keywords": ["cold", "warm", "careful", "temperature", "build"],
    },
    {
        "name": "Clean Air Push - Monza",
        "input": {
            "car": "Lamborghini Huracan GT3", "car_class": "GT3",
            "car_traits": ["mid_engine", "awd", "stable"],
            "track": "Monza", "track_type": "high_speed",
            "lap": 10, "lap_pct": 0.6, "position": 2,
            "fuel_laps_remaining": 18,
            "tire_wear": {"fl": 12, "fr": 15, "rl": 10, "rr": 12},
            "tire_temps": {"fl": 88, "fr": 92, "rl": 84, "rr": 86},
            "gap_ahead": 4.5, "gap_behind": 3.2,
            "last_lap_time": 108.2, "best_lap_time": 107.8,
            "session_laps_remain": 15, "incident_count": 0, "track_temp_c": 30
        },
        "expected_keywords": ["clean", "push", "gap", "good", "p2"],
    },
]


async def run_eval():
    url = "http://127.0.0.1:1234/v1"
    system_prompt = "You are a race engineer. Given the car, track, and telemetry, provide a brief callout to the driver."

    print("=" * 70)
    print("FINE-TUNED MODEL EVALUATION via LM Studio")
    print("=" * 70)

    results = []

    for test in TEST_CASES:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(test["input"])},
            ],
            "max_tokens": 50,
            "temperature": 0.5,
        }

        start = time.perf_counter()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(f"{url}/chat/completions", json=payload) as resp:
                latency = (time.perf_counter() - start) * 1000
                data = await resp.json()
                response = data["choices"][0]["message"]["content"]

        word_count = len(response.split())
        has_keyword = any(kw.lower() in response.lower() for kw in test["expected_keywords"])
        is_concise = 15 <= word_count <= 40

        results.append({
            "name": test["name"],
            "response": response,
            "latency_ms": latency,
            "word_count": word_count,
            "is_concise": is_concise,
            "has_expected_keyword": has_keyword,
        })

        status = "PASS" if (is_concise and has_keyword) else "PARTIAL" if (is_concise or has_keyword) else "FAIL"
        print(f"\n[{status}] {test['name']}")
        print(f"  Latency: {latency:.0f}ms | Words: {word_count} | Keyword match: {has_keyword}")
        print(f"  Response: {response}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r["is_concise"] and r["has_expected_keyword"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    avg_words = sum(r["word_count"] for r in results) / len(results)

    print(f"Passed: {passed}/{len(results)}")
    print(f"Avg latency: {avg_latency:.0f}ms")
    print(f"Avg word count: {avg_words:.1f}")
    print(f"Concise (15-40 words): {sum(1 for r in results if r['is_concise'])}/{len(results)}")
    print(f"Has expected keywords: {sum(1 for r in results if r['has_expected_keyword'])}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(run_eval())
