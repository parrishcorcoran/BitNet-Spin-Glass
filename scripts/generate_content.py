"""Content generation: blog post + X/Twitter thread from analysis results.

Reads analysis JSON + figure paths, calls Claude API to write:
1. A blog post (Markdown) ready to publish
2. An X/Twitter thread ready to post

Outputs saved as files in content/ directory.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"

Usage:
    python -m scripts.generate_content \
        --results results/bitnet2b/layer_analysis.json \
        --figures-dir figures \
        --phase bitnet2b

    python -m scripts.generate_content \
        --results results/measurements/summary.json \
        --figures-dir figures \
        --phase replica_training
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic


# ---------------------------------------------------------------------------
# Tags and handles — edit these for your accounts
# ---------------------------------------------------------------------------

TAGS = {
    # Your accounts
    "author": "@SpinGlassAI",
    "author_name": "SpinGlassAI",
    "blog_url": "https://yourblog.com",    # <-- CHANGE THIS to your blog URL
    "github_repo": "https://github.com/parrishcorcoran/BitNet-Spin-Glass",

    # People/orgs to tag in X posts
    "relevant_tags": [
        "@Microsoft",           # BitNet creators
        "@1bitLLM",             # BitNet team account
        "@MicrosoftResearch",   # Microsoft Research
    ],

    # Hashtags
    "hashtags": [
        "#BitNet",
        "#SpinGlass",
        "#MachineLearning",
        "#Physics",
        "#TernaryNetworks",
        "#arXiv",
    ],
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _build_blog_prompt(results_json: str, phase: str, figure_list: list[str]) -> str:
    figures_str = "\n".join(f"  - {f}" for f in figure_list)
    return f"""You are a science communicator writing about a novel research finding.
Write a blog post about the following spin glass physics analysis of neural networks.

## Context
This is the first-ever spin glass physics measurement on ternary {{-1, 0, +1}} neural networks
(BitNet-style). The research applies statistical physics tools (frustration, overlap distributions,
aging) to quantized neural networks.

## Phase: {phase}
{"BitNet 2B4T Analysis — We analyzed Microsoft's real 2-billion-parameter BitNet model for spin glass signatures." if phase == "bitnet2b" else "Replica Training Results — We trained thousands of small models and measured physics observables."}

## Raw Results (JSON)
```json
{results_json}
```

## Figures Available
{figures_str}

## Instructions
1. Write in an accessible but technically accurate style
2. The audience is ML researchers and physics-curious engineers
3. Explain what each measurement means in plain English, then give the numbers
4. Reference the figures by filename (they'll be embedded in the blog)
5. Include a "Key Findings" section with bullet points
6. Include a "What This Means" section explaining significance
7. Keep it under 1500 words
8. Use markdown formatting
9. Title should be catchy but accurate
10. Include a brief methods section
11. End with "What's Next" section about upcoming experiments
12. Link to the GitHub repo: {TAGS["github_repo"]}

Write the blog post now."""


def _build_x_post_prompt(results_json: str, phase: str, key_findings: str) -> str:
    tags_str = " ".join(TAGS["relevant_tags"])
    hashtags_str = " ".join(TAGS["hashtags"][:4])  # Keep it tight

    return f"""You are writing an X/Twitter thread about a novel research finding.

## Context
First-ever spin glass physics measurements on ternary {{-1, 0, +1}} neural networks (BitNet-style).

## Phase: {phase}

## Key findings summary
{key_findings}

## Instructions
1. Write a thread of 3-5 tweets (each under 280 characters)
2. First tweet should be the hook — make it compelling
3. Include key numbers and findings
4. Tag these accounts: {tags_str}
5. Use these hashtags (spread across tweets, don't dump all in one): {hashtags_str}
6. Last tweet should link to the GitHub repo: {TAGS["github_repo"]}
7. Format as:
   🧵 1/ [first tweet]

   2/ [second tweet]

   3/ [third tweet]
   ...

8. Be precise with numbers — don't round or exaggerate
9. Keep the tone: excited but scientifically rigorous

Write the X thread now."""


# ---------------------------------------------------------------------------
# Analysis summary extraction
# ---------------------------------------------------------------------------

def extract_key_findings(results: list[dict], phase: str) -> str:
    """Extract a plain-text summary of key findings from results JSON."""
    if phase == "bitnet2b":
        import numpy as np
        frustrations = [r["frustration_index"] for r in results]
        diffs = [r.get("frustration_vs_random", 0) for r in results]
        vacancies = [r["vacancy_fraction"] for r in results]

        n_below = sum(1 for d in diffs if d < 0)
        lines = [
            f"Analyzed {len(results)} weight matrices from Microsoft BitNet 2B4T (2B params)",
            f"Mean frustration index: {np.mean(frustrations):.4f} (range {np.min(frustrations):.4f}-{np.max(frustrations):.4f})",
            f"{n_below}/{len(results)} layers are LESS frustrated than random ternary baseline",
            f"Mean frustration difference vs random: {np.mean(diffs):.4f}",
            f"Mean vacancy (zero fraction): {np.mean(vacancies)*100:.1f}%",
            f"This suggests the trained model has learned to reduce frustration compared to random ternary matrices",
        ]
        return "\n".join(lines)
    else:
        return json.dumps(results, indent=2)[:2000]


# ---------------------------------------------------------------------------
# Claude API calls
# ---------------------------------------------------------------------------

def call_claude(prompt: str, max_tokens: int = 4096) -> str:
    """Call Claude API and return the text response."""
    client = anthropic.Anthropic()

    # Use streaming to avoid timeouts on long responses
    with client.messages.stream(
        model="claude-sonnet-4-6",  # Sonnet for cost efficiency on content generation
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    # Extract text from response
    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)

    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blog post + X thread from analysis results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to analysis results JSON")
    parser.add_argument("--figures-dir", type=str, default="figures",
                        help="Directory containing figure PNGs")
    parser.add_argument("--phase", type=str, default="bitnet2b",
                        choices=["bitnet2b", "replica_training", "lr_sweep"],
                        help="Which experiment phase")
    parser.add_argument("--output-dir", type=str, default="content",
                        help="Directory to write blog/X posts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    args = parser.parse_args()

    # Check API key
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable first.")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("")
        print("Get your key at: https://console.anthropic.com/settings/keys")
        sys.exit(1)

    # Load results
    print(f"Loading results from {args.results}")
    with open(args.results) as f:
        results = json.load(f)

    # Find figures
    figures_dir = Path(args.figures_dir)
    figure_files = sorted(figures_dir.glob(f"{args.phase}*.png")) if figures_dir.exists() else []
    figure_names = [f.name for f in figure_files]
    print(f"Found {len(figure_names)} figures: {figure_names}")

    # Extract key findings
    import numpy as np
    key_findings = extract_key_findings(results, args.phase)
    print(f"\nKey findings:\n{key_findings}\n")

    # Build prompts
    results_json = json.dumps(results, indent=2)
    # Truncate if too long (keep under 50K chars for API)
    if len(results_json) > 50000:
        results_json = results_json[:50000] + "\n... (truncated)"

    blog_prompt = _build_blog_prompt(results_json, args.phase, figure_names)
    x_prompt = _build_x_post_prompt(results_json, args.phase, key_findings)

    if args.dry_run:
        print("=" * 60)
        print("BLOG PROMPT:")
        print("=" * 60)
        print(blog_prompt[:500] + "...")
        print()
        print("=" * 60)
        print("X POST PROMPT:")
        print("=" * 60)
        print(x_prompt[:500] + "...")
        return

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate blog post
    print("Generating blog post (calling Claude API)...")
    blog_text = call_claude(blog_prompt, max_tokens=8000)

    blog_path = out_dir / f"blog_{args.phase}_{timestamp}.md"
    with open(blog_path, "w") as f:
        f.write(blog_text)
    print(f"  Blog saved: {blog_path}")

    # Generate X thread
    print("Generating X thread (calling Claude API)...")
    x_text = call_claude(x_prompt, max_tokens=2000)

    x_path = out_dir / f"x_thread_{args.phase}_{timestamp}.md"
    with open(x_path, "w") as f:
        f.write(x_text)
    print(f"  X thread saved: {x_path}")

    # Also save a combined summary
    summary_path = out_dir / f"summary_{args.phase}_{timestamp}.md"
    with open(summary_path, "w") as f:
        f.write(f"# Content Generated: {args.phase}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Key Findings\n{key_findings}\n\n")
        f.write(f"## Figures\n")
        for fig in figure_names:
            f.write(f"- `{fig}`\n")
        f.write(f"\n---\n\n")
        f.write(f"## Blog Post\n\n{blog_text}\n\n")
        f.write(f"---\n\n")
        f.write(f"## X Thread\n\n{x_text}\n")
    print(f"  Combined summary: {summary_path}")

    print(f"\n{'='*60}")
    print("DONE! Your content is ready:")
    print(f"  Blog:    {blog_path}")
    print(f"  X thread: {x_path}")
    print(f"  Summary:  {summary_path}")
    print(f"{'='*60}")

    # Print the X thread to terminal for quick copy
    print(f"\n{'='*60}")
    print("X THREAD (copy/paste ready):")
    print(f"{'='*60}")
    print(x_text)


if __name__ == "__main__":
    main()
