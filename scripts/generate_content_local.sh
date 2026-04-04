#!/bin/bash
# Generate blog + X thread content using local Claude Code.
#
# No API key needed — uses the Claude instance running in your terminal.
#
# Usage:
#   1. Open Claude Code in a terminal: claude
#   2. Tell Claude: "Read and follow scripts/generate_content_local.sh"
#
# Claude will read this file and follow the instructions below.
# ============================================================

cat << 'INSTRUCTIONS'
============================================
  Content Generation via Local Claude Code
============================================

You DON'T need to run this script with bash.
Instead, tell your local Claude Code instance:

  "Read and follow scripts/generate_content_local.sh"

Claude will then:
  1. Read the analysis results
  2. Write a blog post
  3. Write an X/Twitter thread
  4. Save both to content/

============================================
INSTRUCTIONS

# ============================================================
# INSTRUCTIONS FOR CLAUDE CODE (Claude reads this directly)
# ============================================================
#
# Hey Claude — follow these steps:
#
# STEP 1: Read the analysis results
#   Read the file: results/bitnet2b/layer_analysis.json
#
# STEP 2: Read the key figures list
#   Run: ls figures/bitnet2b_*.png
#
# STEP 3: Write a BLOG POST
#   Write a blog post (Markdown) and save it to content/blog_bitnet2b.md
#   Guidelines:
#   - Title should be catchy but scientifically accurate
#   - Audience: ML researchers and physics-curious engineers
#   - Explain each measurement in plain English, then give the numbers
#   - Reference the figure filenames (they'll be embedded in the blog)
#   - Include sections: Key Findings (bullets), What This Means, Methods, What's Next
#   - Keep it under 1500 words
#   - Link to GitHub: https://github.com/parrishcorcoran/BitNet-Spin-Glass
#   - This is the FIRST-EVER spin glass physics analysis of a real production
#     BitNet model. Nobody has done this before. Make that clear.
#   - Context: We analyzed Microsoft's 2-billion-parameter BitNet b1.58 2B4T
#     model using tools from statistical physics (frustration index, vacancy
#     fraction, weight distribution). We compared each layer's frustration
#     against random ternary matrices with matched sparsity.
#
# STEP 4: Write an X/TWITTER THREAD
#   Write a thread (3-5 tweets, each under 280 chars) and save to content/x_thread_bitnet2b.md
#   Guidelines:
#   - First tweet is the hook — make it compelling
#   - Include actual numbers from the results
#   - Tag: @Microsoft @1bitLLM @MicrosoftResearch
#   - Author handle: @SpinGlassAI
#   - Hashtags (spread across tweets): #BitNet #SpinGlass #MachineLearning #Physics
#   - Last tweet links to GitHub repo
#   - Format as:
#     🧵 1/ [tweet]
#
#     2/ [tweet]
#
#     3/ [tweet]
#   - Be precise with numbers — don't round or exaggerate
#   - Tone: excited but scientifically rigorous
#
# STEP 5: Print the X thread to the terminal so I can copy it immediately
#
# STEP 6: Tell me the files were saved and where to find them
