#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Abhivyakti — GitHub Repository Setup Script
# Run this ONCE after cloning / creating your project folder
#
# Usage:
#   chmod +x setup_git.sh
#   ./setup_git.sh
# ─────────────────────────────────────────────────────────────

echo ""
echo "🤟 Abhivyakti — GitHub Setup"
echo "=============================="

# ── Step 1: Initialize git repo ──────────────────────────────
git init
echo "✅ Git initialized"

# ── Step 2: Set your identity (edit these) ───────────────────
# git config user.name "Your Name"
# git config user.email "your@email.com"

# ── Step 3: Initial commit ───────────────────────────────────
git add .
git commit -m "feat: initial project structure — Abhivyakti ISL Sign to Speech"
echo "✅ Initial commit created"

# ── Step 4: Create main branch ───────────────────────────────
git branch -M main

# ── Step 5: Add remote (edit with your GitHub URL) ───────────
# git remote add origin https://github.com/YOUR_USERNAME/abhivyakti.git

# ── Step 6: Push to GitHub ───────────────────────────────────
# git push -u origin main

echo ""
echo "📋 Next steps:"
echo "   1. Create repo on github.com/new  (name: abhivyakti)"
echo "   2. Uncomment and run the git remote + push lines above"
echo "   3. Verify at: https://github.com/YOUR_USERNAME/abhivyakti"
echo ""
echo "🏷️  After pushing, create your first release tag:"
echo "   git tag v0.1-pipeline"
echo "   git push origin v0.1-pipeline"
