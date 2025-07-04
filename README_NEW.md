# 🎬 Video Critique Tool

Transform your video frames into professional Spielberg-style critiques using AI analysis.

https://youtu.be/2ig7244Mixs

## 🚀 Quick Start (3 Steps)

1. **Activate Environment:**
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Tool:**
   ```bash
   python image_batch_processor.py
   ```

That's it! The tool will guide you through everything else.

## 📋 What You Need

- **Video frames** (exported as JPG/PNG images)
- **Llama API key** (set as `LLAMA_API_KEY` environment variable)
- **Optional:** Transcript file in the same folder

## 🎯 What It Does

1. Analyzes your video frames using AI vision
2. Includes your transcript (if available)
3. Generates a Spielberg-style critique considering your real constraints

## 🎬 Example Output

Creates `spielberg_postproduction_critique.md` with honest feedback like:

> **Rating: 7/10** - "Given the constraints (2 hours, unpaid, self-taught, random garbage footage), this is actually a surprisingly watchable and coherent video."

## 🔧 Set Your API Key

```bash
set LLAMA_API_KEY=your-api-key-here


##NOTE!!!!
VIDEO MUST BE IN A FOLDER AND BROKEN DOWN INTO AN IMAGE SEQUENCE & Transcript (if you have whisper locally or via an api)

IF YOU DONT KNOW HOW TO DO THIS:

ffmpeg -i input.mp4 %04d.jpg 

```

**Made for content creators who hustle with impossible deadlines! 🎯**
