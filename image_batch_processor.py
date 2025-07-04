#!/usr/bin/env python3
"""
Image Batch Processor with Llama API
====================================

This script processes a folder of images using the Llama API for image analysis.
It can analyze images in batches and generate comprehensive reports.

Usage: python image_batch_processor.py
"""

import base64
import os
import json
import time
import sys
from pathlib import Path

# Add parent directory to path to import llama client
sys.path.append(str(Path(__file__).parent))

try:
    from llama_api_client import LlamaAPIClient
except ImportError:
    print("❌ llama-api-client not found. Please install it first:")
    print("   pip install llama-api-client")
    sys.exit(1)


def image_to_base64(image_path: str) -> str:
    """Converts an image to base64 string
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error converting {image_path} to base64: {e}")
        return ""


def process_folder(folder_path: str, chunk_size: int = 5, skip_frames: int = 100) -> None:
    """Processes an entire folder of images using Llama API
    
    Args:
        folder_path: Path to folder containing images
        chunk_size: Number of images to process in each API call
        skip_frames: Skip every N frames (for video frame analysis)
    """
    
    # Validate folder path
    if not os.path.exists(folder_path):
        print(f"❌ Folder path does not exist: {folder_path}")
        return
    
    # Initialize the Llama API client
    try:
        api_key = os.environ.get("LLAMA_API_KEY")
        if not api_key:
            print("⚠️ No LLAMA_API_KEY found in environment variables.")
            print("🔧 Attempting to use client without explicit API key...")
        
        client = LlamaAPIClient()
        print("✅ Llama API client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Llama API client: {e}")
        return

    # Get a list of all image files in the folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    if not image_files:
        print(f"❌ No image files found in {folder_path}")
        return
    
    print(f"📸 Found {len(image_files)} image files")
    print(f"⚙️ Processing in chunks of {chunk_size}, skipping every {skip_frames} frames")

    # Initialize a list to store the analysis results
    analysis_results = []
    processed_count = 0
    total_chunks = (len(image_files) + chunk_size - 1) // chunk_size

    # Filter images based on skip_frames parameter first
    if skip_frames > 1:
        filtered_images = [image_files[i] for i in range(0, len(image_files), skip_frames)]
        print(f"🎯 Filtered to {len(filtered_images)} images (every {skip_frames} frames)")
    else:
        filtered_images = image_files
    
    total_chunks = (len(filtered_images) + chunk_size - 1) // chunk_size

    # Process the filtered images in chunks
    for i in range(0, len(filtered_images), chunk_size):
        chunk = filtered_images[i:i+chunk_size]
        chunk_number = i // chunk_size + 1

        print(f"\n🔄 Processing chunk {chunk_number} of {total_chunks}...")
        print(f"   Images: {', '.join(chunk)}")

        # Create messages for the Llama API
        messages = []
        valid_images = []
        
        for image_file in chunk:
            image_path = os.path.join(folder_path, image_file)
            base64_image = image_to_base64(image_path)
            
            if base64_image:  # Only process if base64 conversion was successful
                valid_images.append(image_file)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image '{image_file}' and describe what you see. Include details about objects, people, scenes, colors, composition, and any notable features.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                })
        
        if not valid_images:
            print("   ⚠️ No valid images in this chunk, skipping...")
            continue

        try:
            # Create a completion request for the Llama API
            response = client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=messages,
                max_completion_tokens=2048,
                temperature=0.7,
            )

            # Process the response
            if hasattr(response, 'completion_message') and response.completion_message:
                content = response.completion_message.content
                
                # Handle different response formats
                if isinstance(content, list):
                    for j, message in enumerate(content):
                        if j < len(valid_images):
                            result_text = message.text if hasattr(message, 'text') else str(message)
                            print(f"   📝 {valid_images[j]}: {result_text[:100]}...")
                            analysis_results.append({
                                "image": valid_images[j],
                                "analysis": result_text,
                                "chunk": chunk_number,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                elif isinstance(content, str):
                    print(f"   📝 Batch analysis: {content[:100]}...")
                    analysis_results.append({
                        "images": valid_images,
                        "analysis": content,
                        "chunk": chunk_number,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                elif hasattr(content, 'text'):
                    result_text = content.text
                    print(f"   📝 Batch analysis: {result_text[:100]}...")
                    analysis_results.append({
                        "images": valid_images,
                        "analysis": result_text,
                        "chunk": chunk_number,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    print(f"   ⚠️ Unexpected response format: {type(content)}")
                    analysis_results.append({
                        "images": valid_images,
                        "analysis": str(content),
                        "chunk": chunk_number,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
            else:
                print("   ⚠️ No completion message in response")

            processed_count += len(valid_images)

        except Exception as e:
            print(f"   ❌ Error processing chunk {chunk_number}: {e}")
            # Add error entry to results
            analysis_results.append({
                "images": valid_images,
                "error": str(e),
                "chunk": chunk_number,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        # Print progress bar
        progress = processed_count / len(filtered_images)
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   📊 Progress: [{bar}] {int(progress*100)}% ({processed_count}/{len(filtered_images)})")

        # Wait to avoid overwhelming the API
        print("   ⏳ Waiting 2 seconds...")
        time.sleep(2)

    # Save the analysis results to a JSON file
    results_file = 'analysis_results.json'
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Analysis results saved to {results_file}")
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")
        return

    # Analyze the JSON file and produce an output.md file
    if analysis_results:
        print("\n🎬 Generating Steven Spielberg's REALITY-CHECK critique...")
        analyze_json_and_transcript(folder_path)
    else:
        print("\n⚠️ No analysis results to process")


def find_transcript_file(folder_path: str) -> str:
    """Find transcript file in the same folder as images"""
    transcript_extensions = ('.txt', '.srt', '.vtt', '.transcript')
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(transcript_extensions):
            transcript_path = os.path.join(folder_path, file)
            print(f"📄 Found transcript file: {file}")
            return transcript_path
    
    print("⚠️ No transcript file found in folder")
    return ""


def analyze_json_and_transcript(folder_path: str) -> None:
    """Analyzes the JSON file and transcript with Spielberg-style critique"""
    
    # Load the analysis results from the JSON file
    results_file = 'analysis_results.json'
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            analysis_results = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {results_file}: {e}")
        return

    if not analysis_results:
        print("⚠️ No analysis results found in JSON file")
        return
    
    # Load transcript if available
    transcript_content = ""
    transcript_file = find_transcript_file(folder_path)
    if transcript_file:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            print("✅ Transcript loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading transcript: {e}")
            transcript_content = ""

    # Initialize the Llama API client
    try:
        client = LlamaAPIClient()
    except Exception as e:
        print(f"❌ Failed to initialize Llama API client for analysis: {e}")
        return

    # Prepare the analysis data for the AI as a video sequence
    analysis_text = "VIDEO SEQUENCE ANALYSIS:\n"
    analysis_text += "========================\n\n"
    
    for i, result in enumerate(analysis_results, 1):
        if "error" in result:
            analysis_text += f"FRAME {i}: ERROR - {result['error']}\n\n"
        elif "images" in result:
            # Extract frame numbers from filenames for better context
            frame_info = []
            for img in result["images"]:
                # Try to extract frame number from filename
                import re
                frame_match = re.search(r'(\d{4})', img)
                frame_num = frame_match.group(1) if frame_match else str(i)
                frame_info.append(f"Frame {frame_num}")
            
            frames_desc = ", ".join(frame_info)
            analysis_text += f"SEQUENCE {i} ({frames_desc}):\n"
            analysis_text += f"{result.get('analysis', 'No analysis')}\n\n"
        elif "image" in result:
            # Extract frame number from filename
            import re
            frame_match = re.search(r'(\d{4})', result['image'])
            frame_num = frame_match.group(1) if frame_match else str(i)
            analysis_text += f"FRAME {frame_num} ({result['image']}):\n"
            analysis_text += f"{result.get('analysis', 'No analysis')}\n\n"

    # Add transcript to the analysis if available
    transcript_section = ""
    if transcript_content:
        transcript_section = f"\n\nAUDIO TRANSCRIPT:\n================\n{transcript_content}\n\n"
    
    # Create a brutally honest but contextual Spielberg prompt
    message = {
        "role": "user",
        "content": f"""You are Steven Spielberg reviewing a FINISHED video project. Here's the FULL CONTEXT you need to understand:

THE CREATOR'S SITUATION:
- 100% SELF-TAUGHT (no film school, no training, learning by trial and error)
- COMPLETELY UNPAID (doing this for free, zero budget, passion project)
- GETS RANDOM FOOTAGE from "Stosh" with NO advance notice (could be anything)
- Had roughly 2 HOURS TOTAL to turn this into something deliverable
- NO CHOICE but to produce SOMETHING (can't refuse or say "this is impossible")
- Working ALONE with whatever tools they have access to
- NO CONTROL over source material quality, content, or timing
- MUST DELIVER regardless of what garbage they receive

THE IMPOSSIBLE EQUATION:
Random garbage footage + 2 hours + self-taught + unpaid + no choice = ???

CONTEXT: You're seeing their FINAL ATTEMPT after 2 hours of frantically editing, animating, sound designing, and rendering. They got handed random trash and HAD to make something watchable in 2 hours for FREE.

Your job: Judge this REALISTICALLY. Is this good enough for someone in this impossible situation?

# SPIELBERG'S BRUTALLY HONEST REALITY-CHECK
*"Did You Actually Pull This Off Given Your Constraints?"*

## IMMEDIATE GUT REACTION
[First impression: Given the constraints (2 hours, unpaid, self-taught, random garbage footage), is this actually watchable or still unwatchable?]

## THE 2-HOUR MIRACLE ASSESSMENT
[What did they actually accomplish in 2 hours? Did they use their time smartly, or did they waste it on the wrong things?]

## SELF-TAUGHT SURVIVAL SKILLS
[For someone with no training working under pressure, what techniques did they attempt? Were their instincts right?]

## TECHNICAL TRIAGE UNDER FIRE
[Given the time crunch and skill level, how's the technical execution? What corners did they smartly cut vs. what hurt them?]

## THE "SOMETHING FROM NOTHING" VERDICT
[The core question: Did they successfully turn garbage into something deliverable, or is this still unwatchable garbage?]

## IMPOSSIBLE DEADLINE DECISIONS
[Were their creative choices appropriate for someone with 2 hours and no budget? Did they prioritize correctly under pressure?]

## HARSH BUT FAIR RATING
[Rate 1-10 considering: self-taught + unpaid + 2 hours + random garbage + must deliver. Is this a miracle or a disaster?]

## REAL-WORLD ADVICE FOR THE HUSTLE
[What should they focus on to survive future impossible deadlines? What would actually help in this situation?]

---

IMPORTANT: Judge this with FULL CONTEXT. They're not Pixar with unlimited time and budget. They're a self-taught person with 2 hours trying to make random garbage into something not embarrassing for FREE. Give credit where it's due, but be brutally honest about what doesn't work.

This is the reality of content creation hustle - impossible deadlines, no resources, random material, but you MUST deliver something.

Here's what they managed to create in 2 hours with garbage footage:

{analysis_text}
{transcript_section}

Final question: Given the impossible constraints, did they actually pull this off? Be honest - is this good enough to not be embarrassing, or should they have just given up?"""
    }

    try:
        # Create a completion request for the Llama API
        print("🎭 Consulting with Steven Spielberg (reality-check mode)...")
        response = client.chat.completions.create(
            model="Llama-4-Scout-17B-16E-Instruct-FP8",  # Using Scout model as requested
            messages=[message],
            max_completion_tokens=4000,
            temperature=0.9,  # Higher temperature for more personality
        )

        # Extract the response content
        content = ""
        if hasattr(response, 'completion_message') and response.completion_message:
            if hasattr(response.completion_message, 'content'):
                if isinstance(response.completion_message.content, list):
                    content = response.completion_message.content[0].text if response.completion_message.content else ""
                elif hasattr(response.completion_message.content, 'text'):
                    content = response.completion_message.content.text
                else:
                    content = str(response.completion_message.content)
            else:
                content = str(response.completion_message)
        else:
            content = str(response)

        # Save the response to an output.md file
        output_file = 'spielberg_postproduction_critique.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"🎬 Steven Spielberg's REALITY-CHECK critique saved to {output_file}")
        
        # Print a brief summary
        lines = content.split('\n')
        summary_lines = [line for line in lines[:15] if line.strip()]
        if summary_lines:
            print("\n📝 Spielberg's realistic verdict:")
            for line in summary_lines[:5]:
                print(f"   {line}")
            if len(summary_lines) > 5:
                print("   ...")

    except Exception as e:
        print(f"❌ Error generating Spielberg analysis: {e}")


def main():
    """Main function to run the image batch processor"""
    
    print("🎬 Image Batch Processor with Llama API")
    print("=" * 50)
    
    # Get folder path from user
    while True:
        folder_path = input("📁 Enter the folder path containing images: ").strip()
        if folder_path.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            return
        
        if os.path.exists(folder_path):
            break
        else:
            print(f"❌ Folder not found: {folder_path}")
            print("   Please enter a valid folder path or 'quit' to exit.")
    
    # Get processing parameters
    try:
        chunk_size = int(input("📦 Chunk size (images per API call, default 3): ") or "3")
        skip_frames = int(input("⏭️  Skip frames (process every Nth image, default 1): ") or "1")
    except ValueError:
        print("⚠️ Invalid input, using defaults: chunk_size=3, skip_frames=1")
        chunk_size = 3
        skip_frames = 1
    
    print("\n🚀 Starting processing...")
    print(f"   Folder: {folder_path}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Skip frames: {skip_frames}")
    
    # Process the folder
    start_time = time.time()
    process_folder(folder_path, chunk_size, skip_frames)
    end_time = time.time()
    
    print(f"\n✅ Processing completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
