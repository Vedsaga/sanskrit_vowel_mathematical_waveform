import os
import subprocess
import csv
from datetime import date

# --- Configuration ---
LINKS_FILE = "config/download_speech_audio_links.csv"

OUTPUT_PATH = "data/01_raw/speech/"
TRACKER_FILE = "data/download_tracker.csv"

# --- Main Script ---
def read_links_from_csv():
    """Reads the links and filenames from the specified CSV file."""
    links = []
    try:
        with open(LINKS_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f) # Reads rows as dictionaries
            for row in reader:
                links.append({'url': row['url'], 'filename': row['filename']})
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{LINKS_FILE}' was not found.")
        print("Please create it in the root of your project.")
    return links

def get_existing_filenames():
    """Returns a set of filenames that are already in the output path."""
    if not os.path.exists(OUTPUT_PATH):
        return set()
    return set(os.listdir(OUTPUT_PATH))

def update_tracker(url, filename):
    """Appends a new record to the tracker CSV."""
    file_exists = os.path.exists(TRACKER_FILE)
    with open(TRACKER_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['url', 'filename', 'download_date'])
        writer.writerow([url, filename, date.today().isoformat()])

def download_audio_with_yt_dlp():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    links_to_download = read_links_from_csv()

    if not links_to_download:
        return # Stop if the links file is missing or empty

    existing_files = get_existing_filenames()

    print("Starting intelligent download...")
    print(f"{len(existing_files)} files already exist in the target folder.")

    for item in links_to_download:
        url = item['url']
        filename = item['filename']

        if filename in existing_files:
            print(f"\n-> Skipping '{filename}'. File already exists.")
            continue

        print(f"\n-> Processing new file: '{filename}' from {url}")

        output_template = os.path.join(OUTPUT_PATH, os.path.splitext(filename)[0] + '.%(ext)s')

        try:
            command = [
                'yt-dlp',
                '-x', '--audio-format', 'mp3',
                '-o', output_template,
                url
            ]
            subprocess.run(command, check=True)

            downloaded_file = os.path.splitext(output_template)[0] + '.mp3'
            final_path = os.path.join(OUTPUT_PATH, filename)
            if os.path.exists(downloaded_file) and downloaded_file != final_path:
                 os.rename(downloaded_file, final_path)

            print(f"  -> ✅ Download complete for '{filename}'")
            update_tracker(url, filename)

        except Exception as e:
            print(f"  -> ❌ An error occurred: {e}")

    print("\nDownload process complete.")

if __name__ == "__main__":
    download_audio_with_yt_dlp()
