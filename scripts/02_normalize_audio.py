import os
import subprocess

# --- Configuration ---
# The folder with your original MP3 downloads
SOURCE_FOLDER = "data/01_raw/speech/"

# The folder where the converted WAV files will be saved
DESTINATION_FOLDER = "data/01_raw/normalized/"

# Desired audio properties
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1         # Mono

# --- Main Script ---
def normalize_audio_files():
    """
    Converts all audio files in the source folder to a standard WAV format
    using FFmpeg and saves them in the destination folder.
    """
    # 1. Create the destination folder if it doesn't exist
    os.makedirs(DESTINATION_FOLDER, exist_ok=True)
    print(f"Ensured destination folder exists: {DESTINATION_FOLDER}")

    # 2. Get the list of files to convert
    try:
        files_to_convert = os.listdir(SOURCE_FOLDER)
    except FileNotFoundError:
        print(f"❌ ERROR: Source folder not found: {SOURCE_FOLDER}")
        print("Please make sure you have your raw audio in the correct directory.")
        return

    print(f"Found {len(files_to_convert)} files to process in {SOURCE_FOLDER}")

    # 3. Loop through each file and convert it
    for filename in files_to_convert:
        # Construct full input path
        input_path = os.path.join(SOURCE_FOLDER, filename)

        # Skip directories, just in case
        if not os.path.isfile(input_path):
            continue

        # Construct the new filename with a .wav extension
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.wav"
        output_path = os.path.join(DESTINATION_FOLDER, output_filename)

        print(f"\nProcessing '{filename}'...")

        # 4. Build and run the FFmpeg command
        command = [
            'ffmpeg',
            '-i', input_path,        # Input file
            '-ar', str(SAMPLE_RATE), # Set audio sample rate (16kHz)
            '-ac', str(CHANNELS),    # Set number of audio channels (mono)
            '-y',                    # Overwrite output file if it exists
            output_path
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"  -> ✅ Successfully converted to '{output_path}'")
        except subprocess.CalledProcessError as e:
            print(f"  -> ❌ Failed to convert '{filename}'.")
            print(f"  -> FFmpeg error:\n{e.stderr}")

    print("\nNormalization process complete.")


if __name__ == "__main__":
    normalize_audio_files()
