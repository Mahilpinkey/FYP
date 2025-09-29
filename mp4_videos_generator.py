import pandas as pd
import os
import yt_dlp
import time
import subprocess
import sys
import chardet

def install_required_packages():
    """Install required packages if not already installed"""
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp
    
    try:
        import pandas
    except ImportError:
        print("Installing pandas...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas
        
    try:
        import chardet
    except ImportError:
        print("Installing chardet...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
        import chardet

def detect_encoding(file_path):
    """Detect the encoding of a file"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']

def read_csv_with_encoding(file_path):
    """Read CSV file with automatic encoding detection"""
    try:
        # First try utf-8
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            # Try to detect encoding
            encoding = detect_encoding(file_path)
            print(f"Detected encoding: {encoding}")
            return pd.read_csv(file_path, encoding=encoding)
        except:
            # Try common encodings
            encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    print(f"Trying encoding: {encoding}")
                    return pd.read_csv(file_path, encoding=encoding)
                except:
                    continue
            raise Exception("Could not read CSV file with any encoding")

def download_youtube_videos(csv_file_path, output_folder):
    """
    Download YouTube videos from CSV file and save them as MP4 files
    
    Args:
        csv_file_path (str): Path to the CSV file
        output_folder (str): Folder to save downloaded videos
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")
    
    # Read CSV file with encoding detection
    try:
        df = read_csv_with_encoding(csv_file_path)
        print(f"Successfully read CSV file: {csv_file_path}")
        print(f"Found {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we have the required columns
        if 'Word' not in df.columns or 'YouTube_Link' not in df.columns:
            print("Error: CSV must contain 'Word' and 'YouTube_Link' columns")
            return None
        
        # Add a column for local file path if it doesn't exist
        if 'Local_File_Path' not in df.columns:
            df['Local_File_Path'] = ''
            
        # Show first few rows for verification
        print("\nFirst few rows of CSV:")
        print(df[['Word', 'YouTube_Link']].head())
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # YouTube downloader options
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(output_folder, '%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'socket_timeout': 30,
        'retries': 3,
    }
    
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0
    
    print("\nStarting video downloads...")
    
    for index, row in df.iterrows():
        word = str(row['Word']).strip()
        youtube_link = str(row['YouTube_Link']).strip()
        
        # Skip empty links
        if not youtube_link or pd.isna(youtube_link) or youtube_link == 'nan' or youtube_link == 'None':
            print(f"Skipping row {index + 1}: No YouTube link for '{word}'")
            skipped_downloads += 1
            continue
        
        # Check if file already exists and path is recorded in CSV
        expected_filename = f"{word}.mp4"
        expected_file = os.path.join(output_folder, expected_filename)
        
        if os.path.exists(expected_file) and df.at[index, 'Local_File_Path']:
            print(f"Skipping '{word}' - file already exists and path recorded")
            skipped_downloads += 1
            continue
        elif os.path.exists(expected_file):
            # File exists but path not recorded in CSV
            df.at[index, 'Local_File_Path'] = expected_file
            print(f"Found existing file for '{word}', updating CSV")
            successful_downloads += 1
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {index + 1}/{len(df)}: {word}")
        print(f"URL: {youtube_link}")
        print(f"{'='*50}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(youtube_link, download=False)
                video_id = info.get('id', 'unknown')
                
                print(f"Downloading: {info.get('title', 'Unknown Title')}")
                print(f"Duration: {info.get('duration', 'Unknown')} seconds")
                
                # Actually download the video
                ydl.download([youtube_link])
                
                # Rename the file from video_id.mp4 to word.mp4
                temp_file = os.path.join(output_folder, f"{video_id}.mp4")
                if os.path.exists(temp_file):
                    os.rename(temp_file, expected_file)
                    print(f"✓ Successfully downloaded and renamed: {word}.mp4")
                    
                    # Update CSV with local file path
                    df.at[index, 'Local_File_Path'] = expected_file
                    successful_downloads += 1
                else:
                    # Check if file exists with different extension
                    found = False
                    for ext in ['.mp4', '.webm', '.mkv']:
                        temp_file_ext = os.path.join(output_folder, f"{video_id}{ext}")
                        if os.path.exists(temp_file_ext):
                            os.rename(temp_file_ext, expected_file)
                            print(f"✓ Successfully downloaded and renamed: {word}.mp4")
                            df.at[index, 'Local_File_Path'] = expected_file
                            successful_downloads += 1
                            found = True
                            break
                    
                    if not found:
                        print(f"⚠ Downloaded file not found: {temp_file}")
                        failed_downloads += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Error downloading {word}: {e}")
            failed_downloads += 1
            continue
    
    # Save the updated CSV
    try:
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        print(f"\nUpdated CSV file with local file paths: {csv_file_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        # Try saving with different encoding
        try:
            df.to_csv(csv_file_path, index=False, encoding='latin-1')
            print(f"Saved CSV with latin-1 encoding: {csv_file_path}")
        except:
            print("Could not save CSV file with any encoding")
            return None
    
    print(f"\n{'='*50}")
    print("Download Summary:")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Skipped downloads: {skipped_downloads}")
    print(f"Total processed: {len(df)}")
    print(f"{'='*50}")
    
    return df

def main():
    # Install required packages
    install_required_packages()
    
    # Configuration - using your project structure
    csv_file_path = "word_links.csv"  # Your CSV file
    output_folder = "youtube_videos"  # Folder where videos will be saved
    
    print("YouTube Video Downloader for FYP")
    print("=" * 50)
    print(f"Input CSV: {csv_file_path}")
    print(f"Output folder: {output_folder}")
    print("=" * 50)
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found.")
        print("Please make sure the file exists in the current directory.")
        return
    
    # Get file size
    file_size = os.path.getsize(csv_file_path)
    print(f"CSV file size: {file_size} bytes")
    
    # Download videos and update CSV
    result_df = download_youtube_videos(csv_file_path, output_folder)
    
    if result_df is not None:
        print("\nProcess completed successfully!")
        print(f"Videos saved to: {output_folder}")
        print(f"CSV file updated with local paths: {csv_file_path}")
        
        # Show a sample of the updated CSV
        print("\nSample of updated CSV:")
        print(result_df[['Word', 'YouTube_Link', 'Local_File_Path']].head())
    else:
        print("\nProcess completed with errors.")

if __name__ == "__main__":
    main()