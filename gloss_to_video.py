from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# Set up Chrome options
options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--start-maximized')  # Start maximized to look more normal
# options.add_argument("--headless=new")  # KEEP THIS COMMENTED OUT FOR NOW. Run with visible browser first.

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

# List to hold our data
data = []

try:
    # 1. Navigate to the main dictionary page
    print("Loading main dictionary page...")
    driver.get("https://indiansignlanguage.org/search-dictionary/")
    
    # 2. NEW: WAIT FOR CLOUDFLARE TO FINISH
    # Wait for the title to NOT be the Cloudflare challenge title
    WebDriverWait(driver, 25).until_not(
        EC.title_contains("Checking") # Waits until the title no longer contains "Checking"
    )
    # Then, wait for the actual page content to load. Let's use the page's H1 title.
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "h1"))
    )
    print("Main page loaded successfully.")
    time.sleep(3)  # Extra safety sleep

    # 3. Find all the word links on the page
    print("Finding all word links...")
    # Let's use a more specific selector to avoid any other links
    word_elements = driver.find_elements(By.CSS_SELECTOR, "div.letter-section ul.az-columns li a")
    
    word_links = {}
    for elem in word_elements:
        text = elem.text.strip()
        href = elem.get_attribute('href')
        if text and href:  # Only add if both text and link exist
            word_links[text] = href
            
    print(f"Found {len(word_links)} words to process.")
    
    # 4. Loop through each word and its URL
    # TEST WITH JUST 3 WORDS FIRST
    test_words = dict(list(word_links.items())[:3])
    
    for word, url in test_words.items():
        print(f"Processing: {word}")
        
        try:
            # Navigate to the word's page
            driver.get(url)
            
            # WAIT FOR CLOUDFLARE ON THE INDIVIDUAL PAGE TOO
            WebDriverWait(driver, 25).until_not(
                EC.title_contains("Checking")
            )
            # Wait for some content on the word's page to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
            time.sleep(4)  # Crucial delay to let everything load fully.
            
            # 5. Try to find the YouTube iframe or link.
            yt_url = "NOT_FOUND"
            # Look for an iframe that has 'youtube.com' or 'youtu.be' in its src
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                src = iframe.get_attribute('src')
                if src and ('youtube.com' in src or 'youtu.be' in src):
                    yt_url = src
                    print(f"  Found YouTube URL: {yt_url}")
                    break # Stop at the first YouTube iframe found
            
            # If no iframe was found, maybe it's a direct link?
            if yt_url == "NOT_FOUND":
                links = driver.find_elements(By.TAG_NAME, "a")
                for link in links:
                    href = link.get_attribute('href')
                    if href and ('youtube.com' in href or 'youtu.be' in href):
                        yt_url = href
                        print(f"  Found YouTube URL (direct link): {yt_url}")
                        break
            
            # Append the data
            data.append({
                "word": word,
                "yt_url": yt_url
            })
            
        except Exception as e:
            print(f"  ERROR processing {word}: {str(e)}")
            data.append({"word": word, "yt_url": f"ERROR: {str(e)}"})
        
        # Be very polite to avoid triggering rate limits or bans
        print("  Waiting 5 seconds before next word...")
        time.sleep(5) 

finally:
    # Ensure the browser closes even if the script crashes
    driver.quit()
    print("Browser closed.")

# 5. Save the collected data to a CSV file
if data:
    df = pd.DataFrame(data)
    csv_filename = 'indian_sign_language_dataset.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Successfully saved {len(data)} entries to '{csv_filename}'")
else:
    print("No data was collected.")