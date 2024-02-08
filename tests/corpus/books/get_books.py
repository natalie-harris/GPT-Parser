import os
import requests
from bs4 import BeautifulSoup

# Base URL of the site
base_url = 'https://www.gutenberg.org'

# Directory to save the books
save_dir = '/Users/natalieharris/UTK/NIMBioS/GPTPipeline/tests/corpus/books'

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Session for persistent connections
session = requests.Session()

# URL of the page to start with
start_url = base_url + '/browse/scores/top'

response = session.get(start_url)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')
print(soup)
ol = soup.select_one('h2[id="books-last1"] + ol')

if ol:
    for li in ol.find_all('li'):
        for a in li.find_all('a', href=lambda href: href and href.startswith('/ebooks/')):
            ebook_url = base_url + a['href']
            ebook_response = session.get(ebook_url)
            ebook_soup = BeautifulSoup(ebook_response.text, 'html.parser')

            title = ebook_soup.find('h1', itemprop=lambda itemprop: itemprop and itemprop.startswith('name'))

            download_link = ebook_soup.find('a', href=lambda href: href and href.endswith('.txt.utf-8'))
            
            if download_link:
                download_url = base_url + download_link['href']
                # Extract the ebook number from the URL for use in filename
                ebook_number = download_link['href'].split('/')[-1].replace('.txt.utf-8', '')
                file_path = os.path.join(save_dir, f'ebook_{ebook_number}.txt')
                
                # Download and save the ebook
                with open(file_path, 'wb') as file:
                    file.write(session.get(download_url).content)
                print(f"Saved: {file_path}")
else:
    print("Ordered list after '#authors-last30' not found")
