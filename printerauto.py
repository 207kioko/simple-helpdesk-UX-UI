import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

# Replace with your printer's actual IP
printer_ip = '192.168.2.40'
printer_url = f'https://{printer_ip}/'  # root page


def fetch_printer_status():
    try:
        response = requests.get(printer_url, timeout=5, verify=False)
        response.raise_for_status()  # Raise error if page fails to load
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Could not connect to printer: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    print("=== Page content snippet ===")
    print(response.text[:2000])  # prints first 2000 characters of the HTML
    print("===========================")


    # Find toner level
    toner_level = None
    paper_status = None

    for tag in soup.find_all('script'):
        if 'toner' in tag.text.lower() or 'paper' in tag.text.lower():
            print("📄 Found embedded script data that might contain toner/paper levels.")
            print(tag.text)  # Optional: Inspect to extract exact data

    # Example: parsing raw HTML (adjust as needed based on actual layout)
    for div in soup.find_all('div'):
        text = div.get_text(strip=True)
        if 'Toner' in text:
            print("🔋", text)
        if 'Paper' in text:
            print("📄", text)

if __name__ == '__main__':
    fetch_printer_status()
