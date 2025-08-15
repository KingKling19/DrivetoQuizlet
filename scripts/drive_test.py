from __future__ import print_function
import os.path
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scope: read-only access to Drive
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def main():
    # Ensure config directory exists
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    token_path = config_dir / 'token.json'
    credentials_path = config_dir / 'credentials.json'
    
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                print(f"ERROR: Please download credentials.json from Google Cloud Console and place it at {credentials_path}")
                return
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    # List the first 10 files the account can see
    results = service.files().list(
        pageSize=10, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('First 10 files in your Drive:')
        for item in items:
            print(f"{item['name']} ({item['id']})")

if __name__ == '__main__':
    main()
