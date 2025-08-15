from __future__ import print_function
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = "1LUeSOwi4o9tBJ4rtcC65j9sFCe6CuxGp"  # paste your folder ID

PRESENTATION_MIMES = [
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-powerpoint",                                             # .ppt
    "application/vnd.ms-powerpoint.presentation.macroEnabled.12",                # .pptm
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",    # .ppsx
    "application/vnd.google-apps.presentation",                                  # Google Slides
    "application/vnd.google-apps.shortcut",                                      # Shortcuts we will resolve
]

def creds():
    c = None
    if os.path.exists('token.json'):
        c = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not c or not c.valid:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        c = flow.run_local_server(port=0)
        with open('token.json', 'w') as t:
            t.write(c.to_json())
    return c

def main():
    c = creds()
    svc = build('drive', 'v3', credentials=c)

    # Build OR query across allowed mime types
    mime_or = " or ".join([f"mimeType = '{m}'" for m in PRESENTATION_MIMES])
    query = f"'{FOLDER_ID}' in parents and ({mime_or}) and trashed = false"

    items = []
    page = None
    while True:
        resp = svc.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, shortcutDetails)",
            pageToken=page,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            orderBy="modifiedTime desc"
        ).execute()
        items.extend(resp.get("files", []))
        page = resp.get("nextPageToken")
        if not page:
            break

    if not items:
        print("No presentations found. Double-check the folder ID.")
        return

    print(f"Found {len(items)} presentation(s):")
    for f in items:
        mt = f.get("mimeType")
        line = f"- {f['name']}  | mimeType={mt} | id={f['id']} | modified={f.get('modifiedTime','')}"
        # If it's a shortcut, show target details
        if mt == "application/vnd.google-apps.shortcut":
            sd = f.get("shortcutDetails", {})
            target_id = sd.get("targetId")
            target_mime = sd.get("targetMimeType")
            line += f"\n    ↳ shortcut → target_id={target_id} target_mime={target_mime}"
        print(line)

if __name__ == '__main__':
    main()
