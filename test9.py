import os
import smtplib
import socket
from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # <-- IMPORTANT: loads .env into environment variables
def must_get(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def send_email(subject: str, body: str) -> None:
    email_to = must_get("EMAIL_TO")
    email_from = must_get("EMAIL_FROM")

    host = must_get("SMTP_HOST")
    port = int((os.getenv("SMTP_PORT") or "587").strip())
    user = must_get("SMTP_USER")
    password = must_get("SMTP_PASS")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(msg)

def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    container = os.getenv("HOSTNAME", "unknown-container")
    host = socket.gethostname()

    subject = f"[DONE] Pipeline finished successfully ({now})"
    body = (
        f"All scripts completed successfully.\n\n"
        f"Time: {now}\n"
        f"Container: {container}\n"
        f"SLOT_CHOICE: {os.getenv('SLOT_CHOICE','')}\n"
    )

    send_email(subject, body)
    print("[MAIL] Sent success email to", os.getenv("EMAIL_TO"))

if __name__ == "__main__":
    main()
