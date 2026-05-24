#!/usr/bin/env python
"""Bootstrap the first admin user.

Usage:
    python create_admin.py
    python create_admin.py --username admin --email admin@school.edu --password secret
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.auth import hash_password
from backend.database import SessionLocal, init_db
from backend.models import User


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the first admin user")
    parser.add_argument("--username", default=None)
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    args = parser.parse_args()

    init_db()

    username = args.username or input("Username: ").strip()
    email    = args.email    or input("Email:    ").strip()
    password = args.password or getpass.getpass("Password: ")

    if not all([username, email, password]):
        print("[ERROR] All fields are required.")
        sys.exit(1)

    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            print(f"[ERROR] User '{username}' already exists.")
            sys.exit(1)

        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            role="admin",
        )
        db.add(user)
        db.commit()
        print(f"\n  Admin user '{username}' created successfully.")
        print(f"  Role: admin | Email: {email}")
        print("\n  Start the backend:  python run_backend.py")
    finally:
        db.close()


if __name__ == "__main__":
    main()
