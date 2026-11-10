"""Manage the router's approved API keys (the API_KEYS_TABLE DynamoDB table).

  python api_keys.py add    --email ada@example.com --name "Ada Lovelace"   # prints the new key once
  python api_keys.py list
  python api_keys.py revoke --email ada@example.com                         # or --hash <key_hash prefix>
  python api_keys.py enable --hash 3f2a9c

The table holds only each key's SHA-256 (key_hash), never the key: a lost key cannot be recovered, only revoked and
replaced. Revocation reaches the router within its KEY_CACHE_S (60 s by default). Table: --table, else
$API_KEYS_TABLE, else behemoth-api-keys. Uses the caller's AWS credentials.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from datetime import datetime, timezone

from router import key_hash


def _items(db, table):
    kw = {"TableName": table}
    while True:
        page = db.scan(**kw)
        yield from page.get("Items", [])
        if "LastEvaluatedKey" not in page:
            return
        kw["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def _select(db, table, args):
    if not args.email and not args.hash:
        sys.exit("give --email or --hash")
    chosen = [i for i in _items(db, table)
              if (args.email and i["email"]["S"].lower() == args.email.lower())
              or (args.hash and i["key_hash"]["S"].startswith(args.hash))]
    if not chosen:
        sys.exit("no matching key")
    if args.hash and len(chosen) > 1:
        sys.exit("hash prefix matches several keys; give more of it")
    return chosen


def _set_enabled(db, table, args, enabled: bool):
    for i in _select(db, table, args):
        db.update_item(TableName=table, Key={"key_hash": i["key_hash"]},
                       UpdateExpression="SET enabled = :e, updated = :t",
                       ExpressionAttributeValues={":e": {"BOOL": enabled}, ":t": {"S": _now()}})
        print(f"{'enabled' if enabled else 'revoked'} {i['key_hash']['S'][:12]} {i['name']['S']} <{i['email']['S']}>")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=os.environ.get("API_KEYS_TABLE", "behemoth-api-keys"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="issue a key")
    add.add_argument("--email", required=True)
    add.add_argument("--name", required=True)
    sub.add_parser("list", help="list keys (hash prefix, owner, state)")
    for cmd in ("revoke", "enable"):
        p = sub.add_parser(cmd)
        p.add_argument("--email")
        p.add_argument("--hash", help="key_hash prefix (from list)")
    args = ap.parse_args()

    import boto3

    db = boto3.client("dynamodb", region_name=args.region)
    if args.cmd == "add":
        if "@" not in args.email:
            sys.exit("--email must be an email address")
        key = "bk-" + secrets.token_hex(32)
        db.put_item(TableName=args.table, ConditionExpression="attribute_not_exists(key_hash)", Item={
            "key_hash": {"S": key_hash(key)}, "email": {"S": args.email}, "name": {"S": args.name},
            "enabled": {"BOOL": True}, "created": {"S": _now()},
        })
        print(f"key for {args.name} <{args.email}> (shown once, store it now):\n{key}")
    elif args.cmd == "list":
        rows = sorted(_items(db, args.table), key=lambda i: (i["email"]["S"].lower(), i.get("created", {}).get("S", "")))
        for i in rows:
            state = "enabled" if i.get("enabled", {}).get("BOOL") else "revoked"
            print(f"{i['key_hash']['S'][:12]}  {state:7}  {i.get('created', {}).get('S', ''):25}  "
                  f"{i['name']['S']} <{i['email']['S']}>")
    else:
        _set_enabled(db, args.table, args, args.cmd == "enable")


if __name__ == "__main__":
    main()
