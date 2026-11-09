# Approved router API keys: one item per key, keyed by the key's SHA-256 (the key itself is never stored), with the
# owner's email and name and an enabled flag. Managed with router/api_keys.py; the router only reads it (GetItem).
resource "aws_dynamodb_table" "api_keys" {
  name         = "${var.name}-api-keys"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "key_hash"

  attribute {
    name = "key_hash"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  deletion_protection_enabled = true
}
