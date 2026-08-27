# AWS and clinical-data security checklist

This document is guidance only. The repository does not provision or mutate AWS
infrastructure.

## Minimum S3 IAM policy shape

Replace placeholders and narrow the prefix. Attach it to a role or SSO-managed
permission set, not a long-lived user access key.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListOnlyProjectPrefix",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::<private-bucket>",
      "Condition": {"StringLike": {"s3:prefix": ["<prefix>", "<prefix>/*"]}}
    },
    {
      "Sid": "ReadWriteProjectObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::<private-bucket>/<prefix>/*"
    }
  ]
}
```

Add `s3:DeleteObject` only to identities that operate an approved retention or
cleanup workflow. If SSE-KMS is used, add narrowly scoped `kms:Encrypt`,
`kms:Decrypt`, and `kms:GenerateDataKey` permissions for one key.

## Private deployment baseline

- S3 Block Public Access on at account and bucket levels; no public-read ACLs.
- S3 encryption at rest, TLS transport policy, versioning, access logging, and
  lifecycle/retention rules reviewed with the data owner.
- RDS encrypted storage, automated backups, TLS, private networking, restricted
  security group, non-superuser application login, rotation, and monitoring.
- Credentials from SSO, an IAM role, or a managed secret—not Git, YAML, notebooks,
  shell history, screenshots, or model artifacts.
- CloudTrail/database logs must avoid identifiers and must themselves have
  controlled retention and access.
- Raw HUKM video must be de-identified where feasible and never be public. Keep
  the re-identification key outside this platform under institutional control.
- Separate development and clinical accounts/buckets/databases; use least
  privilege and periodic access review.
- Obtain institutional ethics, legal/data-use, and security approval before any
  HUKM transfer. Technical encryption alone is not authorization.
