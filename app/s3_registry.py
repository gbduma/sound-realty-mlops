# app/s3_registry.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import os, boto3

def _parse_s3_uri(uri: str):
    assert uri.startswith("s3://"), f"Bad S3 URI: {uri}"
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key

def sync_model_from_s3(prefix: str, model_name: str, dest_root: str = "model") -> Path:
    """
    Download model artifacts from s3://bucket/prefix/<model_name>/ into dest_root/<model_name>/
    """
    bucket, key_prefix = _parse_s3_uri(prefix)
    src_prefix = f"{key_prefix.rstrip('/')}/{model_name}/"
    dest_dir = Path(dest_root) / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            found = True
            rel = obj["Key"][len(src_prefix):]
            if not rel:
                continue
            local = dest_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(local))
    if not found:
        raise FileNotFoundError(f"No objects under s3://{bucket}/{src_prefix}")
    return dest_dir
