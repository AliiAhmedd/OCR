# Open questions — fetching images from Azure Blob / Amazon S3 / Google Cloud Storage

Questions for the supervisors, with what we found in `OCR_data_full` so far and what each answer unblocks.
Tick a question off (and write the answer under it) once it is answered.

## What we know (2026-10-07)

- Core is a Django app. For a transaction, the image references are in
  `services_transactiondatacontainer.meta_data` (jsonb), joined on `transaction_id = services_transaction.id`.
- Fields holding references: `front_url`, `back_url`, `front_img`, `back_img`, `url`, `img`, `url_hrz`,
  `images_path`, `dict_of_paths.{front_img, back_img, img, document_page_N}`, `image_paths[]`.
- Every reference is a **path to a `.zip`** (same format as the NFS share), never a cloud URL. Three shapes:
  `media/…/x.zip` (Django media-relative), `/…/…/x.zip` (absolute), `./…/x.zip` (relative).
- The per-service tables with `front_url`/`back_url` columns (`services_egyptiannationalid`, `services_passport`, …)
  are empty: 0 rows ever inserted.
- `services_transactiondatacontainer.data` is **Fernet-encrypted** in every row (all values start with `gAAAAA`).
- `services_bundle`: all 4 bundles have `save_transaction_images` and `save_transaction_data` on; 1 has a
  `transaction_data_apis_public_key` (PEM public key).
- Our copy is small (≈3 121 transactions, 2 with error 4201) and contains the placeholder host
  `storage.example.com`, so it may be a scrubbed sample.

## Questions

### Database and links
- [ ] **Q1.** Is our `OCR_data_full` a scrubbed / sample copy of production? Do production references look the same?
- [ ] **Q2.** Are `services_transactiondatacontainer.meta_data` fields the right place to read a 4201 transaction's
      images? Which fields are authoritative per service (e.g. `front_url` vs `front_img` vs `dict_of_paths`)?

### Where the files live
- [ ] **Q3.** Does Core save uploads through `django-storages`? Per deployment, which backend (Azure / S3 / GCS / NFS)
      and which bucket or container? (`STORAGES` / `DEFAULT_FILE_STORAGE`, `AWS_STORAGE_BUCKET_NAME`,
      `AZURE_CONTAINER`, `GS_BUCKET_NAME`, any `location` prefix.)
- [ ] **Q4.** What folder do the absolute `/…/x.zip` and relative `./…/x.zip` paths start from? Is it the NFS mount,
      and how does it map to the bucket root (which prefix to strip)?
      *Used by:* `strip_prefixes` of `storage_key()` in `id_classifier/references.py` (e.g. `["/mnt/nfs", "media"]`).
- [ ] **Q5.** Inside the cloud copies, is each `.zip` the same as on NFS (`original.jpeg` inside), or base64 text of a
      resized `.jpg` (as we were told)?

### Credentials and encryption
- [ ] **Q6.** "Public and private keys" for the clouds: is this the access key id + secret (S3), account name + key or
      SAS (Azure), HMAC or service-account JSON (GCS)? How will the pipeline receive them (env vars, Airflow connection)?
- [ ] **Q7.** Are the image objects encrypted (cloud customer-provided key, or app-level encryption with the bundle's
      `transaction_data_apis_public_key`)? If so, which key decrypts them and can the pipeline use it?
- [ ] **Q8.** `data` is Fernet-encrypted. Do we need anything from it? If yes, where is the Fernet key and may the
      pipeline use it?

### Scope
- [x] **Q9.** Is fetching read-only everywhere (no writes, no deletes)?
      **Answer:** yes, read-only.
- [ ] **Q10.** Cloud images are said to be resized. Is a short accuracy check on resized vs original images wanted
      before production (the kNN reference set is built from full-size NFS images)?
