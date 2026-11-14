# Security and credential handling

A previous version of `Step6.24-UploadSubmission.py` in the public repository embedded a personal Synapse authentication token directly in source code. The current file has been changed to read `SYNAPSE_AUTH_TOKEN` from the environment.

## Immediate action required from the credential owner

1. **Revoke the exposed Synapse personal access token and generate a new one**, if still needed. Treat the published token as compromised regardless of whether its current validity is known.
2. Investigate Synapse account access/activity while the exposed token may have been valid.
3. Do not share the new credential in issue comments, source files, notebooks, screenshots or shell history.
4. Store the new value in a secure environment or SLURM secrets facility, and run `Step6.24-UploadSubmission.py` with `SYNAPSE_AUTH_TOKEN` available only in the process environment.
5. Consider coordinated repository-history remediation with the security team. **Editing the latest file does not remove the secret from older Git commits/forks/clones.** Revocation takes priority.

## Example (safe pattern; no literal token)

```bash
# Obtain your secret through a secure credential store, not a committed file.
# Ensure SYNAPSE_AUTH_TOKEN is exported securely in the executing environment.
python Step6.24-UploadSubmission.py
```

Contributors should use environment variables or the Synapse client-supported secure login mechanisms, never committed plaintext credentials. This is not a substitute for a repository-wide historical secret scan.
