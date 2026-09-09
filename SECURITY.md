# Security policy

## Supported version

Security fixes are applied to the latest version on the `main` branch. This project is an
experimental local application and does not currently maintain older release branches.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** option in the repository Security tab to submit a private
security advisory. Include:

- affected commit or version;
- reproduction steps;
- expected impact; and
- a suggested mitigation, if available.

Do not include real API keys, private financial PDFs, personal information, or other secrets in the
report. Replace sensitive values with minimal synthetic examples.

If private vulnerability reporting is unavailable, contact the maintainer through
[the GitHub profile](https://github.com/fotapol) without posting exploit details publicly.

## Deployment warning

Document RAG is intended for localhost, single-user use. It has no authentication or per-user state
isolation, and the current PDF endpoint is accessible to every client connected to the process.
PDFs are also sent to LlamaParse for parsing.

Do not expose the application directly to an untrusted network. A production deployment requires
authentication, authorization, isolated storage and indexes, upload hardening, transport security,
rate limiting, and an explicit retention policy.

## Credential response

If a credential is committed or shared accidentally:

1. revoke or rotate it immediately with the provider;
2. remove it from the current tree;
3. inspect the complete Git history and public caches;
4. notify affected users if appropriate; and
5. add a regression rule or test where practical.

Deleting a secret in a later commit does not make an earlier exposed credential safe.
