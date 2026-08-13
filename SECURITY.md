# Security Policy

## Credentials and local licenses

Never commit credentials, access tokens, private keys, environment files, or
solver license files to this repository. In particular, a Gurobi license may
contain a WLS access identifier and secret. Keep it outside the repository and
let Gurobi discover it through the user's local configuration or an externally
configured license path.

If a credential is committed, removing it in a later commit is not sufficient:
revoke or rotate it immediately, then remove it from the repository history
before publishing the repository.

## Reporting a vulnerability

Please use the repository host's private vulnerability-reporting feature when
it is available. Include the affected file or component, impact, and a minimal
reproduction, but do not include live credentials or private data.

If private reporting is unavailable, open a minimal public issue asking the
maintainer for a private reporting channel. Do not disclose exploit details,
credentials, or confidential data in that issue.

