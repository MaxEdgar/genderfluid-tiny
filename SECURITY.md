# Security Policy

## Reporting vulnerabilities

If you discover a security vulnerability, report it privately through
GitHub's private vulnerability reporting
(https://github.com/MaxEdgar/genderfluid-tiny/security/advisories/new).
Do not open a public issue for unfixed vulnerabilities, and do not
disclose them publicly until a fix is available.

## Scope

The inference library runs entirely locally. It does not:

- Store or transmit user data
- Make network requests during inference
- Handle authentication
- Process sensitive information beyond names you provide to it

The repository also contains an optional data-fetching script
(`fetch_multinational_data.py`) used only when retraining the model;
it makes outbound requests to official government dataset sources at
training time.

## Model safety

The model is trained on public data and does not contain personally
identifiable information. However, the model is not designed for
high-stakes decisions. Do not use it for identity verification,
fraud detection, or similar purposes.
