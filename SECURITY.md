# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainers or use [GitHub's private vulnerability reporting](https://github.com/JihoonJeong/Neural-MRI/security/advisories/new)
3. Include a description of the vulnerability, steps to reproduce, and potential impact

We will acknowledge your report within 48 hours and aim to provide a fix within 7 days for critical issues.

## Scope

This policy covers:

- The Neural MRI Scanner backend (FastAPI server)
- The frontend application
- Docker deployment configurations
- CI/CD pipeline configurations

## Out of Scope

- Vulnerabilities in upstream dependencies (report these to the respective projects)
- The HuggingFace Spaces demo deployment (best-effort only)
- Model weights and their outputs (these are third-party artifacts)

## Security Considerations

Neural MRI Scanner runs model inference locally. Keep in mind:

- **HuggingFace tokens**: Store tokens in `.env` files (gitignored) or environment variables. Never commit tokens.
- **Model loading**: Only load models from trusted sources. The HF Hub search feature loads arbitrary model weights.
- **Network exposure**: The default configuration binds to localhost. If exposing to a network, use a reverse proxy with authentication.
