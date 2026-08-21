# Security Policy

Do not commit API keys, tokens, private endpoints, datasets, checkpoints, or environment files. Use `.env.example` as the configuration template and provide credentials only through the runtime environment.

To report a vulnerability, open a private GitHub security advisory in this repository. Do not place live secrets in a public issue.

The example launchers intentionally avoid shell tracing around credentials. Rotate any credential that may have been exposed in an upstream or local history before using it with MIRAI.

