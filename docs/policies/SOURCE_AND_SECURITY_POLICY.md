# Source, result and secret policy

## Allowed in source

- Python source and tests;
- lightweight versioned configuration;
- specifications, policies and documentation;
- reviewed patches against pinned upstream commits;
- manifest-generation code that contains no runtime data or credentials.

## Prohibited in source and Git

- raw or processed market/fundamental data;
- predictions, portfolios, reports generated from research runs and plots;
- logs, checkpoints, weights, caches, binaries and environment directories;
- credentials, tokens, cookies, proxy authentication or secret-bearing paths;
- absolute Mac user-data paths in runtime configuration.

The server token must remain at its protected server location with mode 0600.
Only the approved proxy client may construct the future Tushare client. The
proxy uses plain HTTP, and every data-provenance report must disclose that
transport limitation without exposing the token.

## Runtime placement

All research materialization occurs only under the approved server root.
Generated directories are untracked and must not be synchronized back as source.
Small, reviewed result documents may be copied to Mac only during an explicit
delivery step and remain excluded from Git.
