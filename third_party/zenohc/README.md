# zenoh-c

This directory contains the arm64 Debian packages needed by RK Studio on RK3588 boards:

- `debian/arm64/libzenohc_1.9.0_arm64.deb`
- `debian/arm64/libzenohc-dev_1.9.0_arm64.deb`

Install them on the board from the repository root:

```bash
./scripts/install_board_deps.sh
```

After installation, CMake should find Zenoh through:

```text
/usr/lib/cmake/zenohc/zenohcConfig.cmake
```
