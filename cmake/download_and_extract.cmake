# CMake post-install script for Obrew Studio
# Note: For macOS DMG installers, this script is not automatically executed.
# Certificate installation is handled on first app launch instead.

message(STATUS "Obrew Studio post-install")

# For macOS: The install_certificates_macos.sh script will be executed
# on first app launch by the Python application (main.py)

# For Linux: Similar approach - defer to first launch
# (Future implementation when Linux builds are supported)

message(STATUS "SSL certificate installation will occur on first app launch")
