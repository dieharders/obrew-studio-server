# CMake post-install script for Obrew Studio
# This script runs after package installation to set up necessary permissions
# and verify required files are present.

message(STATUS "Obrew Studio post-install: Setting up SSL certificate support")

# Get installation directory
if(DEFINED ENV{DESTDIR})
    set(INSTALL_DIR "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
else()
    set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
endif()

message(STATUS "Installation directory: ${INSTALL_DIR}")

# Define paths
set(BUNDLED_DIR "${INSTALL_DIR}/_deps/bundled")
set(CERT_SCRIPT_DIR "${INSTALL_DIR}/_deps/certs")

# Verify bundled directory exists
if(NOT EXISTS "${BUNDLED_DIR}")
    message(WARNING "Bundled directory not found: ${BUNDLED_DIR}")
    message(WARNING "mkcert binaries may not be available")
endif()

# For Unix-like systems (macOS, Linux), make binaries and scripts executable
if(UNIX)
    message(STATUS "Setting executable permissions for mkcert binaries and scripts")

    # Make mkcert binaries executable
    file(GLOB MKCERT_BINARIES "${BUNDLED_DIR}/mkcert-*")
    foreach(BINARY ${MKCERT_BINARIES})
        if(EXISTS "${BINARY}")
            execute_process(
                COMMAND chmod +x "${BINARY}"
                RESULT_VARIABLE CHMOD_RESULT
            )
            if(CHMOD_RESULT EQUAL 0)
                message(STATUS "Made executable: ${BINARY}")
            else()
                message(WARNING "Failed to make executable: ${BINARY}")
            endif()
        endif()
    endforeach()

    # Make certificate installation script executable (macOS)
    set(MACOS_CERT_SCRIPT "${CERT_SCRIPT_DIR}/install_certificates_macos.sh")
    if(EXISTS "${MACOS_CERT_SCRIPT}")
        execute_process(
            COMMAND chmod +x "${MACOS_CERT_SCRIPT}"
            RESULT_VARIABLE CHMOD_RESULT
        )
        if(CHMOD_RESULT EQUAL 0)
            message(STATUS "Made executable: ${MACOS_CERT_SCRIPT}")
        else()
            message(WARNING "Failed to make executable: ${MACOS_CERT_SCRIPT}")
        endif()
    endif()
endif()

# Verify fallback certificates exist
set(CERT_DIR "${INSTALL_DIR}/_deps/backends/ui/public")
set(CERT_FILE "${CERT_DIR}/cert.pem")
set(KEY_FILE "${CERT_DIR}/key.pem")

if(EXISTS "${CERT_FILE}" AND EXISTS "${KEY_FILE}")
    message(STATUS "SSL fallback certificates found")
else()
    message(WARNING "SSL fallback certificates not found in ${CERT_DIR}")
    message(WARNING "Application may not function correctly without certificates")
endif()

message(STATUS "Note: For macOS/Linux, trusted certificate installation will occur on first app launch")
message(STATUS "Note: For Windows, trusted certificates are installed by the Inno Setup installer")
message(STATUS "Obrew Studio post-install completed successfully")
