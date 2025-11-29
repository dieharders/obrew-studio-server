cmake_minimum_required(VERSION 3.31.4)

project(Obrew-Studio)

# Copy pre-existing executable/app bundle to installation directory
if(APPLE)
  # On macOS, PyInstaller creates an app bundle
  # Install the app bundle to the root of the install directory
  # CPack productbuild will handle the /Applications installation
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/dist/Obrew-Studio.app" DESTINATION . USE_SOURCE_PERMISSIONS)
elseif(WIN32)
  # On Windows, PyInstaller creates a directory with the executable and dependencies
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/dist/Obrew-Studio/" DESTINATION .)
else()
  # On Linux, PyInstaller creates a directory with the executable and dependencies
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/dist/Obrew-Studio/" DESTINATION .)
endif()

# Set package information
set(CPACK_PACKAGE_NAME "Obrew-Studio")
set(CPACK_PACKAGE_VERSION "0.12.0")
set(CPACK_PACKAGE_VENDOR "OpenBrewAi")
set(CPACK_PACKAGE_DESCRIPTION "Obrew Studio - Ai Toolkit")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The tool for building personal Ai agents")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Obrew-Studio")
# Specify the output directory for the package
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_SOURCE_DIR}/output")
# Define executable and icon
set(CPACK_PACKAGE_EXECUTABLES "Obrew-Studio" "${CMAKE_SOURCE_DIR}/backends/ui/favicon.ico")

# Platform-specific settings
if(WIN32)
  set(CPACK_GENERATOR "NSIS")
  set(CPACK_PACKAGE_FILE_NAME "Obrew-Studio.WIN.Setup")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
  # Create shortcut to .exe on the Desktop
  install(CODE "
      file(WRITE \"\${CMAKE_INSTALL_PREFIX}/Desktop/Obrew-Studio.lnk\"
      \"[InternetShortcut]\nURL=file:///\${CMAKE_INSTALL_PREFIX}/Obrew-Studio.exe\n\")")
  # Creating another shortcut for the headless version
  install(CODE "
      file(WRITE \"\${CMAKE_INSTALL_PREFIX}/Desktop/Obrew-Studio-headless.lnk\"
      \"[InternetShortcut]\nURL=file:///\${CMAKE_INSTALL_PREFIX}/Obrew-Studio.exe --headless=True\n\")")
elseif(APPLE)
  set(CPACK_GENERATOR "productbuild")
  set(CPACK_PACKAGE_FILE_NAME "Obrew-Studio.macOS.Setup")
  # Set package identifier for productbuild
  set(CPACK_PRODUCTBUILD_IDENTIFIER "com.OpenBrewAi.Obrew-Studio")
  # Important: Don't use CPACK_PACKAGING_INSTALL_PREFIX as it causes double nesting
  # The install() command with DESTINATION . will handle the correct placement
  # Configure post-install script for certificate installation if available
  if(EXISTS "${CMAKE_SOURCE_DIR}/certs")
    set(CPACK_PRODUCTBUILD_SCRIPTS_DIR "${CMAKE_SOURCE_DIR}/certs")
  endif()
  # Don't include LICENSE/README for productbuild - it has strict file extension requirements
elseif(UNIX)
  set(CPACK_GENERATOR "DEB;RPM")
  set(CPACK_PACKAGE_FILE_NAME "Obrew-Studio.Linux.Setup")
endif()

# Include CPack module
include(InstallRequiredSystemLibraries)
include(CPack)
