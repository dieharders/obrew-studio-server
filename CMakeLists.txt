cmake_minimum_required(VERSION 3.31.4)

project(Obrew-Studio)

# Copy pre-existing executable in source folder to root installation directory
# install(FILES "${CMAKE_SOURCE_DIR}/dist/Obrew-Studio/Obrew-Studio.exe" DESTINATION .)
# Copy pre-existing files in source folder to root installation directory
install(DIRECTORY "${CMAKE_SOURCE_DIR}/dist/Obrew-Studio/" DESTINATION .)
# install(DIRECTORY "${CMAKE_SOURCE_DIR}/dist" DESTINATION build_temp)

# Set package information
set(CPACK_PACKAGE_NAME "Obrew-Studio")
set(CPACK_PACKAGE_VERSION "0.9.0")
set(CPACK_PACKAGE_VENDOR "OpenBrewAi")
set(CPACK_PACKAGE_DESCRIPTION "Obrew Studio - Ai Toolkit")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The tool for building personal Ai agents")
set(CPACK_PACKAGE_FILE_NAME "Obrew-Studio.WIN.Setup")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_POST_INSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/certs/download_and_extract.cmake")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Obrew-Studio")
# Specify the output directory for the package
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_SOURCE_DIR}/output")
# Define executable and icon
set(CPACK_PACKAGE_EXECUTABLES "Obrew-Studio" "${CMAKE_SOURCE_DIR}/backends/ui/favicon.ico")

# Platform-specific settings
if(WIN32)
  set(CPACK_GENERATOR "NSIS")
  # Create shortcut to .exe on the Desktop
  install(CODE "
      file(WRITE \"\${CMAKE_INSTALL_PREFIX}/Desktop/Obrew-Studio.lnk\"
      \"[InternetShortcut]\nURL=file:///\${CMAKE_INSTALL_PREFIX}/Obrew-Studio.exe\n\")")
  # Creating another shortcut for the headless version
  install(CODE "
      file(WRITE \"\${CMAKE_INSTALL_PREFIX}/Desktop/Obrew-Studio-headless.lnk\"
      \"[InternetShortcut]\nURL=file:///\${CMAKE_INSTALL_PREFIX}/Obrew-Studio.exe --headless=True\n\")")
elseif(APPLE)
  set(CPACK_GENERATOR "DragNDrop")
elseif(UNIX)
  set(CPACK_GENERATOR "DEB;RPM")
endif()

# Include CPack module
include(InstallRequiredSystemLibraries)
include(CPack)
