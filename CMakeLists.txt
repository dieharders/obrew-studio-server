cmake_minimum_required(VERSION 3.15)

project(Obrew-Studio)

# Copy pre-existing files/folders to the temporary build directory
install(DIRECTORY "${CMAKE_SOURCE_DIR}/dist" DESTINATION build_temp)

# Set package information
set(CPACK_PACKAGE_NAME "Obrew Studio")
set(CPACK_PACKAGE_EXECUTABLES "Obrew-Server")
set(CPACK_PACKAGE_VERSION "0.9.0")
set(CPACK_PACKAGE_VENDOR "OpenBrewAi")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The tool for building personal Ai agents")
set(CPACK_PACKAGE_FILE_NAME "ObrewServer.WIN.Setup")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_POST_INSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/download_and_extract.cmake")
# Specify the output directory for the package
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_SOURCE_DIR}/output")

# Platform-specific settings
if(WIN32)
  set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
  set(CPACK_GENERATOR "DragNDrop")
elseif(UNIX)
  set(CPACK_GENERATOR "DEB;RPM")
endif()

# Include CPack module
include(CPack)
