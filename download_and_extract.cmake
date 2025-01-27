# Download zip archive for cudart dll's and llama.cpp binaries

include(ExternalProject)

# Download the zip file
set(ZIP_DESTINATION "${CMAKE_INSTALL_PREFIX}/_deps/servers/llama.cpp")

set(LLAMA_ZIP_URL "https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-b4562-bin-win-cuda-cu12.4-x64.zip")
set(LLAMA_ZIP_FILE "${ZIP_DESTINATION}/llama.cpp.zip")

set(CUDA_ZIP_URL "https://github.com/ggerganov/llama.cpp/releases/latest/download/cudart-llama-bin-win-cu12.4-x64.zip")
set(CUDA_ZIP_FILE "${ZIP_DESTINATION}/cudart_dlls.zip")

file(MAKE_DIRECTORY ${ZIP_DESTINATION})

# Download file A
file(DOWNLOAD ${LLAMA_ZIP_URL} ${LLAMA_ZIP_FILE} SHOW_PROGRESS)
if(NOT EXISTS ${LLAMA_ZIP_FILE})
  message(FATAL_ERROR "Failed to download ${LLAMA_ZIP_URL}")
endif()

# Download file B
file(DOWNLOAD ${CUDA_ZIP_URL} ${CUDA_ZIP_FILE} SHOW_PROGRESS)
if(NOT EXISTS ${CUDA_ZIP_FILE})
  message(FATAL_ERROR "Failed to download ${CUDA_ZIP_URL}")
endif()

# Unzip the downloaded file (using CMake's built-in commands)
execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xvf ${LLAMA_ZIP_FILE}
    COMMAND ${CMAKE_COMMAND} -E tar xvf ${CUDA_ZIP_FILE}
    WORKING_DIRECTORY ${ZIP_DESTINATION}
)

# Optional: Remove the zip file after extraction
file(REMOVE ${LLAMA_ZIP_FILE})
file(REMOVE ${CUDA_ZIP_FILE})
