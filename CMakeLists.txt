cmake_minimum_required(VERSION 3.10)
project(ButterflyPreprocessing)

# C++17 é necessário para std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontra OpenCV
find_package(OpenCV REQUIRED)

# Inclui diretórios do OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Cria executável
add_executable(preprocess_butterflies preprocessing.cpp)

# Linka com OpenCV
target_link_libraries(preprocess_butterflies ${OpenCV_LIBS})

# Para sistemas Unix, pode ser necessário linkar com stdc++fs
if(UNIX AND NOT APPLE)
    target_link_libraries(preprocess_butterflies stdc++fs)
endif()