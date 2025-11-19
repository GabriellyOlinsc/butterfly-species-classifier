cmake_minimum_required(VERSION 3.10)
project(ButterflyPreprocessing)

# C++17 para std::filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontra OpenCV
find_package(OpenCV REQUIRED)

# Inclui headers do OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Cria executável
add_executable(preprocess preprocessing.cpp)

# Linka com OpenCV
target_link_libraries(preprocess ${OpenCV_LIBS})

# Linux: linka com filesystem library
if(UNIX AND NOT APPLE)
    target_link_libraries(preprocess stdc++fs)
endif()