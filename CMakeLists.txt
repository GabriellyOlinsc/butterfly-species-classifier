cmake_minimum_required(VERSION 3.10)
project(ButterflyPreprocessing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Otimizações de compilação
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# OpenMP para paralelização
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP encontrado - paralelização habilitada")
else()
    message(WARNING "OpenMP não encontrado - executará sequencialmente")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Preprocessamento
add_executable(preprocess preprocessing.cpp)
target_link_libraries(preprocess ${OpenCV_LIBS})

# Feature extraction (versão otimizada)
add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS})

# Linux: filesystem library
if(UNIX AND NOT APPLE)
    target_link_libraries(preprocess stdc++fs)
    target_link_libraries(feature_extraction stdc++fs)
endif()

# OpenMP linking
if(OPENMP_FOUND)
    target_link_libraries(feature_extraction OpenMP::OpenMP_CXX)
endif()