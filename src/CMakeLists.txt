# cmake_minimum_required(VERSION 3.5)
# set(CMAKE_CXX_STANDARD 17)

project(DNN_TEST)

option(USE_CUDA "Using CUDA to accelerate" OFF)
option(USE_NPU "Using AIPU/NPU backend for DNN" ON)

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

if(USE_NPU)
    add_definitions(-DUSE_NPU)
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Добавление исполняемого файла
add_executable(facedetection main.cpp)
target_link_libraries(facedetection ${OpenCV_LIBS} ${DNN_LIBS})
if(USE_CUDA)
    target_link_libraries(facedetection ${CUDA_LIBRARIES})
endif(USE_CUDA)

# Установка путей для библиотек и протоколов
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
