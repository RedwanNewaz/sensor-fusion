cmake_minimum_required(VERSION 3.5)
project(sensor_fusion)

set(CMAKE_CXX_STANDARD 17)
set(CXX_EXTENSIONS OFF)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")

include_directories(third_parties/eigen-3.3.7/)
include_directories(third_parties/nlohmann-3.6.1)
include_directories(third_parties/catch2-2.7.0)
include_directories(include)

file(GLOB sensors_SRC src/sensors/*)
file(GLOB process_models_SRC src/process_models/*)
file(GLOB measurement_models_SRC src/measurement_models/*)
file(GLOB filters_SRC src/filters/*)
file(GLOB primitives_SRC src/primitives/*)
file(GLOB fusion_SRC src/fusion/*)
file(GLOB measurements_SRC src/measurements/*)
file(GLOB utils_SRC src/utils/*)
file(GLOB state_vector_views_SRC src/state_vector_views/*)
file(GLOB measurement_vector_views_SRC src/measurement_vector_views/*)
file(GLOB beliefs_SRC src/beliefs/*)

set(SRC_EXCEPT_MAIN
        ${sensors_SRC}
        ${states_SRC}
        ${process_models_SRC}
        ${measurement_models_SRC}
        ${filters_SRC}
        ${fusion_SRC}
        ${utils_SRC}
        ${beliefs_SRC}
        ${state_vector_views_SRC}
        ${measurement_vector_views_SRC}
)

#add_library(${PROJECT_NAME} ${SRC_EXCEPT_MAIN})
#target_link_libraries(${PROJECT_NAME} Eigen3)

### add ros support
find_package(ament_cmake REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)


add_library(${PROJECT_NAME}  SHARED
        src/mains/main_sensor_fusion.cpp
        ${SRC_EXCEPT_MAIN}
        )
target_include_directories(${PROJECT_NAME}  PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_definitions(${PROJECT_NAME}
        PRIVATE "${PROJECT_NAME}_CPP_BUILDING_DLL")
ament_target_dependencies(${PROJECT_NAME}
        "rclcpp"
        "geometry_msgs"
        "nav_msgs"
        "visualization_msgs"
        "tf2_ros"
        "rclcpp_action"
        "rclcpp_components")
rclcpp_components_register_node(${PROJECT_NAME} PLUGIN "airlab::apriltag_fusion" EXECUTABLE ${PROJECT_NAME}_node)

install(TARGETS
        ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        )

install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
        DESTINATION share/${PROJECT_NAME})


ament_export_dependencies(rosidl_default_runtime)
ament_package()

# add_executable(sensor_fusion_all
#         ${SRC_EXCEPT_MAIN}
#         src/mains/all_main.cpp
# )

add_library(Catch INTERFACE)
add_executable( sensor_fusion_test
        test/sensors_test.cpp ${sensors_SRC}
        test/process_models_test.cpp ${process_models_SRC}
        test/measurement_models_test.cpp ${measurement_models_SRC}
        test/filters_test.cpp ${filters_SRC}
        test/fusion_test.cpp ${fusion_SRC}
        test/beliefs_test.cpp ${beliefs_SRC}
        test/state_vector_views_test.cpp ${state_vector_views_SRC}
        test/measurement_vector_views_test.cpp ${measurement_vector_views_SRC}
        test/utils_test.cpp ${utils_SRC}
        test/test.cpp)
target_link_libraries(sensor_fusion_test Catch)

enable_testing()
add_test( NAME sensor_fusion_test COMMAND sensor_fusion_test )
