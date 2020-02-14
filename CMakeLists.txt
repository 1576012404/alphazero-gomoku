cmake_minimum_required(VERSION 3.15)
project(selfplay)

set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)

add_executable(selfplay main.cpp gomoku.h gomoku.cpp mcts.h mcts.cpp thread_pool.h libtorch.h libtorch.cpp learner.h learner.cpp tools.h)

target_link_libraries(selfplay ${TORCH_LIBRARIES})