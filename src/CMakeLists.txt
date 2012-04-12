# QT variables
SET(app_SOURCES main mainwindow FluidSolver.cu)
SET(app_HEADERS mainwindow.h)

QT4_WRAP_CPP(app_HEADERS_MOC ${app_HEADERS})

# Build
CUDA_ADD_EXECUTABLE(main ${app_SOURCES} ${app_HEADERS_MOC})
TARGET_LINK_LIBRARIES(main ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ${QT_QTSQL_LIBRARIES})
