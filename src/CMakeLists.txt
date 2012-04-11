# QT variables
SET(app_SOURCES FluidSolver.cu)
SET(app_HEADERS FluidSolver.h)

QT4_WRAP_CPP(app_HEADERS_MOC ${app_HEADERS})

# Build
CUDA_ADD_EXECUTABLE(fluidsolver ${app_SOURCES} ${app_HEADERS_MOC})
TARGET_LINK_LIBRARIES(fluidsolver ${QT_LIBRARIES} ${OPENGL_LIBRARIES})
