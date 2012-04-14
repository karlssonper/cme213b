# QT variables
SET(app_SOURCES main MainWindow RenderWindow FluidSolver.cu)
SET(app_HEADERS MainWindow.h RenderWindow.h)

QT4_WRAP_CPP(app_HEADERS_MOC ${app_HEADERS})

# Build
CUDA_ADD_EXECUTABLE(fluidsolver ${app_SOURCES} ${app_HEADERS_MOC})
TARGET_LINK_LIBRARIES(fluidsolver ${QT_LIBRARIES} ${OPENGL_LIBRARIES} GLEW)

INSTALL (TARGETS fluidsolver DESTINATION bin)