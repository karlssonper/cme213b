#include <GL/glew.h>
#include "RenderWindow.h"
#include "FluidSolver.h"
#include <stdio.h>
#include <iostream>

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

RenderWindow::RenderWindow(QWidget *parent) :
    QGLWidget(parent), pbo_(0), tex_(0), solve_(PAUSE)
{
    cuda_pbo_resource_ = 0;
    //TODO, make user define this
    width_ = 256;
    height_ = 256;


    // Resize Widnow
    resize(256,256);
    // Set up Timer to paint scene
    timer = new QTimer(this);
    connect(timer,SIGNAL(timeout()),this,SLOT(updateGL()));

}

void RenderWindow::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1); // set origin to bottom left corner
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void RenderWindow::initializeGL()
{
    makeCurrent();
    glewInit();

    //TODO, query what device to use
    CUDA_SAFE_CALL(cudaGLSetGLDevice(0));

    glDisable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_COLOR_MATERIAL);
        glEnable(GL_BLEND);
        glEnable(GL_POLYGON_SMOOTH);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glClearColor(0, 0, 0, 0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    initPBO();

    fluidSolver_ = new FluidSolver(256,256, 16, 1.0f/256.0f);
    fluidSolver_->init();

    // Start Timer to paint scene
    //timer->start(50);

}

void RenderWindow::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    if (solve_) {
        if (solve_ == STEP) solve_ = PAUSE;
        //TODO calculate dt
        float dt = 1.0f/25.0f;
        fluidSolver_->solve(dt);
    }

    // map PBO to get CUDA device pointer
    CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0));
    uchar4* d_output;
    size_t num_bytes;
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&d_output,
                                                        &num_bytes,
                                                        cuda_pbo_resource_));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    //cudaMemset(d_output, 0, width_*height_*4);

    // call CUDA kernel, writing results to PBO
    fluidSolver_->render(d_output);

    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0));

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // draw using texture

    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
     glBegin(GL_QUADS);
      glTexCoord2f(0, 0); glVertex2f(0, 0);
      glTexCoord2f(1, 0); glVertex2f(1, 0);
      glTexCoord2f(1, 1); glVertex2f(1, 1);
      glTexCoord2f(0, 1); glVertex2f(0, 1);
     glEnd();
    glDisable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderWindow::initPBO()
{
   std::cout << "Initiating Pixel Buffer Object..." << std::endl;
   if (pbo_) {
        // unregister this buffer object from CUDA C
        CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_pbo_resource_));

        // delete old buffer
        glDeleteBuffersARB(1, &pbo_);
        glDeleteTextures(1, &tex_);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo_);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                    width_*height_*sizeof(GLubyte)*4,
                    0,
                    GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource_,
                                             pbo_,
                                             cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 width_,
                 height_,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderWindow::toggleRunStatus()
{
    // If timer is running, we stop it
    if (this->timer->isActive()) {
        solve_ = PAUSE;
        timer->stop();
    } else {
        solve_ = ANIMATE;
        // If timer is stopped, we restart it
        timer->setSingleShot(false);
        timer->start(50);
    }
}
void RenderWindow::stepForward()
{
    // Stop the timer and step forward one step
    solve_ = STEP;
    timer->stop();
    timer->setSingleShot(true);
    timer->start(50);
}
