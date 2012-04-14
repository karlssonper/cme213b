#include <GL/glew.h>
#include "RenderWindow.h"
#include "FluidSolver.h"


RenderWindow::RenderWindow(QWidget *parent, FluidSolver *solver) :
    QGLWidget(parent), fluidSolver_(solver), pbo_(0), tex_(0)
{
    //TODO, make user define this
    width_ = 256;
    height_ = 256;
}

void RenderWindow::initializeGL()
{
    glewInit();

    //TODO, query what device to use
    cudaGLSetGLDevice(0);

    initPBO();
}

void RenderWindow::paintGL()
{
    //TODO calculate dt
    float dt = 1.0f/25.0f;

    fluidSolver_->solve(dt);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output,
                                         &num_bytes,
                                         cuda_pbo_resource_);
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    cudaMemset(d_output, 0, width_*height_*4);

    // call CUDA kernel, writing results to PBO
    fluidSolver_->render();

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0);

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

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
       if (pbo_) {
            // unregister this buffer object from CUDA C
            cudaGraphicsUnregisterResource(cuda_pbo_resource_);

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
        cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource_,
                                     pbo_,
                                     cudaGraphicsMapFlagsWriteDiscard);

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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
}
