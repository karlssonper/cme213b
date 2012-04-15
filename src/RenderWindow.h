#ifndef RENDERWINDOW_H
#define RENDERWINDOW_H

#include <QGLWidget>
#include <QTimer>
#include <cuda_gl_interop.h>

class FluidSolver;
class RenderWindow : public QGLWidget
{
    Q_OBJECT
public:
    explicit RenderWindow(QWidget *parent = 0);
    
signals:
    
public slots:

protected:
    void resizeGL(int w, int h);
    void initializeGL();
    void paintGL();
    FluidSolver * fluidSolver_;
    GLuint width_;
    GLuint height_;
    GLuint pbo_;
    GLuint tex_;
    struct cudaGraphicsResource *cuda_pbo_resource_;
private:
    void initPBO();
    unsigned int image_width;
    unsigned int image_height;
    QGLBuffer*   pixelBuffer;
    GLuint*      textureID;

    QTimer* timer;
};

#endif // RENDERWINDOW_H
