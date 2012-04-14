#ifndef RENDERWINDOW_H
#define RENDERWINDOW_H

#include <QGLWidget>
#include <cuda_gl_interop.h>

class FluidSolver;
class RenderWindow : public QGLWidget
{
    Q_OBJECT
public:
    explicit RenderWindow(QWidget *parent = 0, FluidSolver *solver = 0);
    
signals:
    
public slots:

protected:
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
};

#endif // RENDERWINDOW_H
