#ifndef RENDERWINDOW_H
#define RENDERWINDOW_H

#include <QGLWidget>

class FluidSolver;
class RenderWindow : public QGLWidget
{
    Q_OBJECT
public:
    explicit RenderWindow(QWidget *parent = 0, FluidSolver *solver = 0);
    
signals:
    
public slots:

protected:
    void paintGL();

    FluidSolver * fluidSolver_;
    
};

#endif // RENDERWINDOW_H
