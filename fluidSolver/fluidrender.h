#ifndef FLUIDRENDER_H
#define FLUIDRENDER_H

#include <QGLWidget>

class FluidSolver;

class fluidRender : public QGLWidget
{
    Q_OBJECT
public:
    explicit fluidRender(QWidget *parent = 0, FluidSolver *solver = 0);
    
signals:
    
public slots:
   void gravityChanged(int gravity) {gravity_ = gravity;}

protected:
    void paintGL();

    FluidSolver *solver_;
    
};

#endif // FLUIDRENDER_H
