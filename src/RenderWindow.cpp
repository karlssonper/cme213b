#include "RenderWindow.h"
#include "FluidSolver.h"

RenderWindow::RenderWindow(QWidget *parent, FluidSolver *solver) :
    QGLWidget(parent), fluidSolver_(solver)
{
}

void RenderWindow::paintGL()
{
    // solver_.solve()

    // Call fluidsolver render()

}
