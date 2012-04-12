#include "fluidrender.h"
#include "FluidSolver.h"

fluidRender::fluidRender(QWidget *parent, FluidSolver *solver) :
    QGLWidget(parent), solver_(solver)
{
}

void fluidRender::paintGL()
{
    // solver_.solve()

    // Call fluidsolver render()

}
