#include "fluidrender.h"
#include "FluidSolver.h"

fluidRender::fluidRender(QWidget *parent, FluidSolver *solver) :
    QGLWidget(parent), solver_(solver)
{
}

void fluidRender::paintGL()
{
    // pass gravity_ to Solver here
    // solver_.solve()

    // Call fluidsolver render()

}
