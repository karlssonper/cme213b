#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "RenderWindow.h"
namespace Ui {
class MainWindow;
}

class FluidSolver;
class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

protected:
    RenderWindow *renderWindow_;
    FluidSolver * fluidSolver_;
private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
