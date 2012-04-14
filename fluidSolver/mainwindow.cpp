#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "FluidSolver.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->ui->gravitySlider->setMinimum(0);
    this->ui->gravitySlider->setMaximum(10);

    this->ui->gravityNumber->display(0);


    this->fluidSolver = new FluidSolver(64,64,64);
    this->renderWindow = new fluidRender(this,fluidSolver);


    this->ui->scrollArea->setWidget(this->renderWindow);

    connect(this->ui->gravitySlider,SIGNAL(sliderMoved(int)),
            this->ui->gravityNumber,SLOT(display(int)));
    connect(this->ui->gravitySlider,SIGNAL(sliderMoved(int)),
            renderWindow,SLOT(gravityChanged(int)));

}

MainWindow::~MainWindow()
{
    delete ui;
}
