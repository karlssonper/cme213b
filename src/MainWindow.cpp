#include "MainWindow.h"
#include "ui_mainwindow.h"
#include "FluidSolver.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->gravitySlider->setMinimum(0);
    ui->gravitySlider->setMaximum(10);
    ui->gravityNumber->display(0);

    fluidSolver_ = new FluidSolver(256,256, 16, 1.0f/256.0f);
    renderWindow_ = new RenderWindow(this,fluidSolver_);
    ui->scrollArea->setWidget(renderWindow_);

    connect(ui->gravitySlider,SIGNAL(sliderMoved(int)),
            ui->gravityNumber,SLOT(display(int)));
    connect(ui->gravitySlider,SIGNAL(sliderMoved(int)),
            renderWindow_,SLOT(gravityChanged(int)));
}

MainWindow::~MainWindow()
{
    delete ui;
}
