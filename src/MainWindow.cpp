#include "MainWindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->gravitySlider->setMinimum(0);
    ui->gravitySlider->setMaximum(10);
    ui->gravityNumber->display(0);

    renderWindow_ = new RenderWindow(this);

    ui->scrollArea->setWidget(renderWindow_);
    connect(ui->gravitySlider,SIGNAL(sliderMoved(int)),
            ui->gravityNumber,SLOT(display(int)));
    connect(ui->playButton,SIGNAL(clicked()),
            renderWindow_,SLOT(toggleRunStatus()));
    connect(ui->stepButton,SIGNAL(clicked()),
            renderWindow_,SLOT(stepForward()));
    connect(ui->playButton,SIGNAL(clicked()),
            this,SLOT(togglePlayButtonStatus()));
}

void MainWindow::togglePlayButtonStatus()
{
    if (ui->playButton->text() == QString("Play"))
        ui->playButton->setText(QString("Pause"));
    else
        ui->playButton->setText(QString("Play"));
}

MainWindow::~MainWindow()
{
    delete ui;
}
