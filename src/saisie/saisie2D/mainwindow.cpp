#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::addFiles(const QStringList& filenames)
{
    if (filenames.size())
    {
        QFileInfo fi(filenames[0]);

        //set default working directory as first file subfolder
        QDir Dir = fi.dir();
        Dir.cdUp();
        m_Engine2D->setDir(Dir);
        m_Engine2D->setFilename();
    }
}
