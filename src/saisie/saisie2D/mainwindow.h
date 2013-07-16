#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "Engine2D.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:

    //! Try to load a list of files
    void addFiles(const QStringList& filenames);
    
private:
    Ui::MainWindow *ui;

    cEngine2D*        m_Engine2D;
};

#endif // MAINWINDOW_H
