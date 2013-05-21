#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "GLWidget.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


    //! Checks for loaded entities
    /** If none, a message is displayed to invite the user
        to drag & drop files.
    **/
    bool checkForLoadedEntities();

public slots:
    //! Tries to load a list of files
    /** \param filenames list of all filenames
    **/
    void addFiles(const QStringList& filenames);

private slots:
    //void on_pushButton_clicked();

    void on_actionUndo_triggered();

protected slots:
    void doActionDisplayShortcuts();
    void toggleFullScreen(bool);
    void togglePointsSelection(bool state);

    //default views
    void setFrontView();
    void setBottomView();
    void setTopView();
    void setBackView();
    void setLeftView();
    void setRightView();

    void echoMouseWheelRotate(float);

protected:

    //! Connects all QT actions to slots
    void connectActions();

private:
    Ui::MainWindow *ui;
    GLWidget *m_glWidget;
};

#endif // MAINWINDOW_H
