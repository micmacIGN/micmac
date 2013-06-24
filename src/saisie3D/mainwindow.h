#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#ifndef  WIN32
    #include "GL/glew.h"
#endif
#include <QMainWindow>

#include "GLWidget.h"
#include "Engine.h"

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
    /** If none, a message is displayed to invite user
        to drag & drop files.
    **/
    bool checkForLoadedEntities();

public slots:
    //! Try to load a list of files
    void addFiles(const QStringList& filenames);

private slots:
    void on_actionUndo_triggered();

protected slots:
    void doActionDisplayShortcuts();
    void toggleFullScreen(bool);
    void toggleShowAxis(bool);
    void toggleShowBall(bool);
    void toggleShowCams(bool);
    void toggleShowMessages(bool);
    void togglePointsSelection(bool state);

    void addPoints();
    void deletePoint();

    //default views
    void setFrontView();
    void setBottomView();
    void setTopView();
    void setBackView();
    void setLeftView();
    void setRightView();

    void echoMouseWheelRotate(float);

    void loadPlys();
    void loadCameras();
    void exportMasks();
    void loadAndExport();
    void saveSelectionInfos();

protected:

    //! Connects all QT actions to slots
    void connectActions();

private:
    Ui::MainWindow *ui;

    GLWidget *m_glWidget;

    cEngine  *m_Engine;
};
#endif // MAINWINDOW_H
