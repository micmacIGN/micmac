#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#ifndef  WIN32
#ifndef __APPLE__
#include "GL/glew.h"
#endif
#endif
#include <QMainWindow>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QProgressDialog>

#include "GLWidget.h"
#include "Engine.h"

class GLWidget;

namespace Ui {
class MainWindow;
}

//const QColor colorBG0(30,132,181);
//const QColor colorBG1(70,70,70);

const QColor colorBG0(65,65,60);
const QColor colorBG1(120,115,115);

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

    static void progress(int var, void *obj);

signals:

    void progressInc(int val);

public slots:

    //! Try to load a list of files
    void addFiles(const QStringList& filenames);
    void selectedPoint(uint idC,uint idV,bool select);

//protected slots:

    void doActionDisplayShortcuts();
    void toggleFullScreen(bool);
    void toggleShowAxis(bool);
    void toggleShowBall(bool);
    void toggleShowBBox(bool);
    void toggleShowCams(bool);
    void toggleShowMessages(bool);
    void togglePointsSelection(bool state);

    void addPoints();
    void selectNone();
    void invertSelected();
    void selectAll();
    void removeFromSelection();

    void deletePolylinePoint();

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
    void closeAll();
    void exportMasks();
    void loadAndExport();
    void saveSelectionInfos();

    void openRecentFile();

protected:

    //! Connects all QT actions to slots
    void connectActions();  


private:

    void emitProgress(int progress);
    void createMenus();

    void setCurrentFile(const QString &fileName);
    void updateRecentFileActions();
    QString strippedName(const QString &fullFileName);

    Ui::MainWindow          *ui;

    GLWidget*               m_glWidget;

    cEngine*                m_Engine;

    QFutureWatcher<void>    FutureWatcher;
    QProgressDialog*        ProgressDialog;

    QAction *separatorAct;

    enum { MaxRecentFiles = 3 };
    QAction *               recentFileActs[MaxRecentFiles];
    QString                 m_curFile;
    QStringList             m_FilenamesIn;

};
#endif // MAINWINDOW_H
