#ifndef MAINWINDOW_H
#define MAINWINDOW_H

//#ifndef  WIN32
//#ifndef __APPLE__
//#include "GL/glew.h"
//#endif
//#endif
#include <QMainWindow>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QProgressDialog>
#include <QTimer>
#include <QSignalMapper>

#include "GLWidget.h"
#include "Engine.h"

class GLWidget;

namespace Ui {
class MainWindow;
}

const QColor colorBG0(65,65,60);
const QColor colorBG1(120,115,115);

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(bool mode2D = false, QWidget *parent = 0);
    ~MainWindow();

    //! Checks for loaded data
    bool checkForLoadedData();

    void setPostFix(QString str);

public slots:

    //! Try to load a list of files
    void addFiles(const QStringList& filenames);

    void selectedPoint(uint idC,uint idV,bool select);

    void changeMode(bool mode);

//protected slots:

    void displayShortcuts();

    void toggleFullScreen(bool);
    void toggleShowAxis(bool);
    void toggleShowBall(bool);
    void toggleShowBBox(bool);
    void toggleShowCams(bool);
    void toggleShowMessages(bool);
    void toggleSelectionMode(bool);
    void toggle2D3D();

    void add();
    void selectNone();
    void invertSelected();
    void selectAll();
    void removeFromSelection();
    void reset();

    //default views
    void setFrontView();
    void setBottomView();
    void setTopView();
    void setBackView();
    void setLeftView();
    void setRightView();

    void resetView();

    //zoom
    void zoomPlus();
    void zoomMoins();
    void zoomFit();
    void zoomFactor(int aFactor );

    void echoMouseWheelRotate(float);

    void loadPlys();
    void loadCameras();
    void loadImage();
    void closeAll();
    void exportMasks();
    void exportMasksAs();
    void saveSelectionInfos();

    void openRecentFile();

    void progression();

    void setMode2D(bool mBool);
    bool getMode2D() {return _bMode2D;}

    cEngine* getEngine(){return _Engine;}

	void setGamma(float aGamma);

protected:

    //! Connects all QT actions to slots
    void connectActions();  

private:

    void                    createMenus();

    void                    setCurrentFile(const QString &fileName);
    void                    updateRecentFileActions();
    QString                 strippedName(const QString &fullFileName);

    int *                   _incre;

    Ui::MainWindow*         _ui;

    GLWidget*               _glWidget;

    cEngine*                _Engine;

    QFutureWatcher<void>    _FutureWatcher;
    QProgressDialog*        _ProgressDialog;

    enum { MaxRecentFiles = 3 };
    QAction *               _recentFileActs[MaxRecentFiles];
    QString                 _curFile;
    QStringList             _FilenamesIn;

    QMenu*                  _RFMenu; //recent files menu

    bool                    _bMode2D;
};
#endif // MAINWINDOW_H
