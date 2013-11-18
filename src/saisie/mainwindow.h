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

    void zoomFactor(int aFactor );

    void closeAll();

    void echoMouseWheelRotate(float);

    void openRecentFile();

    void progression();

    void setMode2D(bool mBool);
    bool getMode2D() {return _bMode2D;}

    cEngine* getEngine(){return _Engine;}

	void setGamma(float aGamma);

protected slots:

    //View Menu
    void on_actionShow_axis_toggled(bool);
    void on_actionShow_ball_toggled(bool);
    void on_actionShow_cams_toggled(bool);
    void on_actionShow_bounding_box_toggled(bool);

    void on_actionFullScreen_toggled(bool);
    void on_actionShow_help_messages_toggled(bool);
    void on_actionToggleMode_toggled(bool);

    void on_action2D_3D_mode_triggered();
    void on_actionHelpShortcuts_triggered();
    void on_actionReset_view_triggered();

    void on_actionSetViewTop_triggered();
    void on_actionSetViewBottom_triggered();
    void on_actionSetViewFront_triggered();
    void on_actionSetViewBack_triggered();
    void on_actionSetViewLeft_triggered();
    void on_actionSetViewRight_triggered();

    //Zoom
    void on_actionZoom_Plus_triggered();
    void on_actionZoom_Moins_triggered();
    void on_actionZoom_Fit_triggered();

    //Selection Menu
    void on_actionAdd_triggered();
    void on_actionSelect_none_triggered();
    void on_actionInvertSelected_triggered();
    void on_actionSelectAll_triggered();
    void on_actionReset_triggered();
    void on_actionRemove_triggered();

    //File Menu
    void on_actionLoad_plys_triggered();
    void on_actionLoad_camera_triggered();
    void on_actionLoad_image_triggered();
    void on_actionSave_masks_triggered();
    void on_actionSave_as_triggered();
    void on_actionSave_selection_triggered();

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
