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
#include <QGridLayout>

#include "Engine.h"
#include "GLWidgetSet.h"

namespace Ui {
class MainWindow;
}

const QColor colorBG0("#323232");
const QColor colorBG1("#808080");
const QColor colorBorder("#606060");

//! Interface mode
enum UI_MODE {  MASK2D,         /**< Image mask mode  **/
                MASK3D,         /**< Point cloud mask **/
                POINT2D_INIT,	/**< Points in Image (SaisiePointInit) **/
                POINT2D_PREDICT /**< Points in Image (SaisiePointPredic) **/
};


class MainWindow : public QMainWindow, public GLWidgetSet
{
    Q_OBJECT

public:
    explicit MainWindow( Pt2di aSzW, Pt2di aNbFen, int mode = MASK3D, QString pointName = "", QWidget *parent = 0 );
    ~MainWindow();

    void setPostFix(QString str);

    void setNbFen(QPoint nb);
    void setSzFen(QPoint sz);

    void runProgressDialog(QFuture<void> future);

public slots:

    //! Try to load a list of files
    void addFiles(const QStringList& filenames);

    void zoomFactor(int aFactor);

    void closeAll();

    void closeCurrentWidget();

    void openRecentFile();

    void progression();

    void setMode();

    cEngine* getEngine(){return _Engine;}

	void setGamma(float aGamma);

protected slots:

    void changeCurrentWidget(void* cuWid);

    //View Menu
    void on_actionShow_axis_toggled(bool);
    void on_actionShow_ball_toggled(bool);
    void on_actionShow_cams_toggled(bool);
    void on_actionShow_bbox_toggled(bool);

    void on_actionFullScreen_toggled(bool);
    void on_actionShow_messages_toggled(bool);
    void on_actionToggleMode_toggled(bool);

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
    void on_actionZoom_fit_triggered();

    //Selection Menu
    void on_actionAdd_triggered();
    void on_actionSelect_none_triggered();
    void on_actionInvertSelected_triggered();
    void on_actionSelectAll_triggered();
    void on_actionReset_triggered();
    void on_actionRemove_triggered();
    void on_actionUndo_triggered(){ undo(); }
    void on_actionRedo_triggered(){ undo(false); }

    //File Menu
    void on_actionLoad_plys_triggered();
    void on_actionLoad_camera_triggered();
    void on_actionLoad_image_triggered();
    void on_actionSave_masks_triggered();
    void on_actionSave_as_triggered();
    void on_actionSave_selection_triggered();

    //Help Menu
    void on_actionHelpShortcuts_triggered();
    void on_actionAbout_triggered();

protected:

    //! Connects all QT actions to slots
    void connectActions();  

private:
    void                    createRecentFileMenu();

    void                    setCurrentFile(const QString &fileName);
    void                    updateRecentFileActions();
    QString                 strippedName(const QString &fullFileName);

    void                    undo(bool undo = true);

    int *                   _incre;

    Ui::MainWindow*         _ui;

    cEngine*                _Engine;

    QFutureWatcher<void>    _FutureWatcher;
    QProgressDialog*        _ProgressDialog;

    enum { MaxRecentFiles = 3 };
    QAction *               _recentFileActs[MaxRecentFiles];
    QString                 _curFile;

    QMenu*                  _RFMenu; //recent files menu

    int                     _mode;

    QPoint                  _nbFen;
    QPoint                  _szFen;

    QSignalMapper*          _signalMapper;
    QGridLayout*            _layout;
    QGridLayout*            _zoomLayout;

    QString                 _ptName;
};
#endif // MAINWINDOW_H
