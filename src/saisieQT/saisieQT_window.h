#ifndef SAISIEQTWINDOW_H
#define SAISIEQTWINDOW_H

#include "general/CMake_defines.h"

#ifdef ELISE_Darwin
    #include "OpenGL/gl.h"
#else
    #include "GL/gl.h"
#endif

#include <QMainWindow>
#include <QFutureWatcher>
#include <QtConcurrentRun>
#include <QProgressDialog>
#include <QTimer>
#include <QSignalMapper>
#include <QGridLayout>
#include <QTableView>
#include <QTreeView>

#include "Engine.h"
#include "GLWidgetSet.h"
#include "Settings.h"
#include "qdesktopwidget.h"

#include "Tree.h"

namespace Ui {
class SaisieQtWindow;
}

const QColor colorBorder("#606060");

class SaisieQtWindow : public QMainWindow, public GLWidgetSet
{
    Q_OBJECT

public:

    explicit SaisieQtWindow( int mode = MASK3D, QWidget *parent = 0 );
    ~SaisieQtWindow();

    void setPostFix(QString str);
    QString getPostFix();

    void runProgressDialog(QFuture<void> future);

    void readSettings();

    void writeSettings();

    void applyParams();

    void labelShowMode(bool state);

    void refreshPts();

    void setLayout(uint sy);

    void loadPly(const QStringList& filenames);

    void setUI();

    void updateUI();

    bool eventFilter(QObject *object, QEvent *event);

    QTableView *tableView_PG();

    QTableView *tableView_Images();

    QTableView *tableView_Objects();

    void    resizeTables();

    void    setModel(QAbstractItemModel *model_Pg, QAbstractItemModel *model_Images, QAbstractItemModel *model_Objects);

    void    SelectPointAllWGL(QString pointName = QString(""));

    void    SetDataToGLWidget(int idGLW, cGLData *glData);

    void    loadPlyIn3DPrev(const QStringList &filenames,cData* dataCache);

    void    setCurrentPolygonIndex(int idx);
    void    normalizeCurrentPolygon(bool nrm);

    void    initData();

    int     appMode() const;

    void    setAppMode(int appMode);

    QAction *addCommandTools(QString nameCommand);

    int     checkBeforeClose();

    cParameters *params() const;

    void setParams(cParameters *params);

public slots:

    //! Try to load a list of files
    void addFiles(const QStringList& filenames, bool setGLData = true);

    void zoomFactor(int aFactor);

    void closeAll();

    void closeCurrentWidget();

    void openRecentFile();

    void progression();

    cEngine* getEngine(){return _Engine;}

    void closeEvent(QCloseEvent *event);

    void redraw(bool nbWidgetsChanged=false);

    void setAutoName(QString);

    void setGamma(float);

    cParameters* getParams() { return _params; }

signals:

    void showRefuted(bool);

    void removePoint(QString pointName); //signal used when Treeview is edited

    void selectPoint(QString pointName);

    void setName(QString); //signal coming from cSettingsDlg throw MainWindow

    void imagesAdded(int, bool);

    void undoSgnl(bool);

    void sCloseAll();

    void sgnClose();

protected slots:

    void setImagePosition(QPointF pt);
    void setZoom(float);

    void setImageName(QString name);

    void changeCurrentWidget(void* cuWid);

    //View Menu
    void on_actionSwitch_axis_Y_Z_toggled(bool state);
    void on_actionShow_axis_toggled(bool);
    void on_actionShow_ball_toggled(bool);
    void on_actionShow_cams_toggled(bool);
    void on_actionShow_bbox_toggled(bool);
    void on_actionShow_grid_toggled(bool);

    void on_actionFullScreen_toggled(bool);
    void on_actionShow_messages_toggled(bool);
    void on_actionShow_names_toggled(bool);
    void on_actionShow_refuted_toggled(bool);
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
    void on_actionSettings_triggered();

    //Help Menu
    void on_actionHelpShortcuts_triggered();
    void on_actionAbout_triggered();

    void resizeEvent(QResizeEvent *);
    void moveEvent(QMoveEvent *);

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

    Ui::SaisieQtWindow*     _ui;

    cEngine*                _Engine;

    QFutureWatcher<void>    _FutureWatcher;
    QProgressDialog*        _ProgressDialog;

    enum { MaxRecentFiles = 3 };
    QAction *               _recentFileActs[MaxRecentFiles];
    QString                 _curFile;

    QMenu*                  _RFMenu; //recent files menu

    QSignalMapper*          _signalMapper;
    QGridLayout*            _layout_GLwidgets;
    QGridLayout*            _zoomLayout;

    cParameters*            _params;

    cHelpDlg*               _helpDialog;

    int                     _appMode;

    bool                    _bSaved;

};
#endif // MAINWINDOW_H
