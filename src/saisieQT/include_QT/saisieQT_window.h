#ifndef SAISIEQTWINDOW_H
#define SAISIEQTWINDOW_H

#include "Elise_QT.h"
#include "GLWidgetSet.h"
#include "Settings.h"

void setStyleSheet(QApplication &app);

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

    void applyParams();

    void labelShowMode(bool state);

    void refreshPts();

    void setLayout(uint sy);

    bool loadPly(const QStringList& filenames);

    bool loadImages(const QStringList& filenames);

    bool loadCameras(const QStringList& filenames);

    void setUI();

    void updateUI();

    bool eventFilter(QObject *object, QEvent *event);

    QTableView *tableView_PG();

    QTableView *tableView_Images();

    QTableView *tableView_Objects();

    void    resizeTables();

    void    setModel(QAbstractItemModel *model_Pg, QAbstractItemModel *model_Images);

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

    void    setParams(cParameters *params);



    deviceIOCamera* devIOCamera() const;
    void setDevIOCamera(deviceIOCamera* devIOCamera);

    deviceIOImage* devIOImage() const;
    void setDevIOImage(deviceIOImage* devIOImage);

    int hg_revision() const;
    void setHg_revision(int hg_revision);

    QString banniere() const;
    void setBanniere(const QString& banniere);

	QString textToolBar() const;
	void setTextToolBar(const QString& textToolBar);

public slots:

	//! Try to load a list of files
	void addFiles(const QStringList& filenames, bool setGLData = true);

    void zoomFactor(int aFactor);

    void closeAll(bool checkBeforeClose = true);

    void closeCurrentWidget();

    void openRecentFile();

    void progression();

    cEngine* getEngine(){return _Engine;}

    void closeEvent(QCloseEvent *event);

    void redraw(bool nbWidgetsChanged=false);

    void setAutoName(QString);

    void setGamma(float);

    cParameters* getParams() { return _params; }

    void updateSaveActions();
    void resetSavedState();

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
    void on_actionAdd_inside_triggered();
    void on_actionAdd_outside_triggered();
    void on_actionSelect_none_triggered();
    void on_actionInvertSelected_triggered();
    void on_actionSelectAll_triggered();
    void on_actionReset_triggered();
    void on_actionRemove_inside_triggered();
    void on_actionRemove_outside_triggered();
    void on_actionUndo_triggered();
    void on_actionRedo_triggered();

    //File Menu
    void on_actionLoad_plys_triggered();
    void on_actionLoad_camera_triggered();
    void on_actionLoad_image_triggered();
    void on_actionSave_masks_triggered();
    void on_actionSave_as_triggered();
    void on_actionSettings_triggered();

    void on_menuFile_triggered();

    //Help Menu
    void on_actionHelpShortcuts_triggered();
    void on_actionAbout_triggered();

    // Tools
    void on_actionRule_toggled(bool check);

    void resizeEvent(QResizeEvent *);
    void moveEvent(QMoveEvent *);

    void setNavigationType(int val);

    void on_actionShow_Zoom_window_toggled(bool show);

    void on_actionShow_3D_view_toggled(bool show);

    void on_actionShow_list_polygons_toggled(bool show);

    void selectionObjectChanged(const QItemSelection& select, const QItemSelection& unselect);

    void updateMask(bool reloadMask = true);

    void on_actionConfirm_changes_triggered();

protected:

    //! Connects all QT actions to slots
    void connectActions();

    void setModelObject(QAbstractItemModel* model_Objects);
private:

    void                    createRecentFileMenu();

    void                    setCurrentFile(const QString &fileName);
    void                    updateRecentFileActions();
    QString                 strippedName(const QString &fullFileName);

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

    deviceIOCamera*			_devIOCamera;

    int						_hg_revision;

    QString					_banniere;

	QString					_textToolBar;

};


class ObjectsSFModel : public QSortFilterProxyModel
{
    Q_OBJECT

public:
    ObjectsSFModel(QObject *parent = 0): QSortFilterProxyModel(parent){}

protected:
    bool filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const;

};

class ModelObjects : public QAbstractTableModel
{
    Q_OBJECT
public:

    ModelObjects(QObject *parent, HistoryManager* hMag);

    int             rowCount(const QModelIndex &parent = QModelIndex()) const ;

    int             columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant        data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    QVariant        headerData(int section, Qt::Orientation orientation, int role) const;

    bool            setData(const QModelIndex & index, const QVariant & value, int role = Qt::EditRole);

    Qt::ItemFlags   flags(const QModelIndex &index) const;

    bool            insertRows(int row, int count, const QModelIndex & parent = QModelIndex());

    static QStringList     getSelectionMode();

private:

    HistoryManager *		_hMag;

};

class ComboBoxDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    ComboBoxDelegate(const QStringList &listCombo, int size = 0, QObject *parent = 0);

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const
#if ELISE_QT_VERSION == 5
    Q_DECL_OVERRIDE
#endif
    ;

    void setEditorData(QWidget *editor, const QModelIndex &index) const
#if ELISE_QT_VERSION == 5
    Q_DECL_OVERRIDE
#endif
    ;
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const
#if ELISE_QT_VERSION == 5
    Q_DECL_OVERRIDE
#endif
    ;

    void updateEditorGeometry(QWidget *editor,
        const QStyleOptionViewItem &option, const QModelIndex &index) const
#if ELISE_QT_VERSION == 5
    Q_DECL_OVERRIDE
#endif
    ;
private:
    int             _size;
    QStringList		_enumString;
};


#endif // MAINWINDOW_H
