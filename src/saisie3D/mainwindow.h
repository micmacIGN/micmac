#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDir>

#include "GLWidget.h"
#include "Data.h"

//for cEngine
#include "Cloud.h"

#ifdef Int
    #undef Int
#endif

namespace Ui {
class MainWindow;
}

class cLoader : QObject
{
    public:

        cLoader();
        ~cLoader();

        cElNuage3DMaille * loadCamera(string aFile);
        Cloud* loadCloud( string i_ply_file );

        vector <cElNuage3DMaille *> loadCameras();

        void setDir(QDir aDir){m_Dir = aDir;}

        QStringList m_FilenamesIn;
        QStringList m_FilenamesOut;

        QDir        m_Dir;

        void SetFilenamesOut();
};

class cEngine
{
    public:

        cEngine();
        ~cEngine();

        void addFiles(QStringList);
        void loadCameras();

        void doMasks();

        cData*   getData()  {return m_Data;}

        cLoader *m_Loader;

    private:

        cData   *m_Data;
};

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

    //cEngine* getEngine(){return m_Engine;}

public slots:
    //! Tries to load a list of files
    /** \param filenames list of all filenames
    **/
    void addFiles(const QStringList& filenames);

private slots:
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

    void loadCameras();
    void exportMasks();
    void loadAndExport();

protected:

    //! Connects all QT actions to slots
    void connectActions();

private:
    Ui::MainWindow *ui;

    GLWidget *m_glWidget;

    cEngine  *m_Engine;
};
#endif // MAINWINDOW_H
