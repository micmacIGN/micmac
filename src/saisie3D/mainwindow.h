#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QDir>

#include "GLWidget.h"

#include "StdAfx.h"
#include "general/ptxd.h"
#include "private/cElNuage3DMaille.h"

#ifdef Int
    #undef Int
#endif

namespace Ui {
class MainWindow;
}

class cLoader
{
    public:

        cLoader();
        ~cLoader();

        QStringList m_FilenamesIn;
        QStringList m_FilenamesOut;

        void SetFilenamesOut();
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

    QStringList m_FilenameOut;

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

    cElNuage3DMaille * &	Camera(int aK) {return m_Cameras[aK];}

protected:

    //! Connects all QT actions to slots
    void connectActions();

private:
    Ui::MainWindow *ui;
    GLWidget *m_glWidget;
    QDir m_Dir;
    QVector <cElNuage3DMaille *> m_Cameras;
    cLoader IO;
};
#endif // MAINWINDOW_H
