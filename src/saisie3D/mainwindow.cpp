#include <QLayout>
#include <QMessageBox>

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "StdAfx.h"
#include "general/ptxd.h"
#include "private/cElNuage3DMaille.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_glWidget( NULL ),
    m_Dir(""),
    IO()
{
    ui->setupUi(this);

    m_glWidget = new GLWidget;

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(m_glWidget);

    ui->OpenglLayout->setLayout(layout);

    connectActions();
}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::checkForLoadedEntities()
{
    bool loadedEntities = true;
    m_glWidget->displayNewMessage(QString(), GLWidget::SCREEN_CENTER_MESSAGE); //clear (any) message in the middle area

    if (!m_glWidget->hasCloudLoaded())
    {
        m_glWidget->displayNewMessage("Drag & drop files on the window to load them!", GLWidget::SCREEN_CENTER_MESSAGE);
        loadedEntities = false;
    }

    return loadedEntities;
}

void MainWindow::addFiles(const QStringList& filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        m_glWidget->addPly(filenames[i]);
        m_glWidget->updateGL();
    }

    //set default working directory as first file folder
    if (filenames.size())
    {
        QFileInfo fi(filenames[0]);

        m_Dir  = fi.dir();
    }

    checkForLoadedEntities();
}

void MainWindow::toggleFullScreen(bool state)
{
    if (state)
        showFullScreen();
    else
        showNormal();
    m_glWidget->updateGL();
}

void MainWindow::togglePointsSelection(bool state)
{
    if (state)
        m_glWidget->setInteractionMode(GLWidget::SEGMENT_POINTS);
    else
    {
        m_glWidget->setInteractionMode(GLWidget::TRANSFORM_CAMERA);
        m_glWidget->clearPolyline();
    }
}

void MainWindow::doActionDisplayShortcuts()
{
    QMessageBox msgBox;
    QString text;
    text += "Shortcuts:\n\n";
    text += "F2: toggle move mode / selection mode\n";
    text += "F3: toggle full screen\n";
    text += "\n";
    text += "Key +/-: increase/decrease point size\n";
    text += "\n";
    text += "Selection mode:\n";
    text += "    - Left click : add a point to polyline\n";
    text += "    - Right click: close polyline\n";
    text += "    - Echap: delete polyline\n";
    text += "    - Space bar: keep points inside polyline\n";
    text += "    - Suppr: keep points outside polyline\n";
    text += "    - Ctrl+Z: undo all past selections\n";
    text += "\n";
    text += "Ctrl+O: open camera(s) file(s)\n";
    text += "Ctrl+E: export mask(s) file(s)\n";
    text += "Ctrl+S: open camera(s) and export mask(s)\n";

    msgBox.setText(text);
    msgBox.exec();
}

void MainWindow::connectActions()
{
    connect(m_glWidget,	SIGNAL(filesDropped(const QStringList&)), this,	SLOT(addFiles(const QStringList&)));

    connect(m_glWidget,	SIGNAL(mouseWheelRotated(float)),			this,       SLOT(echoMouseWheelRotate(float)));

    connect(ui->actionFullScreen,       SIGNAL(toggled(bool)), this, SLOT(toggleFullScreen(bool)));

    connect(ui->actionHelpShortcuts,    SIGNAL(triggered()),   this, SLOT(doActionDisplayShortcuts()));

    connect(ui->actionSetViewTop,		SIGNAL(triggered()),   this, SLOT(setTopView()));
    connect(ui->actionSetViewBottom,	SIGNAL(triggered()),   this, SLOT(setBottomView()));
    connect(ui->actionSetViewFront,		SIGNAL(triggered()),   this, SLOT(setFrontView()));
    connect(ui->actionSetViewBack,		SIGNAL(triggered()),   this, SLOT(setBackView()));
    connect(ui->actionSetViewLeft,		SIGNAL(triggered()),   this, SLOT(setLeftView()));
    connect(ui->actionSetViewRight,		SIGNAL(triggered()),   this, SLOT(setRightView()));

    //"Points selection" menu
    connect(ui->actionTogglePoints_selection, SIGNAL(toggled(bool)), this, SLOT(togglePointsSelection(bool)));

    connect(ui->actionLoad_camera,		SIGNAL(triggered()),   this, SLOT(loadCameras()));
    connect(ui->actionExport_mask,		SIGNAL(triggered()),   this, SLOT(exportMasks()));
    connect(ui->actionLoad_and_Export,	SIGNAL(triggered()),   this, SLOT(loadAndExport()));
}

void MainWindow::setTopView()
{
    m_glWidget->setView(MM_TOP_VIEW);
}

void MainWindow::setBottomView()
{
    m_glWidget->setView(MM_BOTTOM_VIEW);
}

void MainWindow::setFrontView()
{
    m_glWidget->setView(MM_FRONT_VIEW);
}

void MainWindow::setBackView()
{
    m_glWidget->setView(MM_BACK_VIEW);
}

void MainWindow::setLeftView()
{
    m_glWidget->setView(MM_LEFT_VIEW);
}

void MainWindow::setRightView()
{
    m_glWidget->setView(MM_RIGHT_VIEW);
}

void MainWindow::echoMouseWheelRotate(float wheelDelta_deg)
{
    GLWidget* sendingWindow = dynamic_cast<GLWidget*>(sender());
    if (!sendingWindow)
        return;

    sendingWindow->onWheelEvent(wheelDelta_deg);
}

void MainWindow::on_actionUndo_triggered()
{
     m_glWidget->undoAll();
}

void MainWindow::loadCameras()
{
    IO.m_FilenamesIn = QFileDialog::getOpenFileNames(this, tr("Open Camera Files"),m_Dir.path(), tr("Files (*.xml)"));

    for (int aK=0;aK < IO.m_FilenamesIn.size();++aK)
    {
        cElNuage3DMaille *aNuage = cElNuage3DMaille::FromFileIm(IO.m_FilenamesIn[aK].toStdString());
        m_Cameras.push_back(aNuage);
    }

    IO.SetFilenamesOut();

    int toto = 0;
}

void MainWindow::exportMasks()
{
    //for (int i=0; i < m_glWidget->m_ply)
    for (int aK=0;aK < m_Cameras.size();++aK)
    {
        Pt3dr pt(0,0,0);
        Pt2dr ptIm = (Camera(aK))->Terrain2Index(pt);
    }
}

void MainWindow::loadAndExport()
{
    loadCameras();
    exportMasks();
}

cLoader::cLoader()
 : m_FilenamesIn(),
   m_FilenamesOut()
{}

cLoader::~cLoader(){}

void cLoader::SetFilenamesOut()
{
    for (int aK=0;aK < m_FilenamesIn.size();++aK)
    {
        QFileInfo fi(m_FilenamesIn[aK]);

        m_FilenamesOut.push_back(fi.absoluteFilePath () + "_mask.tif");
    }
}
