#include <QMessageBox>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_Engine(new cEngine)
{
    ui->setupUi(this);

    m_glWidget = new GLWidget(this,m_Engine->getData());

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(m_glWidget);

    ui->OpenglLayout->setLayout(layout);

    connectActions();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_glWidget;
    delete m_Engine;
}

bool MainWindow::checkForLoadedEntities()
{
    bool loadedEntities = true;
    m_glWidget->displayNewMessage(QString()); //clear (any) message in the middle area

    if (!m_glWidget->hasDataLoaded())
    {
        m_glWidget->displayNewMessage("Drag & drop files on window to load them!");
        loadedEntities = false;
    }

    return loadedEntities;
}

void MainWindow::addFiles(const QStringList& filenames)
{
    if (filenames.size())
    {
        QFileInfo fi(filenames[0]);

        //set default working directory as first file subfolder
        QDir Dir = fi.dir();
        Dir.cdUp();
        m_Engine->setDir(Dir);

        #ifdef _DEBUG
            printf("adding files %s", filenames[0]);
        #endif

        if (fi.suffix() == "ply")
        {
            m_Engine->loadClouds(filenames);

            m_glWidget->setData(m_Engine->getData());
        }
        else if (fi.suffix() == "xml")
        {
            m_Engine->loadCameras(filenames);

            m_glWidget->setCameraLoaded(true);
            m_glWidget->updateGL();
        }

        checkForLoadedEntities();
    }
}

void MainWindow::toggleFullScreen(bool state)
{
    if (state)
        showFullScreen();
    else
        showNormal();
}

void MainWindow::toggleShowBall(bool state)
{
    m_glWidget->showBall(state);
}

void MainWindow::toggleShowAxis(bool state)
{
    m_glWidget->showAxis(state);
}

void MainWindow::toggleShowCams(bool state)
{
    m_glWidget->showCams(state);
}

void MainWindow::toggleShowMessages(bool state)
{
    m_glWidget->showMessages(state);
}

void MainWindow::togglePointsSelection(bool state)
{
    if (state)
    {
        m_glWidget->setInteractionMode(GLWidget::SEGMENT_POINTS);

        if (m_glWidget->hasCloudLoaded()&&m_glWidget->showMessages())
        {
            m_glWidget->showSelectionMessages();
        }
    }
    else
    {
        m_glWidget->setInteractionMode(GLWidget::TRANSFORM_CAMERA);

        if (m_glWidget->hasCloudLoaded()&&m_glWidget->showMessages())
        {
            m_glWidget->clearPolyline();
            m_glWidget->showMoveMessages();
        }
    }
}

void MainWindow::doActionDisplayShortcuts()
{
    QString text = "File menu:\n\n";
    text += "Ctrl+P: open .ply files\n";
    text += "Ctrl+O: open .xml camera files\n";
    text += "Ctrl+E: export mask files\n";
    text += "Ctrl+S: open .xml camera and export mask files\n";
    text += "Ctrl+Q: quit\n\n";
    text += "View:\n\n";
    text += "F2: full screen\n";
    text += "F3: show axis\n";
    text += "F4: show ball\n";
    text += "F5: show cameras\n";
    text += "F6: show help messages\n";
    text += "F7: move mode / selection mode\n";
    text += "\n";
    text += "Key +/-: increase/decrease point size\n\n";
    text += "Selection mode:\n\n";
    text += "    - Left click : add a point to polyline\n";
    text += "    - Right click: close polyline\n";
    text += "    - Echap: delete polyline\n";
    text += "    - Space bar: keep points inside polyline\n";
    text += "    - Shift+Space: add points inside polyline\n";
    text += "    - Del: keep points outside polyline\n";
    text += "    - Ctrl+Z: undo all past selections\n";  

    QMessageBox::information(NULL, "Saisie3D - shortcuts", text);
}

void MainWindow::connectActions()
{
    connect(m_glWidget,	SIGNAL(filesDropped(const QStringList&)), this,	SLOT(addFiles(const QStringList&)));

    connect(m_glWidget,	SIGNAL(mouseWheelRotated(float)),      this, SLOT(echoMouseWheelRotate(float)));

    //View menu
    connect(ui->actionFullScreen,       SIGNAL(toggled(bool)), this, SLOT(toggleFullScreen(bool)));
    connect(ui->actionShow_axis,        SIGNAL(toggled(bool)), this, SLOT(toggleShowAxis(bool)));
    connect(ui->actionShow_ball,        SIGNAL(toggled(bool)), this, SLOT(toggleShowBall(bool)));
    connect(ui->actionShow_cams,        SIGNAL(toggled(bool)), this, SLOT(toggleShowCams(bool)));
    connect(ui->actionShow_help_messages,SIGNAL(toggled(bool)), this, SLOT(toggleShowMessages(bool)));

    connect(ui->actionHelpShortcuts,    SIGNAL(triggered()),   this, SLOT(doActionDisplayShortcuts()));

    connect(ui->actionSetViewTop,		SIGNAL(triggered()),   this, SLOT(setTopView()));
    connect(ui->actionSetViewBottom,	SIGNAL(triggered()),   this, SLOT(setBottomView()));
    connect(ui->actionSetViewFront,		SIGNAL(triggered()),   this, SLOT(setFrontView()));
    connect(ui->actionSetViewBack,		SIGNAL(triggered()),   this, SLOT(setBackView()));
    connect(ui->actionSetViewLeft,		SIGNAL(triggered()),   this, SLOT(setLeftView()));
    connect(ui->actionSetViewRight,		SIGNAL(triggered()),   this, SLOT(setRightView()));

    //"Points selection" menu
    connect(ui->actionTogglePoints_selection, SIGNAL(toggled(bool)), this, SLOT(togglePointsSelection(bool)));
    connect(ui->actionAdd_points,       SIGNAL(triggered()),   this, SLOT(addPoints()));

    //File menu
    connect(ui->actionLoad_plys,		SIGNAL(triggered()),   this, SLOT(loadPlys()));
    connect(ui->actionLoad_camera,		SIGNAL(triggered()),   this, SLOT(loadCameras()));
    connect(ui->actionExport_mask,		SIGNAL(triggered()),   this, SLOT(exportMasks()));
    connect(ui->actionLoad_and_Export,	SIGNAL(triggered()),   this, SLOT(loadAndExport()));
    connect(ui->actionExit,             SIGNAL(triggered()),   this, SLOT(close()));
}

void MainWindow::addPoints()
{
    m_glWidget->segment(true, true);
}

void MainWindow::setTopView()
{
    m_glWidget->setView(TOP_VIEW);
}

void MainWindow::setBottomView()
{
    m_glWidget->setView(BOTTOM_VIEW);
}

void MainWindow::setFrontView()
{
    m_glWidget->setView(FRONT_VIEW);
}

void MainWindow::setBackView()
{
    m_glWidget->setView(BACK_VIEW);
}

void MainWindow::setLeftView()
{
    m_glWidget->setView(LEFT_VIEW);
}

void MainWindow::setRightView()
{
    m_glWidget->setView(RIGHT_VIEW);
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

void MainWindow::loadPlys()
{
    m_Engine->loadPlys();

    m_glWidget->setData(m_Engine->getData());

    checkForLoadedEntities();
}

void MainWindow::loadCameras()
{
    m_Engine->loadCameras();
}

void MainWindow::exportMasks()
{
    m_Engine->doMasks();
}

void MainWindow::loadAndExport()
{
    loadCameras();
    exportMasks();
}


