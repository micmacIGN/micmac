#include <QMessageBox>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_Engine(new cEngine)
{
    ui->setupUi(this);

    ProgressDialog = new QProgressDialog();
    connect(&this->FutureWatcher, SIGNAL(finished()), this, SLOT(slot_finished()));
    connect(&this->FutureWatcher, SIGNAL(finished()), this->ProgressDialog , SLOT(cancel()));

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
//            QFuture<void> future = QtConcurrent::run(m_Engine, &cEngine::loadClouds,filenames);

//            this->FutureWatcher.setFuture(future);
//            this->ProgressDialog->setMinimum(0);
//            this->ProgressDialog->setMaximum(0);
//            this->ProgressDialog->setWindowModality(Qt::WindowModal);
//            this->ProgressDialog->exec();

            m_glWidget->setData(m_Engine->getData());
            m_glWidget->update();
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

void MainWindow::selectedPoint(uint idC, uint idV, bool select)
{
    m_Engine->getData()->getCloud(idC)->getVertex(idV).setVisible(select);
}

void MainWindow::slot_finished()
{
    std::cout << "Finshed" << std::endl;
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

void MainWindow::toggleShowBBox(bool state)
{
    m_glWidget->showBBox(state);
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
        m_glWidget->showBall(false);
    }
    else
    {
        m_glWidget->setInteractionMode(GLWidget::TRANSFORM_CAMERA);

        if (m_glWidget->hasCloudLoaded()&&m_glWidget->showMessages())
        {
            m_glWidget->clearPolyline();
            m_glWidget->showMoveMessages();
        }
        m_glWidget->showBall(true);
    }

    m_glWidget->update();
}

void MainWindow::doActionDisplayShortcuts()
{
    QString text = "File menu:\n\n";
    text += "Ctrl+P: open .ply files\n";
    text += "Ctrl+O: open .xml camera files\n";
    text += "Ctrl+E: export mask files\n";
    text += "Ctrl+Maj+S: open .xml camera and export mask files\n";
    text += "Ctrl+S: save .xml selection stack\n";
    text += "Ctrl+X: unload clouds and cameras\n";
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
    text += "    - Space bar: add points inside polyline\n";
    text += "    - Del: delete points inside polyline\n";
    text += "    - . : delete closest point in polyline\n";
    text += "    - Ctrl+A: select all\n";
    text += "    - Ctrl+D: select none\n";
    text += "    - Ctrl+R: undo all past selections\n";
    text += "    - Ctrl+I: invert selection\n";

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
    connect(ui->actionShow_bounding_box,SIGNAL(toggled(bool)), this, SLOT(toggleShowBBox(bool)));
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
    connect(ui->actionSelect_none,      SIGNAL(triggered()),   this, SLOT(selectNone()));
    connect(ui->actionInvertSelected,   SIGNAL(triggered()),   this, SLOT(invertSelected()));
    connect(ui->actionSelectAll,        SIGNAL(triggered()),   this, SLOT(selectAll()));
    connect(ui->actionReset,            SIGNAL(triggered()),   this, SLOT(selectAll()));
    connect(ui->actionRemove_from_selection,            SIGNAL(triggered()),   this, SLOT(removeFromSelection()));

    connect(ui->actionDeletePolylinepoint,SIGNAL(triggered()),   this, SLOT(deletePolylinePoint()));

    //File menu
    connect(ui->actionLoad_plys,		SIGNAL(triggered()),   this, SLOT(loadPlys()));
    connect(ui->actionLoad_camera,		SIGNAL(triggered()),   this, SLOT(loadCameras()));
    connect(ui->actionExport_mask,		SIGNAL(triggered()),   this, SLOT(exportMasks()));
    connect(ui->actionLoad_and_Export,	SIGNAL(triggered()),   this, SLOT(loadAndExport()));
    connect(ui->actionSave_selection,	SIGNAL(triggered()),   this, SLOT(saveSelectionInfos()));
    connect(ui->actionUnload_all,       SIGNAL(triggered()),   this, SLOT(unloadAll()));
    connect(ui->actionExit,             SIGNAL(triggered()),   this, SLOT(close()));

    connect(m_glWidget,SIGNAL(selectedPoint(uint,uint,bool)),this,SLOT(selectedPoint(uint,uint,bool)));
}



void MainWindow::addPoints()
{
    m_glWidget->Select(ADD);
    m_glWidget->update();
}

void MainWindow::selectNone()
{
    m_glWidget->Select(NONE);
    m_glWidget->clearPolyline();
    m_glWidget->update();
}

void MainWindow::invertSelected()
{
    m_glWidget->Select(INVERT);
    m_glWidget->update();
}

void MainWindow::selectAll()
{
    m_glWidget->Select(ALL);
    m_glWidget->update();
}

void MainWindow::removeFromSelection()
{
    m_glWidget->Select(SUB);
    m_glWidget->update();
}

void MainWindow::deletePolylinePoint()
{
    m_glWidget->deletePolylinePoint();
    m_glWidget->update();
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

void MainWindow::saveSelectionInfos()
{
    m_Engine->saveSelectInfos("SelectionInfos.xml");
}

void MainWindow::unloadAll()
{
    m_Engine->unloadAll();
    m_glWidget->setCloudLoaded(false);
    m_glWidget->setCameraLoaded(false);
    m_glWidget->setBufferGl();
    m_glWidget->update();
}


