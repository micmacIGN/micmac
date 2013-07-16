#include <QMessageBox>

#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_Engine(new cEngine)
{
    ui->setupUi(this);

    QString style = "border: 2px solid gray;"
                    "border-radius: 1px;"
                    "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgb(%1,%2,%3), stop:1 rgb(%4,%5,%6));";

    style = style.arg(colorBG0.red()).arg(colorBG0.green()).arg(colorBG0.blue());
    style = style.arg(colorBG1.red()).arg(colorBG1.green()).arg(colorBG1.blue());

    ui->OpenglLayout->setStyleSheet(style);

    m_ProgressDialog = new QProgressDialog("Load ply files","Stop",0,0,this);
    //ProgressDialog->setMinimumDuration(500);
    m_ProgressDialog->setMinimum(0);
    m_ProgressDialog->setMaximum(100);

    connect(&m_FutureWatcher, SIGNAL(finished()),m_ProgressDialog,SLOT(cancel()));

    m_glWidget = new GLWidget(this,m_Engine->getData());

    toggleShowMessages(ui->actionShow_help_messages->isChecked());
    toggleShowBall(ui->actionShow_ball->isChecked());
    toggleShowAxis(ui->actionShow_axis->isChecked());
    toggleShowBBox(ui->actionShow_bounding_box->isChecked());

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(m_glWidget);
    connectActions();
    ui->OpenglLayout->setLayout(layout);

    createMenus();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_glWidget;
    delete m_Engine;
    delete m_RFMenu;
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
    connect(ui->actionTogglePoints_selection, SIGNAL(triggered(bool)), this, SLOT(togglePointsSelection(bool)));
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
    connect(ui->actionLoad_and_Export,  SIGNAL(triggered()),   this, SLOT(loadAndExport()));
    connect(ui->actionSave_selection,	SIGNAL(triggered()),   this, SLOT(saveSelectionInfos()));
    connect(ui->actionClose_all,        SIGNAL(triggered()),   this, SLOT(closeAll()));
    connect(ui->actionExit,             SIGNAL(triggered()),   this, SLOT(close()));

    connect(m_glWidget,SIGNAL(selectedPoint(uint,uint,bool)),this,SLOT(selectedPoint(uint,uint,bool)));

    for (int i = 0; i < MaxRecentFiles; ++i)
    {
        m_recentFileActs[i] = new QAction(this);
        m_recentFileActs[i]->setVisible(false);
        connect(m_recentFileActs[i], SIGNAL(triggered()),
                this, SLOT(openRecentFile()));
    }
}

void MainWindow::createMenus()
{
    m_RFMenu = new QMenu("Recent files", this);

    ui->menuFile->insertMenu(ui->actionExport_mask, m_RFMenu);
    ui->menuFile->insertSeparator(ui->actionExport_mask);

    for (int i = 0; i < MaxRecentFiles; ++i)
        m_RFMenu->addAction(m_recentFileActs[i]);

    updateRecentFileActions();
}

bool MainWindow::checkForLoadedData()
{
    bool loadedEntities = true;
    m_glWidget->displayNewMessage(QString()); //clear (any) message in the middle area

    if (!m_glWidget->hasDataLoaded())
    {
        m_glWidget->displayNewMessage("Drag & drop files on window to load them!");
        loadedEntities = false;
    }
    else
        toggleShowMessages(ui->actionShow_help_messages->isChecked());

    return loadedEntities;
}

void MainWindow::progression()
{
    if(m_incre)
        m_ProgressDialog->setValue(*m_incre);
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
        m_Engine->setFilename();

        #ifdef _DEBUG
            printf("adding files %s", filenames[0]);
        #endif

        if (fi.suffix() == "ply")
        {            

            QTimer *timer_test = new QTimer(this);
            m_incre = new int(0);
            connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            timer_test->start(10);
            QFuture<void> future = QtConcurrent::run(m_Engine, &cEngine::loadClouds,filenames,m_incre);

            this->m_FutureWatcher.setFuture(future);
            this->m_ProgressDialog->setWindowModality(Qt::WindowModal);
            this->m_ProgressDialog->exec();

            timer_test->stop();
            disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            delete m_incre;

            delete timer_test;

            future.waitForFinished();

            m_glWidget->setData(m_Engine->getData());
            m_glWidget->update();
        }
        else if (fi.suffix() == "xml")
        {          
            QFuture<void> future = QtConcurrent::run(m_Engine, &cEngine::loadCameras,filenames);

            m_glWidget->setCameraLoaded(true);
            m_glWidget->update();
        }

        for (int aK=0; aK< filenames.size();++aK) setCurrentFile(filenames[aK]);

        checkForLoadedData();
    }

    this->setWindowState(Qt::WindowActive);
}

void MainWindow::selectedPoint(uint idC, uint idV, bool select)
{
    m_Engine->getData()->getCloud(idC)->getVertex(idV).setVisible(select);
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
    text += "Ctrl+S: save .xml selection infos\n";
    text += "Ctrl+X: close files\n";
    text += "Ctrl+Q: quit\n\n";
    text += "View:\n\n";
    text += "F2: full screen\n";
    text += "F3: show axis\n";
    text += "F4: show ball\n";
    text += "F5: show bounding box\n";
    text += "F6: show cameras\n";
    text += "F7: show help messages\n";
    text += "\n";
    text += "Key +/-: increase/decrease point size\n\n";
    text += "Selection menu:\n\n";
    text += "F8: move mode / selection mode\n";
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
    m_FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Cloud Files"),QString(), tr("Files (*.ply)"));

    addFiles(m_FilenamesIn);
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
    m_Engine->loadCameras();
    m_Engine->doMasks();
}
void MainWindow::saveSelectionInfos()
{
    m_Engine->saveSelectInfos(m_glWidget->getSelectInfos());
}

void MainWindow::closeAll()
{
    m_Engine->unloadAll();

    m_glWidget->setCloudLoaded(false);
    m_glWidget->setCameraLoaded(false);
    checkForLoadedData();
    m_glWidget->setBufferGl();
    m_glWidget->update();
}

void MainWindow::openRecentFile()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if (action)
        addFiles(QStringList(action->data().toString()));
}

void MainWindow::setCurrentFile(const QString &fileName)
{
    m_curFile = fileName;
    setWindowFilePath(m_curFile);

    QSettings settings;
    QStringList files = settings.value("recentFileList").toStringList();
    QString fname = QDir::toNativeSeparators(fileName);
    files.removeAll(fname);
    files.prepend(fname);
    while (files.size() > MaxRecentFiles)
        files.removeLast();

    settings.setValue("recentFileList", files);

    foreach (QWidget *widget, QApplication::topLevelWidgets()) {
        MainWindow *mainWin = qobject_cast<MainWindow *>(widget);
        if (mainWin)
            mainWin->updateRecentFileActions();
    }
}

void MainWindow::updateRecentFileActions()
 {
     QSettings settings;
     QStringList files = settings.value("recentFileList").toStringList();

     int numRecentFiles = qMin(files.size(), (int)MaxRecentFiles);

     for (int i = 0; i < numRecentFiles; ++i) {
         QString text = tr("&%1 - %2").arg(i + 1).arg(strippedName(files[i]));
         m_recentFileActs[i]->setText(text);
         m_recentFileActs[i]->setData(files[i]);
         m_recentFileActs[i]->setVisible(true);
     }
     for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
         m_recentFileActs[j]->setVisible(false);

     //m_RFMenu->setVisible(numRecentFiles > 0);
 }

 QString MainWindow::strippedName(const QString &fullFileName)
 {
     return QFileInfo(fullFileName).fileName();
 }


