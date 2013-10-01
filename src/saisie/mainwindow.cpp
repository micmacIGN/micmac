#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(bool mode2D, QWidget *parent) :
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

    m_ProgressDialog = new QProgressDialog("Loading files","Stop",0,0,this);
    //ProgressDialog->setMinimumDuration(500);
    m_ProgressDialog->setMinimum(0);
    m_ProgressDialog->setMaximum(100);

    connect(&m_FutureWatcher, SIGNAL(finished()),m_ProgressDialog,SLOT(cancel()));

    m_glWidget = new GLWidget(this,m_Engine->getData());

    toggleShowMessages(ui->actionShow_help_messages->isChecked());
    toggleShowBall(ui->actionShow_ball->isChecked());
    toggleShowAxis(ui->actionShow_axis->isChecked());
    toggleShowBBox(ui->actionShow_bounding_box->isChecked());
    toggleShowCams(ui->actionShow_cams->isChecked());

    setMode2D(mode2D);

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


    //View menu
    connect(ui->actionFullScreen,       SIGNAL(toggled(bool)), this, SLOT(toggleFullScreen(bool)));
    if (!m_bMode2D)
    {
        connect(ui->actionShow_axis,        SIGNAL(toggled(bool)), this, SLOT(toggleShowAxis(bool)));
        connect(ui->actionShow_ball,        SIGNAL(toggled(bool)), this, SLOT(toggleShowBall(bool)));
        connect(ui->actionShow_cams,        SIGNAL(toggled(bool)), this, SLOT(toggleShowCams(bool)));
        connect(ui->actionShow_bounding_box,SIGNAL(toggled(bool)), this, SLOT(toggleShowBBox(bool)));
    }
    connect(ui->actionShow_help_messages,SIGNAL(toggled(bool)), this, SLOT(toggleShowMessages(bool)));

    connect(ui->actionHelpShortcuts,    SIGNAL(triggered()),   this, SLOT(doActionDisplayShortcuts()));

    if (!m_bMode2D)
    {
        connect(ui->actionSetViewTop,		SIGNAL(triggered()),   this, SLOT(setTopView()));
        connect(ui->actionSetViewBottom,	SIGNAL(triggered()),   this, SLOT(setBottomView()));
        connect(ui->actionSetViewFront,		SIGNAL(triggered()),   this, SLOT(setFrontView()));
        connect(ui->actionSetViewBack,		SIGNAL(triggered()),   this, SLOT(setBackView()));
        connect(ui->actionSetViewLeft,		SIGNAL(triggered()),   this, SLOT(setLeftView()));
        connect(ui->actionSetViewRight,		SIGNAL(triggered()),   this, SLOT(setRightView()));
    }

    connect(ui->actionZoom_Plus,		SIGNAL(triggered()),   this, SLOT(zoomPlus()));
    connect(ui->actionZoom_Moins,		SIGNAL(triggered()),   this, SLOT(zoomMoins()));
    connect(ui->actionZoom_fit,		    SIGNAL(triggered()),   this, SLOT(zoomFit()));

    QSignalMapper* signalMapper = new QSignalMapper (this) ;

    connect(ui->action4_1_400,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(ui->action2_1_200,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(ui->action1_1_100,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(ui->action1_2_50,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(ui->action1_4_25,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));

    signalMapper->setMapping (ui->action4_1_400, 400) ;
    signalMapper->setMapping (ui->action2_1_200, 200) ;
    signalMapper->setMapping (ui->action1_1_100, 100) ;
    signalMapper->setMapping (ui->action1_2_50, 50) ;
    signalMapper->setMapping (ui->action1_4_25, 25) ;

    connect (signalMapper, SIGNAL(mapped(int)), this, SLOT(zoomFactor(int))) ;

    //"Selection menu
    connect(ui->actionToggleMode_selection, SIGNAL(triggered(bool)), this, SLOT(toggleSelectionMode(bool)));
    connect(ui->actionAdd,              SIGNAL(triggered()),   this, SLOT(add()));
    connect(ui->actionSelect_none,      SIGNAL(triggered()),   this, SLOT(selectNone()));
    connect(ui->actionInvertSelected,   SIGNAL(triggered()),   this, SLOT(invertSelected()));
    connect(ui->actionSelectAll,        SIGNAL(triggered()),   this, SLOT(selectAll()));
    connect(ui->actionReset,            SIGNAL(triggered()),   this, SLOT(selectAll()));
    connect(ui->actionRemove_from_selection,            SIGNAL(triggered()),   this, SLOT(removeFromSelection()));

    //File menu
    connect(ui->actionLoad_plys,		SIGNAL(triggered()),   this, SLOT(loadPlys()));
    connect(ui->actionLoad_camera,		SIGNAL(triggered()),   this, SLOT(loadCameras()));
    connect(ui->actionLoad_image,		SIGNAL(triggered()),   this, SLOT(loadImage()));
    connect(ui->actionSave_masks,		SIGNAL(triggered()),   this, SLOT(exportMasks()));
    connect(ui->actionSave_as,          SIGNAL(triggered()),   this, SLOT(exportMasksAs()));
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
    m_RFMenu = new QMenu(tr("Recent files"), this);

    ui->menuFile->insertMenu(ui->actionSave_selection, m_RFMenu);
    ui->menuFile->insertSeparator(ui->actionSave_selection);

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
        m_glWidget->displayNewMessage(tr("Drag & drop files on window to load them!"));
        loadedEntities = false;
    }
    else
        toggleShowMessages(ui->actionShow_help_messages->isChecked());

    return loadedEntities;
}

void MainWindow::setPostFix(QString str)
{
   m_Engine->setPostFix("_" + str);
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
        for (int i=0; i< filenames.size();++i)
        {
            QFile Fout(filenames[i]);

            if(!Fout.exists())
            {
                QMessageBox::critical(this, "Error", "File or option does not exist");
                return;
            }
        }

        m_Engine->SetFilenamesIn(filenames);

        bool mode2D = getMode2D();

        if (mode2D)
        {
            setMode2D(false);
            closeAll();
        }

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

            m_Engine->setFilename();
            m_Engine->setFilenamesOut();
        }
        else if (fi.suffix() == "xml")
        {
            QFuture<void> future = QtConcurrent::run(m_Engine, &cEngine::loadCameras, filenames);

            this->m_FutureWatcher.setFuture(future);
            this->m_ProgressDialog->setWindowModality(Qt::WindowModal);
            this->m_ProgressDialog->exec();

            future.waitForFinished();

            m_glWidget->showCams(true);
            ui->actionShow_cams->setChecked(true);
        }
        else
        {
            if (!mode2D)
            {
                setMode2D(true);

                closeAll();
                glLoadIdentity();
            }

            m_Engine->loadImages(filenames);

            //try to load images
            /*QFuture<void> future = QtConcurrent::run(m_Engine, &cEngine::loadImages, filenames);

            this->m_FutureWatcher.setFuture(future);
            this->m_ProgressDialog->setWindowModality(Qt::WindowModal);
            this->m_ProgressDialog->exec();

            future.waitForFinished();*/

            m_Engine->setFilenamesOut();
        }

        m_glWidget->setData(m_Engine->getData());
        m_glWidget->update();

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

void MainWindow::toggleSelectionMode(bool state)
{
    if (state)
    {
        m_glWidget->setInteractionMode(GLWidget::SELECTION);

        if (m_glWidget->hasDataLoaded()&&m_glWidget->showMessages())
        {
            m_glWidget->showSelectionMessages();
        }
        m_glWidget->showBall(false);
        m_glWidget->showCams(false);
        m_glWidget->showAxis(false);
        m_glWidget->showBBox(false);
    }
    else
    {
        m_glWidget->setInteractionMode(GLWidget::TRANSFORM_CAMERA);

        if (m_glWidget->hasDataLoaded()&&m_glWidget->showMessages())
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
    QString text = tr("File menu:") +"\n\n";
    if (!m_bMode2D)
    {
        text += "Ctrl+P: \t" + tr("open .ply files")+"\n";
        text += "Ctrl+C: \t"+ tr("open .xml camera files")+"\n";
    }
    text += "Ctrl+O: \t"+tr("open image files")+"\n";
    if (!m_bMode2D) text += "tr(""Ctrl+E: \t"+tr("save .xml selection infos")+"\n";
    text += "Ctrl+S: \t"+tr("save masks files")+"\n";
    text += "Ctrl+Maj+S: \t"+tr("save masks files as")+"\n";
    text += "Ctrl+X: \t"+tr("close files")+"\n";
    text += "Ctrl+Q: \t"+tr("quit") +"\n\n";
    text += tr("View menu:") +"\n\n";
    text += "F2: \t"+tr("full screen") +"\n";
    if (!m_bMode2D)
    {
        text += "F3: \t"+tr("show axis") +"\n";
        text += "F4: \t"+tr("show ball") +"\n";
        text += "F5: \t"+tr("show bounding box") +"\n";
        text += "F6: \t"+tr("show cameras") +"\n";
    }

    if (!m_bMode2D)
        text += tr("Key +/-: \tincrease/decrease point size") +"\n\n";
    else
    {
        text += tr("Key +/-: \tincrease/decrease zoom") + "\n";
        text += "9: \t"+tr("zoom fit") + "\n";
        text+= "4: \tzoom 400%\n";
        text+= "2: \tzoom 200%\n";
        text+= "1: \tzoom 100%\n";
        text+= "Ctrl+2: \tzoom 50%\n";
        text+= "Ctrl+4: \tzoom 25%\n";
    }

    text += "F7: \t"+tr("show messages") +"\n\n";

    text += tr("Selection menu:") +"\n\n";
    text += "F8: \t"+tr("move mode / selection mode") +"\n\n";
    text += tr("    - Left click : \tadd a point to polyline") +"\n";
    text += tr("    - Right click: \tclose polyline") +"\n";
    text += tr("    - Echap: \t\tdelete polyline") +"\n";
    text += tr("    - Space bar: \tadd points/pixels inside polyline") +"\n";
    text += tr("    - Del: \t\tremove points/pixels inside polyline") +"\n";
    text += tr("    - Shift+click: \t\tinsert point in polyline") +"\n";
    text += tr("    - Drag n drop: \t\tmove polyline point") +"\n";
    text += tr("    - Right click: \t\tdelete polyline point") +"\n";
    text += "    - Ctrl+A: \t\t"+tr("select all") +"\n";
    text += "    - Ctrl+D: \t\t"+tr("select none") +"\n";
    text += "    - Ctrl+R: \t\t"+tr("undo all past selections") +"\n";
    text += "    - Ctrl+I: \t\t"+tr("invert selection") +"\n";

    QMessageBox::information(NULL, tr("Saisie - shortcuts"), text);
}

void MainWindow::add()
{
    m_glWidget->Select(ADD);
}

void MainWindow::selectNone()
{
    m_glWidget->Select(NONE);
    m_glWidget->clearPolyline();
}

void MainWindow::invertSelected()
{
    m_glWidget->Select(INVERT);
}

void MainWindow::selectAll()
{
    m_glWidget->Select(ALL);
}

void MainWindow::removeFromSelection()
{
    m_glWidget->Select(SUB);
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

//zoom
void MainWindow::zoomPlus()
{
    m_glWidget->setZoom(m_glWidget->getParams()->zoom*1.5f);
}

void MainWindow::zoomMoins()
{
    m_glWidget->setZoom(m_glWidget->getParams()->zoom/1.5f);
}

void MainWindow::zoomFit()
{
    m_glWidget->zoomFit();
}

void MainWindow::zoomFactor(int aFactor)
{
    m_glWidget->zoomFactor(aFactor);
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
    m_FilenamesIn = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"),QString(), tr("Files (*.xml)"));

    addFiles(m_FilenamesIn);
}

void MainWindow::loadImage()
{
    QString img_filename = QFileDialog::getOpenFileName(NULL, tr("Open Image File"),QString(), tr("File (*.*)"));

    m_FilenamesIn.clear();
    m_FilenamesIn.push_back(img_filename);

    if (!m_bMode2D)
    {
        m_bMode2D = true;

        closeAll();
        glLoadIdentity();
    }

    // load image (and mask)
    m_Engine->loadImage(img_filename);

    m_glWidget->setData(m_Engine->getData());
    m_glWidget->update();

    setCurrentFile(img_filename);

    checkForLoadedData();
}


void MainWindow::exportMasks()
{
    if (m_Engine->getData()->NbImages())
    {
        m_Engine->doMaskImage(m_glWidget->getGLMask());
    }
    else
    {
        m_Engine->doMasks();
    }
}

void MainWindow::exportMasksAs()
{
    m_Engine->setFilenameOut(QFileDialog::getSaveFileName(NULL, tr("Save mask Files"),QString(), tr("Files (*.*)")));

    if (m_Engine->getData()->NbImages())
    {
        m_Engine->doMaskImage(m_glWidget->getGLMask());
    }
    else
    {
        m_Engine->doMasks();
    }
}

void MainWindow::loadAndExport()
{
    loadCameras();
    m_Engine->doMasks();
}

void MainWindow::saveSelectionInfos()
{
    m_Engine->saveSelectInfos(m_glWidget->getSelectInfos());
}

void MainWindow::closeAll()
{
    m_Engine->unloadAll();

    m_glWidget->reset();

    checkForLoadedData();
    m_glWidget->setBufferGl();
    m_glWidget->update();
}

void MainWindow::openRecentFile()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if (action)
    {
        m_FilenamesIn = QStringList(action->data().toString());

        addFiles(m_FilenamesIn);
    }
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

    foreach (QWidget *widget, QApplication::topLevelWidgets())
    {
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

void MainWindow::setMode2D(bool mBool)
{
    m_bMode2D = mBool;

    ui->actionLoad_plys->setVisible(!mBool);
    ui->actionLoad_camera->setVisible(!mBool);
    ui->actionShow_cams->setVisible(!mBool);
    ui->actionShow_axis->setVisible(!mBool);
    ui->actionShow_ball->setVisible(!mBool);
    ui->actionShow_bounding_box->setVisible(!mBool);

    ui->menuStandard_views->menuAction()->setVisible(!mBool);

    //pour activer/desactiver les raccourcis clavier

    ui->actionLoad_plys->setEnabled(!mBool);
    ui->actionLoad_camera->setEnabled(!mBool);
    ui->actionShow_cams->setEnabled(!mBool);
    ui->actionShow_axis->setEnabled(!mBool);
    ui->actionShow_ball->setEnabled(!mBool);
    ui->actionShow_bounding_box->setEnabled(!mBool);
}

void  MainWindow::setGamma(float aGamma)
{
    m_glWidget->getParams()->setGamma(aGamma);
}



