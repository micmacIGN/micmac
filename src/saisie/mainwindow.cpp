#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(bool mode2D, QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::MainWindow),
    _Engine(new cEngine)
{
    _ui->setupUi(this);

    QString style = "border: 2px solid gray;"
            "border-radius: 1px;"
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgb(%1,%2,%3), stop:1 rgb(%4,%5,%6));";

    style = style.arg(colorBG0.red()).arg(colorBG0.green()).arg(colorBG0.blue());
    style = style.arg(colorBG1.red()).arg(colorBG1.green()).arg(colorBG1.blue());

    _ui->OpenglLayout->setStyleSheet(style);

    _ProgressDialog = new QProgressDialog("Loading files","Stop",0,0,this);
    //ProgressDialog->setMinimumDuration(500);
    _ProgressDialog->setMinimum(0);
    _ProgressDialog->setMaximum(100);

    connect(&_FutureWatcher, SIGNAL(finished()),_ProgressDialog,SLOT(cancel()));

    _glWidget = new GLWidget(this,_Engine->getData());

    toggleShowMessages(_ui->actionShow_help_messages->isChecked());
    //toggleShowBall(_ui->actionShow_ball->isChecked());
    toggleShowAxis(_ui->actionShow_axis->isChecked());
    toggleShowBBox(_ui->actionShow_bounding_box->isChecked());
    toggleShowCams(_ui->actionShow_cams->isChecked());

    setMode2D(mode2D);

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(_glWidget);
    connectActions();
    _ui->OpenglLayout->setLayout(layout);

    createMenus();
}

MainWindow::~MainWindow()
{
    delete _ui;
    delete _glWidget;
    delete _Engine;
    delete _RFMenu;
}

void MainWindow::connectActions()
{
    connect(_glWidget,	SIGNAL(filesDropped(const QStringList&)), this,	SLOT(addFiles(const QStringList&)));

    connect(_glWidget,	SIGNAL(interactionMode(bool)), this,	SLOT(changeMode(bool)));

    //View menu
    connect(_ui->actionFullScreen,       SIGNAL(toggled(bool)), this, SLOT(toggleFullScreen(bool)));
    if (!_bMode2D)
    {
        connect(_ui->actionShow_axis,        SIGNAL(toggled(bool)), this, SLOT(toggleShowAxis(bool)));
        connect(_ui->actionShow_ball,        SIGNAL(toggled(bool)), this, SLOT(toggleShowBall(bool)));
        connect(_ui->actionShow_cams,        SIGNAL(toggled(bool)), this, SLOT(toggleShowCams(bool)));
        connect(_ui->actionShow_bounding_box,SIGNAL(toggled(bool)), this, SLOT(toggleShowBBox(bool)));
    }
    connect(_ui->actionShow_help_messages,   SIGNAL(toggled(bool)), this, SLOT(toggleShowMessages(bool)));

    connect(_ui->actionReset_view,           SIGNAL(triggered()),   this, SLOT(resetView()));
    connect(_ui->action2D_3D_mode,           SIGNAL(triggered()),   this, SLOT(toggle2D3D()));

    connect(_ui->actionHelpShortcuts,        SIGNAL(triggered()),   this, SLOT(displayShortcuts()));

    if (!_bMode2D)
    {
        connect(_ui->actionSetViewTop,		SIGNAL(triggered()),   this, SLOT(setTopView()));
        connect(_ui->actionSetViewBottom,	SIGNAL(triggered()),   this, SLOT(setBottomView()));
        connect(_ui->actionSetViewFront,	SIGNAL(triggered()),   this, SLOT(setFrontView()));
        connect(_ui->actionSetViewBack,		SIGNAL(triggered()),   this, SLOT(setBackView()));
        connect(_ui->actionSetViewLeft,		SIGNAL(triggered()),   this, SLOT(setLeftView()));
        connect(_ui->actionSetViewRight,	SIGNAL(triggered()),   this, SLOT(setRightView()));
    }

    connect(_ui->actionZoom_Plus,		SIGNAL(triggered()),   this, SLOT(zoomPlus()));
    connect(_ui->actionZoom_Moins,		SIGNAL(triggered()),   this, SLOT(zoomMoins()));
    connect(_ui->actionZoom_fit,		SIGNAL(triggered()),   this, SLOT(zoomFit()));

    QSignalMapper* signalMapper = new QSignalMapper (this) ;

    connect(_ui->action4_1_400,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action2_1_200,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action1_1_100,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action1_2_50,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action1_4_25,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));

    signalMapper->setMapping (_ui->action4_1_400, 400) ;
    signalMapper->setMapping (_ui->action2_1_200, 200) ;
    signalMapper->setMapping (_ui->action1_1_100, 100) ;
    signalMapper->setMapping (_ui->action1_2_50, 50) ;
    signalMapper->setMapping (_ui->action1_4_25, 25) ;

    connect (signalMapper, SIGNAL(mapped(int)), this, SLOT(zoomFactor(int))) ;

    //"Selection menu
    connect(_ui->actionToggleMode,       SIGNAL(toggled(bool)), this, SLOT(toggleSelectionMode(bool)));
    connect(_ui->actionAdd,              SIGNAL(triggered()),   this, SLOT(add()));
    connect(_ui->actionSelect_none,      SIGNAL(triggered()),   this, SLOT(selectNone()));
    connect(_ui->actionInvertSelected,   SIGNAL(triggered()),   this, SLOT(invertSelected()));
    connect(_ui->actionSelectAll,        SIGNAL(triggered()),   this, SLOT(selectAll()));
    connect(_ui->actionReset,            SIGNAL(triggered()),   this, SLOT(reset()));
    connect(_ui->actionRemove,           SIGNAL(triggered()),   this, SLOT(removeFromSelection()));

    //File menu
    connect(_ui->actionLoad_plys,		 SIGNAL(triggered()),   this, SLOT(loadPlys()));
    connect(_ui->actionLoad_camera,		 SIGNAL(triggered()),   this, SLOT(loadCameras()));
    connect(_ui->actionLoad_image,		 SIGNAL(triggered()),   this, SLOT(loadImage()));
    connect(_ui->actionSave_masks,		 SIGNAL(triggered()),   this, SLOT(exportMasks()));
    connect(_ui->actionSave_as,          SIGNAL(triggered()),   this, SLOT(exportMasksAs()));
    connect(_ui->actionSave_selection,	 SIGNAL(triggered()),   this, SLOT(saveSelectionInfos()));
    connect(_ui->actionClose_all,        SIGNAL(triggered()),   this, SLOT(closeAll()));
    connect(_ui->actionExit,             SIGNAL(triggered()),   this, SLOT(close()));

    connect(_glWidget,SIGNAL(selectedPoint(uint,uint,bool)),this,SLOT(selectedPoint(uint,uint,bool)));

    for (int i = 0; i < MaxRecentFiles; ++i)
    {
        _recentFileActs[i] = new QAction(this);
        _recentFileActs[i]->setVisible(false);
        connect(_recentFileActs[i], SIGNAL(triggered()),
                this, SLOT(openRecentFile()));
    }
}

void MainWindow::createMenus()
{
    _RFMenu = new QMenu(tr("Recent files"), this);

    _ui->menuFile->insertMenu(_ui->actionSave_selection, _RFMenu);
    _ui->menuFile->insertSeparator(_ui->actionSave_selection);

    for (int i = 0; i < MaxRecentFiles; ++i)
        _RFMenu->addAction(_recentFileActs[i]);

    updateRecentFileActions();
}

bool MainWindow::checkForLoadedData()
{
    bool loadedEntities = true;
    _glWidget->displayNewMessage(QString()); //clear (any) message in the middle area

    if (!_glWidget->hasDataLoaded())
    {
        _glWidget->displayNewMessage(tr("Drag & drop files on window to load them!"));
        loadedEntities = false;
    }
    else
        toggleShowMessages(_ui->actionShow_help_messages->isChecked());

    return loadedEntities;
}

void MainWindow::setPostFix(QString str)
{
   _Engine->setPostFix("_" + str);
}

void MainWindow::progression()
{
    if(_incre)
        _ProgressDialog->setValue(*_incre);
}

void MainWindow::addFiles(const QStringList& filenames)
{
    if (filenames.size())
    {
        _FilenamesIn = filenames;

        for (int i=0; i< filenames.size();++i)
        {
            QFile Fout(filenames[i]);

            if(!Fout.exists())
            {
                QMessageBox::critical(this, "Error", "File or option does not exist");
                return;
            }
        }

        _Engine->SetFilenamesIn(filenames);

        if (getMode2D() != false) closeAll();
        setMode2D(false);

        QFileInfo fi(filenames[0]);

        //set default working directory as first file subfolder
        QDir Dir = fi.dir();
        Dir.cdUp();
        _Engine->setDir(Dir);

#ifdef _DEBUG
        printf("adding files %s", filenames[0]);
#endif

        if (fi.suffix() == "ply")
        {            
            QTimer *timer_test = new QTimer(this);
            _incre = new int(0);
            connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            timer_test->start(10);
            QFuture<void> future = QtConcurrent::run(_Engine, &cEngine::loadClouds,filenames,_incre);

            this->_FutureWatcher.setFuture(future);
            this->_ProgressDialog->setWindowModality(Qt::WindowModal);
            this->_ProgressDialog->exec();

            timer_test->stop();
            disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            delete _incre;

            delete timer_test;

            future.waitForFinished();

            _Engine->setFilename();
            _Engine->setFilenamesOut();
        }
        else if (fi.suffix() == "xml")
        {
            QFuture<void> future = QtConcurrent::run(_Engine, &cEngine::loadCameras, filenames);

            this->_FutureWatcher.setFuture(future);
            this->_ProgressDialog->setWindowModality(Qt::WindowModal);
            this->_ProgressDialog->exec();

            future.waitForFinished();

            _glWidget->showCams(true);
            _ui->actionShow_cams->setChecked(true);
        }
        else
        {
            setMode2D(true);

            glLoadIdentity();

            _Engine->loadImages(filenames);

            //try to load images
            /*QFuture<void> future = QtConcurrent::run(m_Engine, &cEngine::loadImages, filenames);

            this->m_FutureWatcher.setFuture(future);
            this->m_ProgressDialog->setWindowModality(Qt::WindowModal);
            this->m_ProgressDialog->exec();

            future.waitForFinished();*/

            _Engine->setFilenamesOut();
        }

        _glWidget->setData(_Engine->getData());

        for (int aK=0; aK< filenames.size();++aK) setCurrentFile(filenames[aK]);

        checkForLoadedData();
    }

    this->setWindowState(Qt::WindowActive);
}

void MainWindow::selectedPoint(uint idC, uint idV, bool select)
{
    _Engine->getData()->getCloud(idC)->getVertex(idV).setVisible(select);
}

void MainWindow::changeMode(bool mode)
{
    if (mode == true) //mode interaction
    {
        _ui->actionShow_ball->setChecked(false);
        _ui->actionShow_axis->setChecked(false);
        _ui->actionShow_cams->setChecked(false);
        _ui->actionShow_bounding_box->setChecked(false);
    }
    else
    {
        _ui->actionShow_ball->setChecked(true);
        _ui->actionShow_axis->setChecked(false);
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
    _glWidget->showBall(state);

    if (state)
    {
        _glWidget->showAxis(!state);
        _ui->actionShow_axis->setChecked(!state);
    }
}

void MainWindow::toggleShowBBox(bool state)
{
    _glWidget->showBBox(state);
}

void MainWindow::toggleShowAxis(bool state)
{
    _glWidget->showAxis(state);

    if (state)
    {
        _glWidget->showBall(!state);
        _ui->actionShow_ball->setChecked(!state);
    }
}

void MainWindow::toggleShowCams(bool state)
{
    _glWidget->showCams(state);
}

void MainWindow::toggleShowMessages(bool state)
{
    _glWidget->showMessages(state);
}

void MainWindow::toggleSelectionMode(bool state)
{
    _glWidget->setInteractionMode(state ? GLWidget::SELECTION : GLWidget::TRANSFORM_CAMERA);

    _glWidget->update();
}

void MainWindow::displayShortcuts()
{
    QString text = tr("File menu:") +"\n\n";
    if (!_bMode2D)
    {
        text += "Ctrl+P: \t" + tr("open .ply files")+"\n";
        text += "Ctrl+C: \t"+ tr("open .xml camera files")+"\n";
    }
    text += "Ctrl+O: \t"+tr("open image file")+"\n";
    if (!_bMode2D) text += "tr(""Ctrl+E: \t"+tr("save .xml selection infos")+"\n";
    text += "Ctrl+S: \t"+tr("save mask file")+"\n";
    text += "Ctrl+Maj+S: \t"+tr("save mask file as")+"\n";
    text += "Ctrl+X: \t"+tr("close files")+"\n";
    text += "Ctrl+Q: \t"+tr("quit") +"\n\n";
    text += tr("View menu:") +"\n\n";
    text += "F2: \t"+tr("full screen") +"\n";
    if (!_bMode2D)
    {
        text += "F3: \t"+tr("show axis") +"\n";
        text += "F4: \t"+tr("show ball") +"\n";
        text += "F5: \t"+tr("show bounding box") +"\n";
        text += "F6: \t"+tr("show cameras") +"\n";
    }
    text += "F7: \t"+tr("show messages") +"\n";

    if (!_bMode2D)
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

    text += "Shift+R: \t"+tr("reset view") +"\n";
    text += "F8: \t"+tr("2D mode / 3D mode") +"\n\n";

    text += tr("Selection menu:") +"\n\n";
    if (!_bMode2D)
    {
        text += "F9: \t"+tr("move mode / selection mode") +"\n\n";
    }
    text += tr("Left click : \tadd a vertex to polyline") +"\n";
    text += tr("Right click: \tclose polyline") +"\n";
    text += tr("Echap: \tdelete polyline") +"\n";
    if (!_bMode2D)
    {
        text += tr("Space bar: \tadd points inside polyline") +"\n";
        text += tr("Del: \tremove points inside polyline") +"\n";
    }
    else
    {
        text += tr("Space bar: \tadd pixels inside polyline") +"\n";
        text += tr("Del: \tremove pixels inside polyline") +"\n";
    }
    text += tr("Shift+click: \tinsert vertex in polyline") +"\n";
    text += tr("Drag & drop: move polyline vertex") +"\n";
    text += tr("Right click: \tdelete polyline vertex") +"\n";
    text += "Ctrl+A: \t"+tr("select all") +"\n";
    text += "Ctrl+D: \t"+tr("select none") +"\n";
    text += "Ctrl+R: \t"+tr("undo all past selections") +"\n";
    text += "Ctrl+I: \t"+tr("invert selection") +"\n";

    QMessageBox::information(NULL, tr("Saisie - shortcuts"), text);
}

void MainWindow::add()
{
    _glWidget->Select(ADD);
}

void MainWindow::selectNone()
{
    _glWidget->Select(NONE);
    _glWidget->clearPolyline();
}

void MainWindow::invertSelected()
{
    _glWidget->Select(INVERT);
}

void MainWindow::selectAll()
{
    _glWidget->Select(ALL);
}

void MainWindow::removeFromSelection()
{
    _glWidget->Select(SUB);
}

void MainWindow::reset()
{
    if (getMode2D())
    {
        closeAll();

        addFiles(_FilenamesIn);
    }
    else
    {
        _glWidget->Select(ALL);
    }
}

void MainWindow::setTopView()
{
    _glWidget->setView(TOP_VIEW);
}

void MainWindow::setBottomView()
{
    _glWidget->setView(BOTTOM_VIEW);
}

void MainWindow::setFrontView()
{
    _glWidget->setView(FRONT_VIEW);
}

void MainWindow::setBackView()
{
    _glWidget->setView(BACK_VIEW);
}

void MainWindow::setLeftView()
{
    _glWidget->setView(LEFT_VIEW);
}

void MainWindow::setRightView()
{
    _glWidget->setView(RIGHT_VIEW);
}

void MainWindow::resetView()
{
    _glWidget->resetView();
}

//zoom
void MainWindow::zoomPlus()
{
    _glWidget->setZoom(_glWidget->getParams()->zoom*1.5f);
}

void MainWindow::zoomMoins()
{
    _glWidget->setZoom(_glWidget->getParams()->zoom/1.5f);
}

void MainWindow::zoomFit()
{
    _glWidget->zoomFit();
}

void MainWindow::zoomFactor(int aFactor)
{
    _glWidget->zoomFactor(aFactor);
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
    QStringList filenames = QFileDialog::getOpenFileNames(NULL, tr("Open Cloud Files"),QString(), tr("Files (*.ply)"));

    if (!filenames.empty())
        addFiles(filenames);
}

void MainWindow::loadCameras()
{
    QStringList filenames = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"),QString(), tr("Files (*.xml)"));

    if (!filenames.empty())
        addFiles(filenames);
}

void MainWindow::loadImage()
{
    QString img_filename = QFileDialog::getOpenFileName(NULL, tr("Open Image File"),QString(), tr("File (*.*)"));

    if (!img_filename.isEmpty())
    {
        _FilenamesIn.clear();
        _FilenamesIn.push_back(img_filename);

        if (!_bMode2D)
        {
            _bMode2D = true;

            closeAll();
            glLoadIdentity();
        }

        // load image (and mask)
        _Engine->loadImage(img_filename);

        _glWidget->setData(_Engine->getData());

        setCurrentFile(img_filename);

        checkForLoadedData();
    }
}


void MainWindow::exportMasks()
{
    if (_Engine->getData()->NbImages())
    {
        _Engine->doMaskImage(_glWidget->getGLMask());
    }
    else
    {
        _Engine->doMasks();
    }
}

void MainWindow::exportMasksAs()
{
    QString fname = QFileDialog::getSaveFileName(NULL, tr("Save mask Files"),QString(), tr("Files (*.*)"));

    if (!fname.isEmpty())
    {
        _Engine->setFilenameOut(fname);

        if (_Engine->getData()->NbImages())
        {
            _Engine->doMaskImage(_glWidget->getGLMask());
        }
        else
        {
            _Engine->doMasks();
        }
    }
}

void MainWindow::saveSelectionInfos()
{
    _Engine->saveSelectInfos(_glWidget->getSelectInfos());
}

void MainWindow::closeAll()
{
    _Engine->unloadAll();

    _glWidget->reset();

    checkForLoadedData();
    _glWidget->setBufferGl();
    _glWidget->update();
}

void MainWindow::openRecentFile()
{
    QAction *action = qobject_cast<QAction *>(sender());
    if (action)
    {
        _FilenamesIn = QStringList(action->data().toString());

        addFiles(_FilenamesIn);
    }
}

void MainWindow::setCurrentFile(const QString &fileName)
{
    _curFile = fileName;
    setWindowFilePath(_curFile);

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
        _recentFileActs[i]->setText(text);
        _recentFileActs[i]->setData(files[i]);
        _recentFileActs[i]->setVisible(true);
    }
    for (int j = numRecentFiles; j < MaxRecentFiles; ++j)
        _recentFileActs[j]->setVisible(false);

    //m_RFMenu->setVisible(numRecentFiles > 0);
}

QString MainWindow::strippedName(const QString &fullFileName)
{
    return QFileInfo(fullFileName).fileName();
}

void MainWindow::setMode2D(bool mBool)
{
    _bMode2D = mBool;

    _ui->actionLoad_plys->setVisible(!mBool);
    _ui->actionLoad_camera->setVisible(!mBool);
    _ui->actionShow_cams->setVisible(!mBool);
    _ui->actionShow_axis->setVisible(!mBool);
    _ui->actionShow_ball->setVisible(!mBool);
    _ui->actionShow_bounding_box->setVisible(!mBool);
    _ui->actionSave_selection->setVisible(!mBool);
    _ui->actionToggleMode->setVisible(!mBool);

    _ui->menuStandard_views->menuAction()->setVisible(!mBool);

    //pour activer/desactiver les raccourcis clavier

    _ui->actionLoad_plys->setEnabled(!mBool);
    _ui->actionLoad_camera->setEnabled(!mBool);
    _ui->actionShow_cams->setEnabled(!mBool);
    _ui->actionShow_axis->setEnabled(!mBool);
    _ui->actionShow_ball->setEnabled(!mBool);
    _ui->actionShow_bounding_box->setEnabled(!mBool);
    _ui->actionSave_selection->setEnabled(!mBool);
    _ui->actionToggleMode->setEnabled(!mBool);
}

void MainWindow::toggle2D3D()
{
    setMode2D(!_bMode2D);

    closeAll();
}

void  MainWindow::setGamma(float aGamma)
{
    _glWidget->getParams()->setGamma(aGamma);
}



