#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(bool mode2D, QWidget *parent) :
    QMainWindow(parent),
    _ui(new Ui::MainWindow),
    _Engine(new cEngine),
    _nbFen(QPoint(1,1)),
    _szFen(QPoint(800,600))
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

    on_actionShow_messages_toggled(_ui->actionShow_messages->isChecked());
    //on_actionShow_ball_toggled(_ui->actionShow_ball->isChecked());
    on_actionShow_axis_toggled(_ui->actionShow_axis->isChecked());
    on_actionShow_bbox_toggled(_ui->actionShow_bbox->isChecked());
    on_actionShow_cams_toggled(_ui->actionShow_cams->isChecked());

    setMode2D(mode2D);

    QGridLayout* layout = new QGridLayout();
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

    //File menu
    connect(_ui->actionClose_all, SIGNAL(triggered()), this, SLOT(closeAll()));
    connect(_ui->actionExit, SIGNAL(triggered()), this, SLOT(close()));

    //Zoom menu
    QSignalMapper* signalMapper = new QSignalMapper (this) ;

    connect(_ui->action4_1_400,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action2_1_200,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action1_1_100,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action1_2_50,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));
    connect(_ui->action1_4_25,		    SIGNAL(triggered()),   signalMapper, SLOT(map()));

    signalMapper->setMapping (_ui->action4_1_400, 400);
    signalMapper->setMapping (_ui->action2_1_200, 200);
    signalMapper->setMapping (_ui->action1_1_100, 100);
    signalMapper->setMapping (_ui->action1_2_50, 50);
    signalMapper->setMapping (_ui->action1_4_25, 25);

    connect (signalMapper, SIGNAL(mapped(int)), this, SLOT(zoomFactor(int)));

    //Selection
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
        on_actionShow_messages_toggled(_ui->actionShow_messages->isChecked());

    return loadedEntities;
}

void MainWindow::setPostFix(QString str)
{
   _Engine->setPostFix("_" + str);
}

void MainWindow::setNbFen(QPoint nb)
{
   _nbFen = nb;
}

void MainWindow::setSzFen(QPoint sz)
{
   _szFen = sz;
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
        printf("adding files %s", filenames[0].toStdString().c_str());
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

            _ui->actionShow_ball->setChecked(true);
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

void MainWindow::on_actionFullScreen_toggled(bool state)
{
    if (state)
        showFullScreen();
    else
        showNormal();
}

void MainWindow::on_actionShow_ball_toggled(bool state)
{
    if (!_bMode2D)
    {
        _glWidget->showBall(state);

        if (state)
        {
            _glWidget->showAxis(!state);
            _ui->actionShow_axis->setChecked(!state);
        }
    }
}

void MainWindow::on_actionShow_bbox_toggled(bool state)
{
    if(!_bMode2D)
        _glWidget->showBBox(state);
}

void MainWindow::on_actionShow_axis_toggled(bool state)
{
    if (!_bMode2D)
    {
        _glWidget->showAxis(state);

        if (state)
        {
            _glWidget->showBall(!state);
            _ui->actionShow_ball->setChecked(!state);
        }
    }
}

void MainWindow::on_actionShow_cams_toggled(bool state)
{
    if (!_bMode2D)
        _glWidget->showCams(state);
}

void MainWindow::on_actionShow_messages_toggled(bool state)
{
    _glWidget->showMessages(state);
}

void MainWindow::on_actionToggleMode_toggled(bool mode)
{
    if (!_bMode2D)
    {
        _glWidget->setInteractionMode(mode ? GLWidget::SELECTION : GLWidget::TRANSFORM_CAMERA);

        _glWidget->showBall(mode ? GLWidget::TRANSFORM_CAMERA : GLWidget::SELECTION && _Engine->getData()->isDataLoaded());
        _glWidget->showAxis(false);

        if (mode == GLWidget::SELECTION)
        {
            _glWidget->showCams(false);
            _glWidget->showBBox(false);
        }

        _glWidget->update();
    }
}

void MainWindow::on_actionHelpShortcuts_triggered()
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

void MainWindow::on_actionAdd_triggered()
{
    _glWidget->Select(ADD);
}

void MainWindow::on_actionSelect_none_triggered()
{
    _glWidget->Select(NONE);
    _glWidget->clearPolyline();
}

void MainWindow::on_actionInvertSelected_triggered()
{
    _glWidget->Select(INVERT);
}

void MainWindow::on_actionSelectAll_triggered()
{
    _glWidget->Select(ALL);
}

void MainWindow::on_actionReset_triggered()
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

void MainWindow::on_actionRemove_triggered()
{
    _glWidget->Select(SUB);
}

void MainWindow::on_actionSetViewTop_triggered()
{
    if (!_bMode2D)
        _glWidget->setView(TOP_VIEW);
}

void MainWindow::on_actionSetViewBottom_triggered()
{
    if (!_bMode2D)
        _glWidget->setView(BOTTOM_VIEW);
}

void MainWindow::on_actionSetViewFront_triggered()
{
    if (!_bMode2D)
        _glWidget->setView(FRONT_VIEW);
}

void MainWindow::on_actionSetViewBack_triggered()
{
    if (!_bMode2D)
        _glWidget->setView(BACK_VIEW);
}

void MainWindow::on_actionSetViewLeft_triggered()
{
    if (!_bMode2D)
        _glWidget->setView(LEFT_VIEW);
}

void MainWindow::on_actionSetViewRight_triggered()
{
    if (!_bMode2D)
        _glWidget->setView(RIGHT_VIEW);
}

void MainWindow::on_actionReset_view_triggered()
{
    _glWidget->resetView();

    if (!_bMode2D)
    {
         _glWidget->showBall(_Engine->getData()->isDataLoaded());
         _glWidget->showAxis(false);
         _glWidget->showBBox(false);
         _glWidget->showCams(false);
    }
}

//zoom
void MainWindow::on_actionZoom_Plus_triggered()
{
    _glWidget->setZoom(_glWidget->getParams()->m_zoom*1.5f);
}

void MainWindow::on_actionZoom_Moins_triggered()
{
    _glWidget->setZoom(_glWidget->getParams()->m_zoom/1.5f);
}

void MainWindow::on_actionZoom_fit_triggered()
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

void MainWindow::on_actionLoad_plys_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(NULL, tr("Open Cloud Files"),QString(), tr("Files (*.ply)"));

    if (!filenames.empty())
        addFiles(filenames);
}

void MainWindow::on_actionLoad_camera_triggered()
{
    QStringList filenames = QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"),QString(), tr("Files (*.xml)"));

    if (!filenames.empty())
        addFiles(filenames);
}

void MainWindow::on_actionLoad_image_triggered()
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


void MainWindow::on_actionSave_masks_triggered()
{
    if (_Engine->getData()->getNbImages())
    {
        _Engine->doMaskImage();
    }
    else
    {
        _Engine->doMasks();
    }
}

void MainWindow::on_actionSave_as_triggered()
{
    QString fname = QFileDialog::getSaveFileName(NULL, tr("Save mask Files"),QString(), tr("Files (*.*)"));

    if (!fname.isEmpty())
    {
        _Engine->setFilenameOut(fname);

        if (_Engine->getData()->getNbImages())
        {
            _Engine->doMaskImage();
        }
        else
        {
            _Engine->doMasks();
        }
    }
}

void MainWindow::on_actionSave_selection_triggered()
{
    _Engine->saveSelectInfos(_glWidget->getSelectInfos());
}

void MainWindow::closeAll()
{
    _Engine->unloadAll();

    _glWidget->reset();
    _glWidget->resetView();
    checkForLoadedData();
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
    _ui->actionShow_bbox->setVisible(!mBool);
    _ui->actionSave_selection->setVisible(!mBool);
    _ui->actionToggleMode->setVisible(!mBool);

    _ui->menuStandard_views->menuAction()->setVisible(!mBool);

    //pour activer/desactiver les raccourcis clavier

    _ui->actionLoad_plys->setEnabled(!mBool);
    _ui->actionLoad_camera->setEnabled(!mBool);
    _ui->actionShow_cams->setEnabled(!mBool);
    _ui->actionShow_axis->setEnabled(!mBool);
    _ui->actionShow_ball->setEnabled(!mBool);
    _ui->actionShow_bbox->setEnabled(!mBool);
    _ui->actionSave_selection->setEnabled(!mBool);
    _ui->actionToggleMode->setEnabled(!mBool);
}

void MainWindow::on_action2D_3D_mode_triggered()
{
    setMode2D(!_bMode2D);

    closeAll();
}

void  MainWindow::setGamma(float aGamma)
{
    _glWidget->getParams()->setGamma(aGamma);
}



