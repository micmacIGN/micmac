#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(Pt2di aSzW, Pt2di aNbFen, int mode, QString pointName, QWidget *parent) :
        QMainWindow(parent),
        GLWidgetSet(aNbFen.x*aNbFen.y,colorBG0,colorBG1, mode > MASK3D),
        _ui(new Ui::MainWindow),
        _Engine(new cEngine),
        _mode(mode),
        _layout(new QGridLayout),
        _zoomLayout(new QGridLayout),
        _ptName(pointName)
{
    _ui->setupUi(this);

    QString style = "border: 1px solid #707070;"
            "border-radius: 0px;"
            "padding: 0px;"
            "margin: 0px;"
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgb(%1,%2,%3), stop:1 rgb(%4,%5,%6));";

    uint sy = 2;

    _layout->setContentsMargins(sy,sy,sy,sy);
    _layout->setHorizontalSpacing(sy);
    _layout->setVerticalSpacing(sy);

    style = style.arg(colorBorder.red()).arg(colorBorder.green()).arg(colorBorder.blue());
    style = style.arg(colorBorder.red()).arg(colorBorder.green()).arg(colorBorder.blue());

    _ui->OpenglLayout->setStyleSheet(style);
    _ui->OpenglLayout->setContentsMargins(0,0,0,0);

#ifdef ELISE_Darwin
    _ui->actionRemove->setShortcut(QKeySequence(Qt::ControlModifier+ Qt::Key_Y));
    _ui->actionAdd->setShortcut(QKeySequence(Qt::ControlModifier+ Qt::Key_U));
#endif

    _ProgressDialog = new QProgressDialog("Loading files","Stop",0,100,this);

    connect(&_FutureWatcher, SIGNAL(finished()),_ProgressDialog,SLOT(cancel()));

    _nbFen = QPoint(aNbFen.x,aNbFen.y);
    _szFen = QPoint(aSzW.x,aSzW.y);

    setMode();

    int cpt=0;
    for (int aK = 0; aK < aNbFen.x;++aK)
        for (int bK = 0; bK < aNbFen.y;++bK, cpt++)
            _layout->addWidget(getWidget(cpt), bK, aK);

    _signalMapper = new QSignalMapper (this);
    connectActions();
    _ui->OpenglLayout->setLayout(_layout);

    createRecentFileMenu();
}

MainWindow::~MainWindow()
{
    delete _ui;
    delete _Engine;
    delete _RFMenu;
    delete _layout;
    delete _zoomLayout;
    delete _signalMapper;
}

void MainWindow::connectActions()
{
    for (int aK = 0; aK < nbWidgets();++aK)
    {
        connect(getWidget(aK),	SIGNAL(filesDropped(const QStringList&)), this,	SLOT(addFiles(const QStringList&)));
        connect(getWidget(aK),	SIGNAL(overWidget(void*)), this,SLOT(changeCurrentWidget(void*)));
    }

    //File menu
    connect(_ui->actionClose_all, SIGNAL(triggered()), this, SLOT(closeAll()));

    for (int i = 0; i < MaxRecentFiles; ++i)
    {
        _recentFileActs[i] = new QAction(this);
        _recentFileActs[i]->setVisible(false);
        connect(_recentFileActs[i], SIGNAL(triggered()),
                this, SLOT(openRecentFile()));
    }

    //Zoom menu
    connect(_ui->action4_1_400,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_ui->action2_1_200,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_ui->action1_1_100,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_ui->action1_2_50,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_ui->action1_4_25,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));

    _signalMapper->setMapping (_ui->action4_1_400, 400);
    _signalMapper->setMapping (_ui->action2_1_200, 200);
    _signalMapper->setMapping (_ui->action1_1_100, 100);
    _signalMapper->setMapping (_ui->action1_2_50, 50);
    _signalMapper->setMapping (_ui->action1_4_25, 25);

    connect (_signalMapper, SIGNAL(mapped(int)), this, SLOT(zoomFactor(int)));
}

void MainWindow::createRecentFileMenu()
{
    _RFMenu = new QMenu(tr("&Recent files"), this);

    _ui->menuFile->insertMenu(_ui->actionSave_selection, _RFMenu);
    _ui->menuFile->insertSeparator(_ui->actionSave_selection);

    for (int i = 0; i < MaxRecentFiles; ++i)
        _RFMenu->addAction(_recentFileActs[i]);

    updateRecentFileActions();
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

void MainWindow::runProgressDialog(QFuture<void> future)
{
    _FutureWatcher.setFuture(future);
    _ProgressDialog->setWindowModality(Qt::WindowModal);
    _ProgressDialog->exec();

    future.waitForFinished();
}

void MainWindow::addFiles(const QStringList& filenames)
{
    if (filenames.size())
    {
        for (int i=0; i< filenames.size();++i)
        {
            if(!QFile(filenames[i]).exists())
            {
                QMessageBox::critical(this, "Error", "File does not exist (or bad argument)");
                return;
            }
        }

        _Engine->setFilenamesAndDir(filenames);

        QFileInfo fi(filenames[0]);

        if (fi.suffix() == "ply")
        {
            QTimer *timer_test = new QTimer(this);
            _incre = new int(0);
            connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            timer_test->start(10);

            runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadClouds,filenames,_incre));

            timer_test->stop();
            disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            delete _incre;
            delete timer_test;

            _mode = MASK3D;
        }
        else if (fi.suffix() == "xml")
        {
            runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadCameras, filenames));

            _ui->actionShow_cams->setChecked(true);

            _mode = MASK3D;
        }
        else // LOAD IMAGE
        {
            if (_mode <= MASK3D) closeAll();
            if (filenames.size() == 1) _mode = MASK2D;

            _Engine->loadImages(filenames);
        }

        _Engine->allocAndSetGLData(_mode > MASK3D, _ptName);

        for (int aK = 0; aK < nbWidgets();++aK)
        {
            getWidget(aK)->setGLData(_Engine->getGLData(aK),_ui->actionShow_messages);
            if (aK < filenames.size()) getWidget(aK)->getHistoryManager()->setFilename(_Engine->getFilenamesIn()[aK]);
        }

        for (int aK=0; aK < filenames.size();++aK) setCurrentFile(filenames[aK]);
    }
}

void MainWindow::on_actionFullScreen_toggled(bool state)
{   
    return state ? showFullScreen() : showNormal();
}

void MainWindow::on_actionShow_ball_toggled(bool state)
{
    if (_mode == MASK3D)
    {
        currentWidget()->setOption(cGLData::OpShow_Ball,state);

        if (state && _ui->actionShow_axis->isChecked())
        {
            currentWidget()->setOption(cGLData::OpShow_BBox,!state);
            _ui->actionShow_axis->setChecked(!state);
        }
    }
}

void MainWindow::on_actionShow_bbox_toggled(bool state)
{
    if (_mode == MASK3D)
        currentWidget()->setOption(cGLData::OpShow_BBox,state);
}

void MainWindow::on_actionShow_axis_toggled(bool state)
{
    if (_mode == MASK3D)
    {
        currentWidget()->setOption(cGLData::OpShow_Axis,state);

        if (state && _ui->actionShow_ball->isChecked())
        {
            currentWidget()->setOption(cGLData::OpShow_Ball,!state);
            _ui->actionShow_ball->setChecked(!state);
        }
    }
}

void MainWindow::on_actionShow_cams_toggled(bool state)
{
    if (_mode == MASK3D)
        currentWidget()->setOption(cGLData::OpShow_Cams,state);
}

void MainWindow::on_actionShow_messages_toggled(bool state)
{
    currentWidget()->setOption(cGLData::OpShow_Mess,state);
}

void MainWindow::on_actionToggleMode_toggled(bool mode)
{
    if (_mode == MASK3D)
        currentWidget()->setInteractionMode(mode ? SELECTION : TRANSFORM_CAMERA,_ui->actionShow_messages->isChecked());
}

void MainWindow::on_actionHelpShortcuts_triggered()
{
    QString text = tr("File menu:") +"\n\n";
    if (_mode == MASK3D)
    {
        text += "Ctrl+P: \t" + tr("open .ply files")+"\n";
        text += "Ctrl+C: \t"+ tr("open .xml camera files")+"\n";
    }
    text += "Ctrl+O: \t"+tr("open image file")+"\n";
    if (_mode == MASK3D) text += "tr(""Ctrl+E: \t"+tr("save .xml selection infos")+"\n";
    text += "Ctrl+S: \t"+tr("save mask file")+"\n";
    text += "Ctrl+Maj+S: \t"+tr("save mask file as")+"\n";
    text += "Ctrl+X: \t"+tr("close files")+"\n";
    text += "Ctrl+Q: \t"+tr("quit") +"\n\n";
    text += tr("View menu:") +"\n\n";
    text += "F2: \t"+tr("full screen") +"\n";
    if (_mode == MASK3D)
    {
        text += "F3: \t"+tr("show axis") +"\n";
        text += "F4: \t"+tr("show ball") +"\n";
        text += "F5: \t"+tr("show bounding box") +"\n";
        text += "F6: \t"+tr("show cameras") +"\n";
    }
    text += "F7: \t"+tr("show messages") +"\n";

    if (_mode == MASK3D)
        text += tr("Key +/-: \tincrease/decrease point size") +"\n\n";
    else
    {
        text += tr("Key +/-: \tzoom +/-") + "\n";
        text += "9: \t"+tr("zoom fit") + "\n";
        text+= "4: \tzoom 400%\n";
        text+= "2: \tzoom 200%\n";
        text+= "1: \tzoom 100%\n";
        text+= "Ctrl+2: \tzoom 50%\n";
        text+= "Ctrl+4: \tzoom 25%\n";
    }

    text += "Shift+R: \t"+tr("reset view") +"\n\n";

    if (_mode <= MASK3D)
    {
        text += tr("Selection menu:") +"\n\n";
        if (_mode == MASK3D)
        {
            text += "F9: \t"+tr("move mode / selection mode (only 3D)") +"\n\n";
        }
        text += tr("Left click : \tadd a vertex to polyline") +"\n";
        text += tr("Right click: \tclose polyline or delete nearest vertex") +"\n";
        text += tr("Echap: \tdelete polyline") +"\n";

#ifdef ELISE_Darwin
        if (_mode == MASK3D)
        {
            text += tr("Ctrl+Y: \tadd points inside polyline") +"\n";
            text += tr("Ctrl+U: \tremove points inside polyline") +"\n";
        }
        else
        {
            text += tr("Ctrl+Y: \tadd pixels inside polyline") +"\n";
            text += tr("Ctrl+U: \tremove pixels inside polyline") +"\n";
        }
#else
        if (_mode == MASK3D)
        {
            text += tr("Space bar: \tadd points inside polyline") +"\n";
            text += tr("Del: \tremove points inside polyline") +"\n";
        }
        else
        {
            text += tr("Space bar: \tadd pixels inside polyline") +"\n";
            text += tr("Del: \tremove pixels inside polyline") +"\n";
        }
#endif

        text += tr("Shift+drag: \tinsert vertex in polyline") +"\n";
        text += tr("Ctrl+right click: remove last vertex") +"\n";
        text += tr("Drag & drop: move selected polyline vertex") +"\n";
        text += "Ctrl+A: \t"+tr("select all") +"\n";
        text += "Ctrl+D: \t"+tr("select none") +"\n";
        text += "Ctrl+R: \t"+tr("reset") +"\n";
        text += "Ctrl+I: \t"+tr("invert selection") +"\n";
    }
    else
    {
        text += tr("Click: \tadd point")+"\n";
        text += tr("Right click: \tchange selected point state")+"\n";
        text += tr("Drag & drop: \tmove selected point") +"\n";
        text += tr("Shift+right click: \tshow name menu")+"\n";
        text += tr("Ctrl+right click: \tshow window menu")+"\n\n";

        text += tr("History menu:") +"\n\n";
    }
    text += "Ctrl+Z: \t"+tr("undo last action") +"\n";
    text += "Ctrl+Shift+Z: "+tr("redo last action") +"\n";

    QMessageBox msgbox(QMessageBox::Information, tr("Saisie - shortcuts"),text);
    msgbox.setWindowFlags(msgbox.windowFlags() | Qt::WindowStaysOnTopHint);
    msgbox.exec();
}

void MainWindow::on_actionAbout_triggered()
{
    QFont font("Courier New", 9, QFont::Normal);

    QMessageBox msgbox(QMessageBox::NoIcon, tr("Saisie"),QString(getBanniereMM3D().c_str()));
    msgbox.setFont(font);

    //trick to enlarge QMessageBox...
    QSpacerItem* horizontalSpacer = new QSpacerItem(600, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);
    QGridLayout* layout = (QGridLayout*)msgbox.layout();
    layout->addItem(horizontalSpacer, layout->rowCount(), 0, 1, layout->columnCount());

    msgbox.exec();
}

void MainWindow::on_actionAdd_triggered()
{
    currentWidget()->Select(ADD);
}

void MainWindow::on_actionSelect_none_triggered()
{
    currentWidget()->Select(NONE);
}

void MainWindow::on_actionInvertSelected_triggered()
{
    currentWidget()->Select(INVERT);
}

void MainWindow::on_actionSelectAll_triggered()
{
    currentWidget()->Select(ALL);
}

void MainWindow::on_actionReset_triggered()
{
    if (_mode != MASK3D)
    {
        closeAll();

        addFiles(_Engine->getFilenamesIn());
    }
    else
    {
        currentWidget()->Select(ALL);
    }
}

void MainWindow::on_actionRemove_triggered()
{
    if (_mode > MASK3D)
        currentWidget()->polygon().removeSelectedPoint();
    else
        currentWidget()->Select(SUB);
}

void MainWindow::on_actionSetViewTop_triggered()
{
    if (_mode == MASK3D)
        currentWidget()->setView(TOP_VIEW);
}

void MainWindow::on_actionSetViewBottom_triggered()
{
    if (_mode == MASK3D)
        currentWidget()->setView(BOTTOM_VIEW);
}

void MainWindow::on_actionSetViewFront_triggered()
{
    if (_mode == MASK3D)
        currentWidget()->setView(FRONT_VIEW);
}

void MainWindow::on_actionSetViewBack_triggered()
{
    if (_mode == MASK3D)
        currentWidget()->setView(BACK_VIEW);
}

void MainWindow::on_actionSetViewLeft_triggered()
{
    if (_mode == MASK3D)
        currentWidget()->setView(LEFT_VIEW);
}

void MainWindow::on_actionSetViewRight_triggered()
{
    if (_mode == MASK3D)
        currentWidget()->setView(RIGHT_VIEW);
}

void MainWindow::on_actionReset_view_triggered()
{
    currentWidget()->resetView(true,true,true,true);
}

void MainWindow::on_actionZoom_Plus_triggered()
{
    currentWidget()->setZoom(currentWidget()->getZoom()*1.5f);
}

void MainWindow::on_actionZoom_Moins_triggered()
{
    currentWidget()->setZoom(currentWidget()->getZoom()/1.5f);
}

void MainWindow::on_actionZoom_fit_triggered()
{
    currentWidget()->zoomFit();
}

void MainWindow::zoomFactor(int aFactor)
{
    currentWidget()->zoomFactor(aFactor);
}

void MainWindow::on_actionLoad_plys_triggered()
{
    addFiles(QFileDialog::getOpenFileNames(NULL, tr("Open Cloud Files"),QString(), tr("Files (*.ply)")));
}

void MainWindow::on_actionLoad_camera_triggered()
{
    addFiles(QFileDialog::getOpenFileNames(NULL, tr("Open Camera Files"),QString(), tr("Files (*.xml)")));
}

void MainWindow::on_actionLoad_image_triggered()
{
    QString img_filename = QFileDialog::getOpenFileName(NULL, tr("Open Image File"),QString(), tr("File (*.*)"));

    if (!img_filename.isEmpty())
    {
        //TODO: factoriser
        QStringList & filenames = _Engine->getFilenamesIn();
        filenames.clear();
        filenames.push_back(img_filename);

        setCurrentFile(img_filename);

        addFiles(filenames);
    }
}

void MainWindow::on_actionSave_masks_triggered()
{
    _Engine->saveMask(currentWidgetIdx());
}

void MainWindow::on_actionSave_as_triggered()
{
    QString fname = QFileDialog::getSaveFileName(NULL, tr("Save mask Files"),QString(), tr("Files (*.*)"));

    if (!fname.isEmpty())
    {
        _Engine->setFilenameOut(fname);

        _Engine->saveMask(currentWidgetIdx());
    }
}

void MainWindow::on_actionSave_selection_triggered()
{
    currentWidget()->getHistoryManager()->save();
}

void MainWindow::closeAll()
{
    _Engine->unloadAll();

    for (int aK=0; aK < nbWidgets(); ++aK)
        getWidget(aK)->reset();

    if (zoomWidget() != NULL)
    {
        zoomWidget()->reset();
        zoomWidget()->setOption(cGLData::OpShow_Mess,false);
    }
}

void MainWindow::closeCurrentWidget()
{
    _Engine->unloadAll();
    //_Engine->unload(currentWidgetIdx());

    currentWidget()->reset();
}

void MainWindow::openRecentFile()
{
    // A TESTER en multi images

#if WINVER == 0x0601 
	QAction *action = dynamic_cast<QAction *>(sender());
#else 
	QAction *action = qobject_cast<QAction *>(sender());
#endif

    if (action)
    {
        _Engine->setFilenamesAndDir(QStringList(action->data().toString()));

        addFiles(_Engine->getFilenamesIn());
    }
}

void MainWindow::setCurrentFile(const QString &fileName)
{
    // Rafraichit le menu des fichiers récents
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
		#if WINVER == 0x0601 
			MainWindow *mainWin = dynamic_cast<MainWindow *>(widget);
		#else
			MainWindow *mainWin = qobject_cast<MainWindow *>(widget);
		#endif
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

void hideAction(QAction* action, bool show)
{
    action->setVisible(show);
    action->setEnabled(show);
}

void MainWindow::setMode()
{
    bool isMode3D = _mode == MASK3D;

    hideAction(_ui->actionLoad_plys,  isMode3D);
    hideAction(_ui->actionLoad_camera,isMode3D);
    hideAction(_ui->actionShow_cams,  isMode3D);
    hideAction(_ui->actionShow_axis,  isMode3D);
    hideAction(_ui->actionShow_ball,  isMode3D);
    hideAction(_ui->actionShow_bbox,  isMode3D);
    hideAction(_ui->actionShow_cams,  isMode3D);
    hideAction(_ui->actionToggleMode, isMode3D);

    _ui->menuStandard_views->menuAction()->setVisible(isMode3D);

    if (_mode > MASK3D)
    {
        resize(_szFen.x() + _ui->zoomLayout->width(), _szFen.y());


        QString style = "border: 2px solid #707070;"
                "border-radius: 0px;"
                "padding: 0px;"
                "margin: 0px;";

        _ui->zoomLayout->setStyleSheet(style);
        //zoom Window
        _zoomLayout->addWidget(zoomWidget());
        _zoomLayout->setContentsMargins(2,2,2,2);

        _ui->zoomLayout->setLayout(_zoomLayout);
        _ui->zoomLayout->setContentsMargins(0,0,0,0);

        //disable some actions
        hideAction(_ui->actionAdd, false);
        hideAction(_ui->actionSelect_none, false);
        hideAction(_ui->actionInvertSelected, false);
        hideAction(_ui->actionSelectAll, false);
        hideAction(_ui->actionReset, false);

        hideAction(_ui->actionRemove, true);

        _ui->menuSelection->setTitle(tr("H&istory"));
    }
    else
    {
        resize(_szFen.x(), _szFen.y());

        _ui->verticalLayout->removeWidget(_ui->zoomLayout);
        _ui->verticalLayout->removeItem(_ui->verticalSpacer);

        delete _ui->zoomLayout;
        delete _ui->verticalSpacer;
    }
}

void  MainWindow::setGamma(float aGamma)
{
    _Engine->setGamma(aGamma);
}

void MainWindow::changeCurrentWidget(void *cuWid)
{
    GLWidget* glW = (GLWidget*)cuWid;

    setCurrentWidget(glW);

    if (zoomWidget())
    {
        zoomWidget()->setGLData(glW->getGLData(),false,true,false,false);
        zoomWidget()->setZoom(3.f);
        zoomWidget()->setOption(cGLData::OpShow_Mess,false);
        connect((GLWidget*)cuWid, SIGNAL(newImagePosition(int, int)), zoomWidget(), SLOT(centerViewportOnImagePosition(int,int)));
    }
}

void MainWindow::undo(bool undo)
{
    if (currentWidget()->getHistoryManager()->size())
    {
        if (_mode != MASK3D)
        {
            int idx = currentWidgetIdx();

            _Engine->reloadImage(idx);

            currentWidget()->setGLData(_Engine->getGLData(idx),_ui->actionShow_messages);
        }

        undo ? currentWidget()->getHistoryManager()->undo() : currentWidget()->getHistoryManager()->redo();
        currentWidget()->applyInfos();
    }
}
