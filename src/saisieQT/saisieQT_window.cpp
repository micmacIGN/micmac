#include "saisieQT_window.h"
#include "ui_saisieQT_window.h"

void setStyleSheet(QApplication &app)
{
    QFile file(app.applicationDirPath() + "/../include/qt/style.qss");
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        Q_INIT_RESOURCE(icones);

        app.setStyleSheet(file.readAll());
        file.close();
    }
    else
        QMessageBox::critical(NULL, QObject::tr("Error"), QObject::tr("Can't find qss file"));
}

SaisieQtWindow::SaisieQtWindow(int mode, QWidget *parent) :
        QMainWindow(parent),
        _ui(new Ui::SaisieQtWindow),
        _Engine(new cEngine),
        _layout_GLwidgets(new QGridLayout),
        _zoomLayout(new QGridLayout),
        _params(new cParameters),
        _appMode(mode),
		_bSaved(false),
		_devIOCamera(NULL)
{
    #ifdef ELISE_Darwin
        setWindowFlags(Qt::WindowStaysOnTopHint);
    #endif

    _ui->setupUi(this);

    _params->read();

    _Engine->setParams(_params);

    init(_params, _appMode > MASK3D);

    setUI();

    connectActions();

    createRecentFileMenu();

    applyParams();

    if (_appMode != MASK3D)
    {
        setImagePosition(QPointF(-1.f,-1.f));
        setImageName("");
    }

    tableView_PG()->setContextMenuPolicy(Qt::CustomContextMenu);
    tableView_Images()->setContextMenuPolicy(Qt::CustomContextMenu);
    tableView_Objects()->setContextMenuPolicy(Qt::CustomContextMenu);

    tableView_PG()->setMouseTracking(true);
    tableView_Objects()->setMouseTracking(true);

    _helpDialog = new cHelpDlg(QApplication::applicationName() + tr(" shortcuts"), this);
}

SaisieQtWindow::~SaisieQtWindow()
{
    delete _ui;
    delete _Engine;
    if (_appMode <= MASK3D) delete _RFMenu;
    delete _layout_GLwidgets;
    delete _zoomLayout;
    delete _signalMapper;
    delete _params;
    delete _helpDialog;
}

void SaisieQtWindow::connectActions()
{
    _ProgressDialog = new QProgressDialog(tr("Loading files"), tr("Stop"),0,100,this, Qt::ToolTip);

    connect(&_FutureWatcher, SIGNAL(finished()),_ProgressDialog, SLOT(cancel()));

    connect(_ui->menuFile, SIGNAL(aboutToShow()),this, SLOT(on_menuFile_triggered()));

    for (int aK = 0; aK < nbWidgets();++aK)
    {
        connect(getWidget(aK),	SIGNAL(filesDropped(const QStringList&, bool)), this,	SLOT(addFiles(const QStringList&, bool)));
        connect(getWidget(aK),	SIGNAL(overWidget(void*)), this,SLOT(changeCurrentWidget(void*)));
        connect(getWidget(aK),	SIGNAL(maskEdited()), this,SLOT(resetSavedState()));
        connect(this, SIGNAL(selectPoint(QString)),getWidget(aK),SLOT(selectPoint(QString)));
    }

    //File menu
    if (_appMode <= MASK3D) connect(_ui->actionClose_all, SIGNAL(triggered()), this, SLOT(closeAll()));

    for (int i = 0; i < MaxRecentFiles; ++i)
    {
        _recentFileActs[i] = new QAction(this);
        _recentFileActs[i]->setVisible(false);
        connect(_recentFileActs[i], SIGNAL(triggered()), this, SLOT(openRecentFile()));
    }

    //Zoom menu
    _signalMapper = new QSignalMapper (this);

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

void SaisieQtWindow::createRecentFileMenu()
{
    if (_appMode <= MASK3D)
    {
        _RFMenu = new QMenu(tr("&Recent files"), this);

        _ui->menuFile->insertMenu(_ui->actionSettings, _RFMenu);

        for (int i = 0; i < MaxRecentFiles; ++i)
            _RFMenu->addAction(_recentFileActs[i]);

        updateRecentFileActions();
    }
}

void SaisieQtWindow::setPostFix(QString str)
{
    _params->setPostFix(str);

    _Engine->setPostFix();
}

QString SaisieQtWindow::getPostFix()
{
    return _params->getPostFix();
}

void SaisieQtWindow::progression()
{
    if(_incre)
        _ProgressDialog->setValue(*_incre);
}

void SaisieQtWindow::runProgressDialog(QFuture<void> future)
{
    bool bShowMsgs = _ui->actionShow_messages->isChecked();
    on_actionShow_messages_toggled(false);

    _FutureWatcher.setFuture(future);
    _ProgressDialog->setWindowModality(Qt::WindowModal);
    _ProgressDialog->setCancelButton(NULL);

    float szFactor = 1.f;
    if (_params->getFullScreen())
    {
        QRect screen = QApplication::desktop()->screenGeometry ( -1 );

        szFactor = (float) screen.width() / size().width();
    }

    int ax = pos().x() + (_ui->frame_GLWidgets->size().width() * szFactor - _ProgressDialog->size().width())/2;
    int ay = pos().y() + (_ui->frame_GLWidgets->size().height() * szFactor - _ProgressDialog->size().height())/2;

    _ProgressDialog->move(ax, ay);
    _ProgressDialog->exec();

    future.waitForFinished();
    on_actionShow_messages_toggled(bShowMsgs);
}

bool SaisieQtWindow::loadPly(const QStringList& filenames)
{
    QTimer *timer_test = new QTimer(this);
    _incre = new int(0);
    connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
    timer_test->start();

    runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadClouds,filenames,_incre));

    timer_test->stop();
    disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
    delete _incre;
    delete timer_test;

    return true;
}

bool SaisieQtWindow::loadImages(const QStringList& filenames)
{
    QTimer *timer_test = new QTimer(this);
    _incre = new int(0);
    connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
    timer_test->start();

    if (filenames.size() == 1) _ProgressDialog->setMaximum(0);
    runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadImages,filenames,_incre));

    timer_test->stop();
    disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
    delete _incre;
    delete timer_test;

    return true;
}

bool SaisieQtWindow::loadCameras(const QStringList& filenames)
{
	if(_devIOCamera == NULL || _Engine->Loader() == NULL)
		return false;

	_Engine->Loader()->setDevIOCamera(_devIOCamera);

	cLoader *tmp = _Engine->Loader();
    for (int i=0;i<filenames.size();++i)
    {
         if (!tmp->loadCamera(filenames[i]))
         {
             QMessageBox::critical(this, tr("Error"), tr("Bad file: ") + filenames[i]);
             return false;
         }
    }

    QTimer *timer_test = new QTimer(this);
    _incre = new int(0);
    connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
    timer_test->start();

    runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadCameras,filenames,_incre));

    timer_test->stop();
    disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
    delete _incre;
    delete timer_test;

    return true;
}

void SaisieQtWindow::addFiles(const QStringList& filenames, bool setGLData)
{
	if(threeDWidget())
	{
		init3DPreview(getEngine()->getData(),*params());
	}

    if (filenames.size())
    {
        for (int i=0; i< filenames.size();++i)
        {
            if(!QFile(filenames[i]).exists())
            {
                QMessageBox::critical(this, tr("Error"), tr("File does not exist (or bad argument)"));
                return;
            }
            else
            {
                QString sufx = QFileInfo(filenames[i]).suffix();

                bool formatIsSupported = false;
                QStringList slist = QStringList("cr2")<<"arw"<<"crw"<<"dng"<<"mrw"<<"nef"<<"orf"<<"pef"<<"raf"<<"x3f"<<"rw2"<<"tif"; //main formats supported by ImageMagick
                QList<QByteArray> list = QImageReader::supportedImageFormats(); //formats supported by QImage
                for (int aK=0; aK< list.size();++aK) slist.push_back(QString(list[aK]));
                if (slist.contains(sufx, Qt::CaseInsensitive))  formatIsSupported = true;

                if ((sufx != "ply") && (sufx != "xml") && (formatIsSupported == false))
                {
                    QMessageBox::critical(this, tr("Error"), tr("File format not supported: ") + sufx);
                    return;
                }
            }
        }

        bool loadOK = false;

        _Engine->setFilenames(filenames);

        QString suffix = QFileInfo(filenames[0]).suffix();  //TODO: separer la stringlist si differents types de fichiers

        if (suffix == "ply")
        {
            if ((_appMode == MASK2D) && (currentWidget()->hasDataLoaded()))
                closeAll();

            loadOK = loadPly(filenames);
            if (loadOK)
            {
                initData();

                currentWidget()->getHistoryManager()->load(_Engine->getSelectionFilenamesOut()[0]);

                _appMode = MASK3D;
            }
        }
        else if (suffix == "xml")
        {
            loadOK = loadCameras(filenames);

            if (loadOK)
            {
                _ui->actionShow_cams->setChecked(true);

                _appMode = MASK3D;
            }
        }
        else // LOAD IMAGE
        {
            if ((_appMode <= MASK3D) && (currentWidget()->hasDataLoaded()))
                closeAll();

            currentWidget()->getHistoryManager()->reset();

            initData(); //TODO: ne pas detruire les polygones dans le closeAll

            if ((filenames.size() == 1) && (_appMode == MASK3D)) _appMode = MASK2D;

            int maxTexture;

            glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexture);

            //_Engine->setGLMaxTextureSize(maxTexture/16); //!!!!!!!!!!!!!!!!!!!!!!!
            _Engine->setGLMaxTextureSize(maxTexture);

            loadOK = loadImages(filenames);
        }

        if (loadOK)
        {
            _Engine->allocAndSetGLData(_appMode, *_params);

			setNavigationType(_params->eNavigation());

            if (setGLData)
            {

                for (int aK = 0; aK < nbWidgets();++aK)
                {
					getWidget(aK)->setGLData(_Engine->getGLData(aK), _ui->actionShow_messages->isChecked(), _ui->actionShow_cams->isChecked(),true,true,_params->eNavigation());
                    getWidget(aK)->setParams(_params);

                    if (getWidget(aK)->getHistoryManager()->size())
                    {
                        getWidget(aK)->applyInfos();
                        getWidget(aK)->getMatrixManager()->resetViewPort();
                        //_bSaved = false;
                    }
                }
            }
            else
                emit imagesAdded(-4, false);

            for (int aK=0; aK < filenames.size();++aK) setCurrentFile(filenames[aK]);

            updateUI();

            _ui->actionClose_all->setEnabled(true);
        }
    }

}

void SaisieQtWindow::on_actionFullScreen_toggled(bool state)
{
    state ? showFullScreen() : showNormal();

    _params->setFullScreen(state);
    _params->setPosition(pos());
    _params->setSzFen(size()); //ambiguité entre size() et screen.size() => scale factor quand fullScreen
}

void SaisieQtWindow::on_actionShow_ball_toggled(bool state)
{
    if (_appMode == MASK3D)
    {
        currentWidget()->setOption(cGLData::OpShow_Ball,state);

        if (state && _ui->actionShow_axis->isChecked())
        {
            currentWidget()->setOption(cGLData::OpShow_BBox,!state);
            _ui->actionShow_axis->setChecked(!state);
        }
    }
}

void SaisieQtWindow::on_actionShow_bbox_toggled(bool state)
{
    if (_appMode == MASK3D)
	{
        currentWidget()->setOption(cGLData::OpShow_BBox,state);

		if(threeDWidget())
			currentWidget()->setOption(cGLData::OpShow_BBox,state);
	}
}

void SaisieQtWindow::on_actionShow_grid_toggled(bool state)
{
    if (_appMode == MASK3D)
	{
        currentWidget()->setOption(cGLData::OpShow_Grid,state);
	}
}

void SaisieQtWindow::on_actionShow_axis_toggled(bool state)
{
    if (_appMode == MASK3D)
    {
        currentWidget()->setOption(cGLData::OpShow_Axis,state);

        if (state && _ui->actionShow_ball->isChecked())
        {
            currentWidget()->setOption(cGLData::OpShow_Ball,!state);
            _ui->actionShow_ball->setChecked(!state);
        }
    }
}


void SaisieQtWindow::on_actionSwitch_axis_Y_Z_toggled(bool state)
{
    for (int aK = 0; aK < nbWidgets();++aK)
    {
        if (getWidget(aK)->hasDataLoaded())
        {
			QVector3D rotation(state ? -90.f : 0.f,0.f,0.f);
            getWidget(aK)->getGLData()->setRotation(rotation);
            getWidget(aK)->update();
        }
    }
}

void SaisieQtWindow::on_actionShow_cams_toggled(bool state)
{
    if (_appMode == MASK3D)
        currentWidget()->setOption(cGLData::OpShow_Cams,state);
}

void SaisieQtWindow::on_actionShow_messages_toggled(bool state)
{
    for (int aK = 0; aK < nbWidgets();++aK)
        getWidget(aK)->setOption(cGLData::OpShow_Mess,state);

    labelShowMode(state);
}

void SaisieQtWindow::on_actionShow_names_toggled(bool show)
{
    for (int aK = 0; aK < nbWidgets();++aK)
    {
        if (getWidget(aK)->hasDataLoaded())
        {
            getWidget(aK)->getGLData()->currentPolygon()->showNames(show);
            getWidget(aK)->update();
        }
    }
}

void SaisieQtWindow::on_actionShow_refuted_toggled(bool show)
{
    emit showRefuted( !show );
}

void SaisieQtWindow::on_actionToggleMode_toggled(bool mode)
{
    if (_appMode == MASK3D)
        currentWidget()->setInteractionMode(mode ? SELECTION : TRANSFORM_CAMERA,_ui->actionShow_messages->isChecked(), _ui->actionShow_cams->isChecked());
}

void fillStringList(QStringList & actions, int appMode)
{
    if ((appMode == MASK3D) || (appMode == MASK2D))
    {
        actions.push_back(QObject::tr("select inside polygon"));
        actions.push_back(QObject::tr("remove inside polygon"));
        actions.push_back(QObject::tr("select outside polygon"));
        actions.push_back(QObject::tr("remove outside polygon"));
    }
}

void SaisieQtWindow::on_actionHelpShortcuts_triggered()
{
    const QPoint global = qApp->desktop()->availableGeometry().center();
    _helpDialog->move(global.x() - _helpDialog->width() / 2, global.y() - _helpDialog->height() / 2);

    _helpDialog->show();

    QStringList shortcuts;
    QStringList actions;

    shortcuts.push_back(tr("File Menu"));
    actions.push_back("");

    #ifdef ELISE_Darwin
		QString Ctrl="Cmd+";
	#else
		QString Ctrl = "Ctrl+";
    #endif

    if (_appMode == MASK3D)
    {
        shortcuts.push_back(Ctrl + "P");
        actions.push_back(tr("open .ply files"));
        shortcuts.push_back(Ctrl + "C");
        actions.push_back(tr("open .xml camera files"));
        shortcuts.push_back(Ctrl + "O");
        actions.push_back(tr("open image file"));
        shortcuts.push_back(Ctrl + "+S");
        actions.push_back(tr("save mask"));
        shortcuts.push_back(Ctrl + "Maj+S");
        actions.push_back(tr("save file as"));
        shortcuts.push_back(Ctrl + "X");
        actions.push_back(tr("close files"));
    }
    shortcuts.push_back(Ctrl + "T");
    actions.push_back(tr("settings"));
    shortcuts.push_back(Ctrl + "Q");
    actions.push_back(tr("quit"));

    shortcuts.push_back("");
    actions.push_back("");

    shortcuts.push_back(tr("View Menu"));
    actions.push_back("");

    shortcuts.push_back("F2");
    actions.push_back(tr("full screen"));
    if (_appMode == MASK3D)
    {
        shortcuts.push_back("F3");
        actions.push_back(tr("show axis"));
        shortcuts.push_back("F4");
        actions.push_back(tr("show ball"));
        shortcuts.push_back("F5");
        actions.push_back(tr("show bounding box"));
        shortcuts.push_back("F6");
        actions.push_back(tr("show grid"));
        shortcuts.push_back("F7");
        actions.push_back(tr("show cameras"));
    }
    shortcuts.push_back("F8");
    actions.push_back(tr("show messages"));
    if (_appMode > MASK3D)
    {
         shortcuts.push_back(Ctrl + "N");
         actions.push_back(tr("show names"));
         shortcuts.push_back(Ctrl + "R");
         actions.push_back(tr("show refuted"));
    }

    if (_appMode == MASK3D)
    {
        shortcuts.push_back(tr("Key +/-"));
        actions.push_back(tr("increase/decrease point size"));
    }
    else
    {
        shortcuts.push_back(tr("Key +/-"));
        actions.push_back(tr("zoom +/-"));
        shortcuts.push_back(tr("Key 9"));
        actions.push_back(tr("zoom fit"));
        shortcuts.push_back("Key 4");
         actions.push_back(tr("zoom 400%"));
        shortcuts.push_back(tr("Key 2"));
        actions.push_back(tr("zoom 200%"));
        shortcuts.push_back(tr("Key 1"));
        actions.push_back(tr("zoom 100%"));
        shortcuts.push_back(Ctrl + "2");
        actions.push_back(tr("zoom 50%"));
        shortcuts.push_back(Ctrl + "4");
        actions.push_back(tr("zoom 25%"));
    }

    shortcuts.push_back("Shift+R");
    actions.push_back(tr("reset view"));

    shortcuts.push_back("");
    actions.push_back("");

    if (_appMode == MASK3D)
    {
		shortcuts.push_back(tr("Navigation 3D"));
		actions.push_back("");

		shortcuts.push_back(tr("camera rotate x and y"));
		actions.push_back("Left button \t+ move mouse");

		shortcuts.push_back(tr("camera rotate z"));
		actions.push_back("Right button \t+ move mouse (only ball navigation)");

		shortcuts.push_back(tr("Zoom"));
		actions.push_back("wheel or shift + middle button");

		shortcuts.push_back(tr("move"));
		actions.push_back("middle button + move mouse");

		shortcuts.push_back(tr("move on vertex"));
		actions.push_back("Double click on vertex");

		shortcuts.push_back(tr(""));
		actions.push_back("");

        shortcuts.push_back(tr("Selection Menu"));
        actions.push_back("");
    }
    else if (_appMode == MASK2D)
    {
        shortcuts.push_back(tr("Mask Edition Menu"));
        actions.push_back("");
    }

    if (_appMode <= MASK3D)
    {
        if (_appMode == MASK3D)
        {
            shortcuts.push_back("F9");
            actions.push_back(tr("move mode / selection mode (only 3D)"));
        }
        shortcuts.push_back(tr("Left click"));
        actions.push_back(tr("add a vertex to polygon"));
        shortcuts.push_back(tr("Right click"));
        actions.push_back(tr("close polygon or delete nearest vertex"));
        shortcuts.push_back(tr("Echap"));
        actions.push_back(tr("delete polygon"));

#ifdef ELISE_Darwin
    #if ELISE_QT_VERSION >= 5
            shortcuts.push_back("Cmd+U");
            shortcuts.push_back("Cmd+Y");
            shortcuts.push_back("Shift+U");
            shortcuts.push_back("Shift+Y");
            fillStringList(actions, _appMode);
    #else
            shortcuts.push_back(tr("Space bar"));
            shortcuts.push_back(tr("Del"));
            shortcuts.push_back(tr("Ctrl+Space bar"));
            shortcuts.push_back(tr("Ctrl+Del"));
            fillStringList(actions, _appMode);
    #endif
#else
        shortcuts.push_back(tr("Space bar"));
        shortcuts.push_back(tr("Del"));
        shortcuts.push_back(tr("Ctrl+Space bar"));
        shortcuts.push_back(tr("Ctrl+Del"));
        fillStringList(actions, _appMode);
#endif

        shortcuts.push_back(tr("Shift+drag"));
        actions.push_back(tr("insert vertex in polygon"));
        shortcuts.push_back(Ctrl + tr("right click"));
        actions.push_back(tr("remove last vertex"));
        shortcuts.push_back(tr("Drag & drop"));
        actions.push_back(tr("move selected polygon vertex"));
        shortcuts.push_back(tr("Arrow keys"));
        actions.push_back(tr("move selected vertex"));
        shortcuts.push_back(tr("Alt+arrow keys"));
        actions.push_back(tr("move selected vertex faster"));
        shortcuts.push_back(tr("Key W+drag"));
        actions.push_back(tr("move polygon"));
        shortcuts.push_back(Ctrl + "A");
        actions.push_back(tr("select all"));
        shortcuts.push_back(Ctrl + "D");
        actions.push_back(tr("select none"));
        shortcuts.push_back(Ctrl + "R");
        actions.push_back(tr("reset selection"));
        shortcuts.push_back(Ctrl + "I");
        actions.push_back(tr("invert selection"));
    }
    else
    {
        shortcuts.push_back(tr("Left click"));
        actions.push_back(tr("add point"));
        shortcuts.push_back(tr("Right click"));
        actions.push_back(tr("show state menu or window menu"));
        shortcuts.push_back(tr("Drag & drop"));
        actions.push_back(tr("move selected point"));
    }
    if (_appMode <= MASK3D) //TEMP: TODO corriger le undo Elise
    {
        shortcuts.push_back(Ctrl +"Z");
        actions.push_back(tr("undo last action"));
        shortcuts.push_back(Ctrl + "Shift+Z");
        actions.push_back(tr("redo last action"));
    }

    _helpDialog->populateTableView(shortcuts, actions);
}

void SaisieQtWindow::on_actionAbout_triggered()
{
    QFont font("Courier New", 9, QFont::Normal);

    QMessageBox *msgBox = new QMessageBox(this);

	QString qStr("getBanniereMM3D().c_str()");
    #if (ELISE_windows || (defined ELISE_Darwin))
        qStr.replace( "**", "  " );
    #endif

    qStr += "\nApplication\t"           + QApplication::applicationName() +
            tr("\nBuilt with\t\tQT ")   + QT_VERSION_STR + //QString::number(ELISE_QT_VERSION) +
			tr("\nRevision\t\t")        + QString(string("__HG_REV__").c_str()) + "\n";

    msgBox->setText(qStr);
    msgBox->setWindowTitle(QApplication::applicationName());
    msgBox->setFont(font);

    //trick to enlarge QMessageBox...
    #if (!ELISE_windows)
        QSpacerItem* horizontalSpacer = new QSpacerItem(600, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);
        QGridLayout* layout = (QGridLayout*)msgBox->layout();
        layout->addItem(horizontalSpacer, layout->rowCount(), 0, 1, layout->columnCount());
    #endif

    msgBox->setWindowModality(Qt::NonModal);
    msgBox->show();
}

void SaisieQtWindow::on_actionRule_toggled(bool check)
{
//    if(check)
//        qDebug() << "Rules";

//	  setEnabled(false);
//	  QString program = "mm3d";
//	  QStringList arguments;
//	  arguments << "vMalt";

//	  QProcess *myProcess = new QProcess(this);

//	  myProcess->waitForFinished(-1);
//	  myProcess->start(program, arguments);
	  //setEnabled(true);
}

void SaisieQtWindow::resizeEvent(QResizeEvent *)
{
    _params->setSzFen(size());
}

void SaisieQtWindow::moveEvent(QMoveEvent *)
{
	_params->setPosition(pos());
}

void SaisieQtWindow::setNavigationType(int val)
{
	if (_appMode == MASK3D)
	{
			on_actionShow_grid_toggled(val);
			_ui->actionShow_grid->setChecked(val);
	}
}

void SaisieQtWindow::on_actionAdd_inside_triggered()
{
    currentWidget()->Select(ADD_INSIDE);
}

void SaisieQtWindow::on_actionAdd_outside_triggered()
{
    currentWidget()->Select(ADD_OUTSIDE);
}

void SaisieQtWindow::on_actionSelect_none_triggered()
{
    currentWidget()->Select(NONE);
}

void SaisieQtWindow::on_actionInvertSelected_triggered()
{
    currentWidget()->Select(INVERT);
}

void SaisieQtWindow::on_actionSelectAll_triggered()
{
    currentWidget()->Select(ALL);
}

void SaisieQtWindow::on_actionReset_triggered()
{
    if (_appMode != MASK3D)
    {
        closeAll(false);
        initData();

        addFiles(_Engine->getFilenamesIn());
    }
    else
    {
        currentWidget()->Select(ALL);
    }
}

void SaisieQtWindow::on_actionRemove_inside_triggered()
{
    if (_appMode > MASK3D)
        currentWidget()->polygon()->removeSelectedPoint();  //TODO: actuellement on ne garde pas le point selectionné (ajouter une action)
    else
        currentWidget()->Select(SUB_INSIDE);
}

void SaisieQtWindow::on_actionRemove_outside_triggered()
{
    currentWidget()->Select(SUB_OUTSIDE);
}

void SaisieQtWindow::on_actionSetViewTop_triggered()
{
    if (_appMode == MASK3D)
        currentWidget()->setView(TOP_VIEW);
}

void SaisieQtWindow::on_actionSetViewBottom_triggered()
{
    if (_appMode == MASK3D)
        currentWidget()->setView(BOTTOM_VIEW);
}

void SaisieQtWindow::on_actionSetViewFront_triggered()
{
    if (_appMode == MASK3D)
        currentWidget()->setView(FRONT_VIEW);
}

void SaisieQtWindow::on_actionSetViewBack_triggered()
{
    if (_appMode == MASK3D)
        currentWidget()->setView(BACK_VIEW);
}

void SaisieQtWindow::on_actionSetViewLeft_triggered()
{
    if (_appMode == MASK3D)
        currentWidget()->setView(LEFT_VIEW);
}

void SaisieQtWindow::on_actionSetViewRight_triggered()
{
    if (_appMode == MASK3D)
        currentWidget()->setView(RIGHT_VIEW);
}

void SaisieQtWindow::on_actionReset_view_triggered()
{
    currentWidget()->resetView(true,true,true,true,true);
}

void SaisieQtWindow::on_actionZoom_Plus_triggered()
{
    currentWidget()->setZoom(currentWidget()->getZoom()*1.5f);
}

void SaisieQtWindow::on_actionZoom_Moins_triggered()
{
    currentWidget()->setZoom(currentWidget()->getZoom()/1.5f);
}

void SaisieQtWindow::on_actionZoom_fit_triggered()
{
    currentWidget()->zoomFit();
}

void SaisieQtWindow::zoomFactor(int aFactor)
{
    currentWidget()->zoomFactor(aFactor);
}

void SaisieQtWindow::on_actionLoad_plys_triggered()
{
    addFiles(QFileDialog::getOpenFileNames(this, tr("Open Cloud Files"),QString(), tr("Files (*.ply)")));
}

void SaisieQtWindow::on_actionLoad_camera_triggered()
{
    addFiles(QFileDialog::getOpenFileNames(this, tr("Open Camera Files"),QString(), tr("Files (*.xml)")));
}

void SaisieQtWindow::on_actionLoad_image_triggered()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image File"),QString(), tr("File (*.*)"));

    if (filename.size())
        addFiles( QStringList(filename) );
}

void SaisieQtWindow::on_actionSave_masks_triggered()
{
    if (_appMode == MASK2D)
        _Engine->saveMask(currentWidgetIdx(), currentWidget()->isFirstAction());
    else if(_appMode == MASK3D)
        currentWidget()->getHistoryManager()->save();
    _bSaved = true;
}

void SaisieQtWindow::on_actionSave_as_triggered()
{
    QString fname = QFileDialog::getSaveFileName(NULL, tr("Save mask Files"),QString(), tr("Files (*.*)"));

    if (!fname.isEmpty())
    {
        if (QFileInfo(fname).suffix().isEmpty()) fname += ".tif";

        _Engine->setFilenameOut(fname);

        on_actionSave_masks_triggered();
    }
}

void SaisieQtWindow::on_actionSettings_triggered()
{
    cSettingsDlg _settingsDialog(this, _params, _appMode);

    connect(&_settingsDialog, SIGNAL(prefixTextEdit(QString)), this, SLOT(setAutoName(QString)));

    for (int aK = 0; aK < nbWidgets();++aK)
    {
        connect(&_settingsDialog, SIGNAL(lineThicknessChanged(float)), getWidget(aK), SLOT(lineThicknessChanged(float)));
        connect(&_settingsDialog, SIGNAL(pointDiameterChanged(float)), getWidget(aK), SLOT(pointDiameterChanged(float)));
        connect(&_settingsDialog, SIGNAL(gammaChanged(float)),         getWidget(aK), SLOT(gammaChanged(float)));
        connect(&_settingsDialog, SIGNAL(forceGray(bool)),             getWidget(aK), SLOT(forceGray(bool)));
        connect(&_settingsDialog, SIGNAL(showMasks(bool)),             getWidget(aK), SLOT(showMasks(bool)));
        connect(&_settingsDialog, SIGNAL(selectionRadiusChanged(int)), getWidget(aK), SLOT(selectionRadiusChanged(int)));
        connect(&_settingsDialog, SIGNAL(shiftStepChanged(float)),     getWidget(aK), SLOT(shiftStepChanged(float)));
        connect(&_settingsDialog, SIGNAL(setCenterType(int)),          getWidget(aK), SLOT(setCenterType(int)));
		connect(&_settingsDialog, SIGNAL(setNavigationType(int)),      getWidget(aK), SLOT(setNavigationType(int)));
    }	

	connect(&_settingsDialog, SIGNAL(setNavigationType(int)), this, SLOT(setNavigationType(int)));

	if (threeDWidget())
	{
		connect(&_settingsDialog, SIGNAL(setNavigationType(int)), threeDWidget(), SLOT(setNavigationType(int)));
	}

	if (zoomWidget())
    {
        connect(&_settingsDialog, SIGNAL(zoomWindowChanged(float)), zoomWidget(), SLOT(setZoom(float)));
        //connect(zoomWidget(), SIGNAL(zoomChanged(float)), this, SLOT(setZoom(float)));
    }

    const QPoint global = qApp->desktop()->availableGeometry().center();
    _settingsDialog.move(global.x() - _settingsDialog.width() / 2, global.y() - _settingsDialog.height() / 2);

    //_settingsDialog.setFixedSize(uiSettings.size());
    _settingsDialog.exec();

    /*#if defined(Q_OS_SYMBIAN)
        _settingsDialog.showMaximized();
    #else
        _settingsDialog.show();
    #endif*/

    disconnect(&_settingsDialog, 0, 0, 0);
}

void hideAction(QAction* action, bool show)
{
    action->setVisible(show);
    action->setEnabled(show);
}

void SaisieQtWindow::updateSaveActions()
{
    if (currentWidget()->getHistoryManager()->sizeChanged())
    {
        hideAction(_ui->actionSave_masks, true);
        hideAction(_ui->actionSave_as, true);
    }
    else if ((_appMode == MASK3D) || (_appMode == MASK2D))
    {
        _ui->actionSave_masks->setVisible(true);
        _ui->actionSave_masks->setEnabled(false);

        _ui->actionSave_as->setVisible(true);
        _ui->actionSave_as->setEnabled(false);
    }
}

void SaisieQtWindow::on_menuFile_triggered()
{
    //mode saisieAppuisInit
    hideAction(_ui->actionSave_masks, false);
    hideAction(_ui->actionSave_as, false);

    updateSaveActions();
}

void SaisieQtWindow::closeAll(bool check)
{
    if (check)
    {
        int reply = checkBeforeClose();

        // 1 close without saving
        if (reply == 2) //cancel
            return;
        else if (reply == 0) // save
            on_actionSave_masks_triggered();
    }

    emit sCloseAll();

	const int nbWidg = nbWidgets();

	for (int idGLW = 0; idGLW < nbWidg; ++idGLW)

		getWidget(idGLW)->setGLData(NULL);

    _Engine->unloadAll();

    if (zoomWidget() != NULL)
    {
        zoomWidget()->reset();
        zoomWidget()->setOption(cGLData::OpShow_Mess,false);
    }

    setImageName("");

    _ui->actionClose_all->setDisabled(true);
}

void SaisieQtWindow::closeCurrentWidget()
{
    _Engine->unloadAll();
    //_Engine->unload(currentWidgetIdx());

    currentWidget()->reset();
}

void SaisieQtWindow::openRecentFile()
{
    // A TESTER en multi images

#if WINVER == 0x0601
    QAction *action = dynamic_cast<QAction *>(sender());
#else
    QAction *action = qobject_cast<QAction *>(sender());
#endif

    if (action)
    {
        addFiles(QStringList(action->data().toString()));
    }
}

void SaisieQtWindow::setCurrentFile(const QString &fileName)
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
			SaisieQtWindow *mainWin = dynamic_cast<SaisieQtWindow *>(widget);
		#else
			SaisieQtWindow *mainWin = qobject_cast<SaisieQtWindow *>(widget);
		#endif
		if (mainWin)
			mainWin->updateRecentFileActions();
	}

}

void SaisieQtWindow::updateRecentFileActions()
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

QString SaisieQtWindow::strippedName(const QString &fullFileName)
{
    return QFileInfo(fullFileName).fileName();
}

void SaisieQtWindow::setLayout(uint sy)
{
    _layout_GLwidgets->setContentsMargins(sy,sy,sy,sy);
    _layout_GLwidgets->setHorizontalSpacing(sy);
    _layout_GLwidgets->setVerticalSpacing(sy);
    _ui->QFrame_OpenglLayout->setLayout(_layout_GLwidgets);

    int cpt=0;
    for (int aK = 0; aK < _params->getNbFen().x();++aK)
        for (int bK = 0; bK < _params->getNbFen().y();++bK, cpt++)
            _layout_GLwidgets->addWidget(getWidget(cpt), bK, aK);
}

void SaisieQtWindow::updateUI()
{
    labelShowMode(true);

    bool isMode3D = _appMode == MASK3D;

    hideAction(_ui->actionLoad_plys,  isMode3D);
    hideAction(_ui->actionLoad_camera,isMode3D);
    hideAction(_ui->actionShow_cams,  isMode3D);
    hideAction(_ui->actionSwitch_axis_Y_Z,  isMode3D);
    hideAction(_ui->actionShow_axis,  isMode3D);
    hideAction(_ui->actionShow_ball,  isMode3D);
    hideAction(_ui->actionShow_bbox,  isMode3D);
    hideAction(_ui->actionShow_grid,  isMode3D);
    hideAction(_ui->actionShow_cams,  isMode3D);
    hideAction(_ui->actionToggleMode, isMode3D);

    bool isModeMask = _appMode == MASK3D || _appMode == MASK2D;
    hideAction(_ui->actionShow_names, !isModeMask);
    hideAction(_ui->actionShow_refuted, !isModeMask);

    //disable some actions
    hideAction(_ui->actionAdd_inside, isModeMask);
    hideAction(_ui->actionAdd_outside, isModeMask);
    hideAction(_ui->actionSelect_none, isModeMask);
    hideAction(_ui->actionInvertSelected, isModeMask);
    hideAction(_ui->actionSelectAll, isModeMask);
    hideAction(_ui->actionReset, isModeMask);

    hideAction(_ui->actionRemove_inside, isModeMask);
    hideAction(_ui->actionRemove_outside, isModeMask);

    _ui->menuStandard_views->menuAction()->setVisible(isMode3D);

    if (_appMode == MASK2D)
    {
        _ui->menuSelection->setTitle(tr("&Mask edition"));
        _ui->actionAdd_inside->setText(tr("Add inside to mask"));
        _ui->actionRemove_inside->setText(tr("Remove inside from mask"));
        _ui->actionAdd_outside->setText(tr("Add outside to mask"));
        _ui->actionRemove_outside->setText(tr("Remove outside from mask"));
        _ui->actionSave_masks->setText(tr("&Save mask"));
    }
    else if (_appMode == MASK3D)
        _ui->actionSave_masks->setText(tr("&Save selection info"));

    _ui->actionAdd_inside->setShortcut(Qt::Key_Space);
    _ui->actionRemove_inside->setShortcut(Qt::Key_Delete);
    _ui->actionAdd_outside->setShortcut(QKeySequence(Qt::ControlModifier +Qt::Key_Space));
    _ui->actionRemove_outside->setShortcut(QKeySequence(Qt::ControlModifier +Qt::Key_Delete));

    #ifdef ELISE_Darwin
    #if(ELISE_QT_VERSION >= 5)
        _ui->actionRemove_inside->setShortcut(QKeySequence(Qt::ControlModifier + Qt::Key_Y));
        _ui->actionAdd_inside->setShortcut(QKeySequence(Qt::ControlModifier + Qt::Key_U));
        _ui->actionRemove_inside->setShortcut(QKeySequence(Qt::ShiftModifier + Qt::Key_Y));
        _ui->actionAdd_inside->setShortcut(QKeySequence(Qt::ShiftModifier + Qt::Key_U));
    #endif
    #endif
}

void SaisieQtWindow::setUI()
{

    setLayout(0);

    updateUI();

    if (_appMode > MASK3D)
    {
        if (_appMode == POINT2D_INIT)          setWindowTitle("Micmac - SaisieAppuisInitQT");
        else if (_appMode == POINT2D_PREDIC)   setWindowTitle("Micmac - SaisieAppuisPredicQT");
        else if (_appMode == BASC)             setWindowTitle("Micmac - SaisieBascQT");

        hideAction(_ui->actionLoad_image, false);
        hideAction(_ui->actionSave_masks, false);
        hideAction(_ui->actionSave_as, false);
        hideAction(_ui->actionClose_all, false);

        //zoom Window
        _zoomLayout->addWidget(zoomWidget());
        _zoomLayout->setContentsMargins(2,2,2,2);
        _ui->QFrame_zoom->setLayout(_zoomLayout);
        _ui->QFrame_zoom->setContentsMargins(0,0,0,0);

         QGridLayout*            _tdLayout = new QGridLayout;

         _tdLayout->addWidget(threeDWidget());
         _tdLayout->setContentsMargins(2,2,2,2);
        _ui->frame_preview3D->setLayout(_tdLayout);
        _ui->frame_preview3D->setContentsMargins(0,0,0,0);
        //TEMP: undo ne marche pas du coté Elise (a voir avec Marc)
        hideAction(_ui->menuSelection->menuAction(), false);
        hideAction(_ui->actionUndo, false);
        hideAction(_ui->actionRedo, false);
        //_ui->menuSelection->setTitle(tr("H&istory"));
        //fin TEMP

        tableView_PG()->installEventFilter(this);
        tableView_Objects()->installEventFilter(this);

        _ui->splitter_Tools->setContentsMargins(2,0,0,0);
    }
    else
    {
		_ui->QFrame_zoom->close();
		_ui->splitter_Tools->close();
		_ui->tableView_Objects->close();
		_ui->tableView_PG->close();
		_ui->tableView_Images->close();
		_ui->frame_preview3D->close();

		//_ui->frame_preview3D->close();
		//_ui->splitter_Tools->hide();
    }

	/*if (_appMode != BASC)*/

	//_ui->tableView_Objects->hide();

	_ui->tableView_Objects->close();

    //TEMP:
	hideAction(_ui->menuTools->menuAction(), false);
}

bool SaisieQtWindow::eventFilter( QObject* object, QEvent* event )
{
    if( object == tableView_PG() )
    {
        QAbstractItemView* table    = (QAbstractItemView*)object;

        QItemSelectionModel* sModel = table->selectionModel();

        if(sModel)
        {
            QString pointName = sModel->currentIndex().data(Qt::DisplayRole).toString();

            if(event->type() == QEvent::KeyRelease )
            {
                QKeyEvent *keyEvent = static_cast<QKeyEvent *>(event);

                if (keyEvent->key() == Qt::Key_Delete)

                    emit removePoint(pointName); // we send point name, because point has not necessarily a widget index (point non saisi)

                else if (keyEvent->key() == Qt::Key_Up || keyEvent->key() == Qt::Key_Down)
                {
                    emit selectPoint(pointName);
                }
            }
        }
    }

    return false;
}

QTableView *SaisieQtWindow::tableView_PG(){return _ui->tableView_PG;}

QTableView *SaisieQtWindow::tableView_Images(){return _ui->tableView_Images;}

QTableView *SaisieQtWindow::tableView_Objects(){return _ui->tableView_Objects;}

void SaisieQtWindow::resizeTables()
{
    tableView_PG()->resizeColumnsToContents();
    tableView_PG()->resizeRowsToContents();
    tableView_PG()->horizontalHeader()->setStretchLastSection(true);

    tableView_Images()->resizeColumnsToContents();
    tableView_Images()->resizeRowsToContents();
    tableView_Images()->horizontalHeader()->setStretchLastSection(true);

    tableView_Objects()->resizeColumnsToContents();
    tableView_Objects()->resizeRowsToContents();
    tableView_Objects()->horizontalHeader()->setStretchLastSection(true);
}

void SaisieQtWindow::setModel(QAbstractItemModel *model_Pg, QAbstractItemModel *model_Images/*, QAbstractItemModel *model_Objects*/)
{
    tableView_PG()->setModel(model_Pg);
    tableView_Images()->setModel(model_Images);
   // tableView_Objects()->setModel(model_Objects);
}

void SaisieQtWindow::SelectPointAllWGL(QString pointName)
{
    emit selectPoint(pointName);
}

void SaisieQtWindow::SetDataToGLWidget(int idGLW, cGLData *glData)
{
    if (glData)
    {
        GLWidget * glW = getWidget(idGLW);
        glW->setGLData(glData, glData->stateOption(cGLData::OpShow_Mess), glData->stateOption(cGLData::OpShow_Cams));
        glW->setParams(getParams());
    }
}

void SaisieQtWindow::loadPlyIn3DPrev(const QStringList &filenames, cData *dataCache)
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

        QString suffix = QFileInfo(filenames[0]).suffix();

        if (suffix == "ply")
        {
            loadPly(filenames);
            threeDWidget()->getGLData()->clearClouds();
            dataCache->computeBBox(1);
			threeDWidget()->getGLData()->setData(dataCache,false, _params->getSceneCenterType());
            threeDWidget()->resetView(false,false,false,false,true);
            option3DPreview();
        }
    }
}

void SaisieQtWindow::setCurrentPolygonIndex(int idx)
{
    for (int aK = 0; aK < getEngine()->nbGLData(); ++aK)
    {
        _Engine->getGLData(aK)->setCurrentPolygonIndex(idx);
    }
}

void SaisieQtWindow::normalizeCurrentPolygon(bool nrm)
{
    for (int aK = 0; aK < getEngine()->nbGLData(); ++aK)
    {
        _Engine->getGLData(aK)->normalizeCurrentPolygon(nrm);
    }
}

void SaisieQtWindow::initData()
{
    if (_appMode == BOX2D)
    {
        _Engine->addObject(new cRectangle());
    }
    /*else if (_appMode == BASC)
    {
        _Engine->addObject(new cPolygon(2)); //line
        _Engine->addObject(new cPolygon(1)); //origin
        _Engine->addObject(new cPolygon(2)); //scale
    }*/
    else
    {
        _Engine->addObject(new cPolygon());
    }
}

void  SaisieQtWindow::setGamma(float aGamma)
{
    _params->setGamma(aGamma);
}

void SaisieQtWindow::resetSavedState()
{
    _bSaved = false;
    updateSaveActions();
}

void SaisieQtWindow::closeEvent(QCloseEvent *event)
{
    int reply = checkBeforeClose();

    if (reply == 2)
    {
        event->ignore();
        return;
    }
    else if (reply == 0)
        on_actionSave_masks_triggered();

    emit sgnClose();

    if (zoomWidget())
        _params->setZoomWindowValue(zoomWidget()->getZoom());

    _params->write();

    event->accept();

    QMainWindow::closeEvent(event);
}

void SaisieQtWindow::redraw(bool nbWidgetsChanged)
{
    if (nbWidgetsChanged)
    {
        delete _layout_GLwidgets;
        _layout_GLwidgets = new QGridLayout;

        int newWidgetNb = _params->getNbFen().x()*_params->getNbFen().y();
        int col =  _layout_GLwidgets->columnCount();
        int row =  _layout_GLwidgets->rowCount();

        if (col < _params->getNbFen().x() || row < _params->getNbFen().y())
        {
            widgetSetResize(newWidgetNb);

            int cpt = 0;
            for (; cpt < nbWidgets();++cpt)
                _layout_GLwidgets->removeWidget(getWidget(cpt));

            cpt = 0;
            for (int aK =0; aK < _params->getNbFen().x();++aK)
                for (int bK =0; bK < _params->getNbFen().y();++bK)
                {
                    _layout_GLwidgets->addWidget(getWidget(cpt), bK, aK);

                    if (cpt < _Engine->getData()->getNbImages())
						getWidget(cpt)->setGLData(_Engine->getGLData(cpt),_ui->actionShow_messages->isChecked(), _ui->actionShow_cams->isChecked(),true,true,params()->eNavigation());

                    cpt++;
                }
            _ui->QFrame_OpenglLayout->setLayout(_layout_GLwidgets);
        }
        else
        {
            //TODO
        }
    }
}

void SaisieQtWindow::setAutoName(QString val)
{
    emit setName(val);
}

void SaisieQtWindow::setImagePosition(QPointF pt)
{
    QString text(tr("Image position : "));
    //QString text(tr("Zoom x Scale factor : "));

    if (pt.x() >= 0.f && pt.y() >= 0.f)
    {
        GLWidget* glW = currentWidget();
        if(glW)
            if ( glW->hasDataLoaded() && !glW->getGLData()->is3D() && (glW->isPtInsideIm(pt)))
            {
				int imHeight = glW->getGLData()->glImageMasked()._m_image->height();

                //text = QString(text + QString::number(pt.x(),'f',1) + ", " + QString::number((imHeight - pt.y()),'f',1)+" px");
                text = QString(text + QString::number(pt.x(),'f',1) + ", " + QString::number((imHeight - pt.y()),'f',1)+" px");
            }
    }

    _ui->label_ImagePosition_1->setText(text);
    _ui->label_ImagePosition_2->setText(text);
}

void SaisieQtWindow::setImageName(QString name)
{
    _ui->label_ImageName->setText(QString(tr("Image name : ") + name));
}

void SaisieQtWindow::setZoom(float val)
{
    _params->setZoomWindowValue(val);
}

void SaisieQtWindow::changeCurrentWidget(void *cuWid)
{
    GLWidget* glW = (GLWidget*) cuWid;

    setCurrentWidget(glW);

    if (_appMode != MASK3D)
    {
        connect(glW, SIGNAL(newImagePosition(QPointF)), this, SLOT(setImagePosition(QPointF)));
        connect(glW, SIGNAL(gammaChangedSgnl(float)), this, SLOT(setGamma(float)));

        if (zoomWidget())
        {
            zoomWidget()->setGLData(glW->getGLData(),false,false,true,false);
            zoomWidget()->setZoom(_params->getZoomWindowValue());
            zoomWidget()->setOption(cGLData::OpShow_Mess,false);

            connect(glW, SIGNAL(newImagePosition(QPointF)), zoomWidget(), SLOT(centerViewportOnImagePosition(QPointF)));
        }

		glW->checkTiles();
    }

    if (_appMode > MASK3D)
    {
        if ( glW->hasDataLoaded() && !glW->getGLData()->isImgEmpty() )
			setImageName(glW->getGLData()->glImageMasked().cObjectGL::name());
    }
}

void SaisieQtWindow::undo(bool undo)
{
	// TODO seg fault dans le undo � cause de la destruction des images...

    if (_appMode <= MASK3D)
    {
        if (currentWidget()->getHistoryManager()->size())
        {

            if ((_appMode != MASK3D) && undo)
            {
                int idx = currentWidgetIdx();

				currentWidget()->setGLData(NULL, _ui->actionShow_messages->isChecked(), _ui->actionShow_cams->isChecked(),false,false);
                //_Engine->reloadImage(_appMode, idx);
                _Engine->reloadMask(_appMode, idx);

				currentWidget()->setGLData(_Engine->getGLData(idx), _ui->actionShow_messages->isChecked(), _ui->actionShow_cams->isChecked(), false,false);
            }



            undo ? currentWidget()->getHistoryManager()->undo() : currentWidget()->getHistoryManager()->redo();
            currentWidget()->applyInfos();
            _bSaved = false;
        }
    }
    else
    {
        emit undoSgnl(undo);
    }
}
deviceIOImage* SaisieQtWindow::devIOImage() const
{
	return _Engine->Loader()->devIOImageAlter();
}

void SaisieQtWindow::setDevIOImage(deviceIOImage* devIOImage)
{
	_Engine->Loader()->setDevIOImageAlter(devIOImage);
}

deviceIOCamera* SaisieQtWindow::devIOCamera() const
{
	return _devIOCamera;
}

void SaisieQtWindow::setDevIOCamera(deviceIOCamera* devIOCamera)
{
	_devIOCamera = devIOCamera;
}

cParameters *SaisieQtWindow::params() const
{
	return _params;
}

void SaisieQtWindow::setParams(cParameters *params)
{
	_params = params;
}

int SaisieQtWindow::appMode() const
{
    return _appMode;
}

void SaisieQtWindow::setAppMode(int appMode)
{
    _appMode = appMode;
}

QAction* SaisieQtWindow::addCommandTools(QString nameCommand)
{
    _ui->menuTools->setEnabled(true);

    return _ui->menuTools->addAction(nameCommand);
}

int SaisieQtWindow::checkBeforeClose()
{
    if ((!_bSaved) && (_appMode == MASK3D || _appMode == MASK2D) && currentWidget()->getHistoryManager()->sizeChanged() )
    {
        return QMessageBox::question(this, tr("Warning"), tr("Save before closing?"),tr("&Save"),tr("&Close without saving"),tr("Ca&ncel"));
        //TODO: highlight default button (stylesheet ?)
        //return QMessageBox::question(this, tr("Warning"), tr("Save before closing?"),QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
    }
    else return -1;
}

void SaisieQtWindow::applyParams()
{
    move(_params->getPosition());

    QSize szFen = _params->getSzFen();

    if (_params->getFullScreen())
    {
        showFullScreen();

        QRect screen = QApplication::desktop()->screenGeometry ( -1 );

        _params->setSzFen(screen.size());
        _params->setPosition(QPoint(0,0));

        _params->write();

        _ui->actionFullScreen->setChecked(true);
    }
    else if (_appMode > MASK3D)
        resize(szFen.width() + _ui->QFrame_zoom->width(), szFen.height());
    else
        resize(szFen);
}

void SaisieQtWindow::labelShowMode(bool state)
{
    if ((!state) || (_appMode == MASK3D))
    {
        _ui->label_ImagePosition_1->hide();
        _ui->label_ImagePosition_2->hide();
        _ui->label_ImageName->hide();
    }
    else
    {
        if(_appMode <= MASK2D)
        {
            _ui->label_ImagePosition_1->hide();
            _ui->label_ImagePosition_2->show();
            _ui->label_ImageName->hide();
        }
        else if(_appMode > MASK3D)
        {
            _ui->label_ImagePosition_1->show();
            _ui->label_ImagePosition_2->hide();
            _ui->label_ImageName->show();
        }
    }
}
