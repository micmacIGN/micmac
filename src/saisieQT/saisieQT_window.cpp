#include "saisieQT_window.h"
#include "ui_saisieQT_window.h"
#include "GlExtensions.h"

#ifdef USE_MIPMAP_HANDLER
	#include "StdAfx.h"
#endif

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
        _ProgressDialog(NULL),
        _layout_GLwidgets(new QGridLayout),
        _zoomLayout(new QGridLayout),
        _params(new cParameters),
        _appMode(mode),
        _bSaved(false),
        _devIOCamera(NULL),
        _git_revision("Unknown"),
        _banniere("No comment"),
        _workBench(NULL)
{
    /*#ifdef ELISE_Darwin
        setWindowFlags(Qt::WindowStaysOnTopHint);
    #endif*/

    _ui->setupUi(this);

    _params->read();

    _Engine->setParams(_params);

    init(_params, _appMode > MASK3D);

	#ifdef USE_MIPMAP_HANDLER
		for (int iWidget = 0; iWidget < nbWidgets(); iWidget++)
			getWidget(iWidget)->setParent(this);
	#endif

    setUI();

    connectActions();

    createRecentFileMenu();

    applyParams();

    if (_appMode != MASK3D)
    {
        setImagePosition(QPointF(-1.f,-1.f));
        setImageName("");
    }

    ObjectsSFModel*     proxyObjectModel = new ObjectsSFModel(this);
    proxyObjectModel->setSourceModel(new ModelObjects(0,currentWidget()->getHistoryManager()));
    setModelObject(proxyObjectModel);
    connect(currentWidget(),SIGNAL(changeHistory()),proxyObjectModel,SLOT(invalidate()));
    connect(tableView_Objects()->selectionModel(), SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),this,SLOT(selectionObjectChanged(QItemSelection,QItemSelection)));
    connect(tableView_Objects()->selectionModel()->model(), SIGNAL(dataChanged(QModelIndex,QModelIndex)),this,SLOT(updateMask()));

    tableView_Objects()->setItemDelegateForColumn(2,new ComboBoxDelegate(ModelObjects::getSelectionMode(),SIZE_OF_SELECTION_MODE));

    tableView_PG()->setContextMenuPolicy(Qt::CustomContextMenu);
    tableView_Images()->setContextMenuPolicy(Qt::CustomContextMenu);
//    tableView_Objects()->setContextMenuPolicy(Qt::CustomContextMenu);

    tableView_PG()->setMouseTracking(true);
    tableView_Objects()->setMouseTracking(true);

    _helpDialog = new cHelpDlg(QApplication::applicationName() + tr(" shortcuts"), this);

	#ifdef __QT_5_SHORTCUT_PATCH
		//~ shortcuts do not work under linux (bug or architecture pb)
		addAction(_ui->menuSelection->menuAction());
		addAction(_ui->menuFile->menuAction());
		addAction(_ui->menuView->menuAction());
		addAction(_ui->menuStandard_views->menuAction());
		addAction(_ui->menuZoom->menuAction());
		addAction(_ui->menuHelp->menuAction());
		addAction(_ui->menuTools->menuAction());
		addAction(_ui->menuWindows->menuAction());

		// some shortcuts names do not appear
	#endif

	CHECK_GL_ERROR("SaisieQtWindow::SaisieQtWindow");
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

ProgressDialogUpdateSignaler::ProgressDialogUpdateSignaler(QProgressDialog &aProgressDialog):
	_progressDialog(aProgressDialog)
{
}

void ProgressDialogUpdateSignaler::operator ()()
{
	_progressDialog.setValue(_progressDialog.value() + 1);
}

void SaisieQtWindow::activateLoadImageProgressDialog(int aMin, int aMax)
{
	if (_ProgressDialog == NULL)
	{
		_ProgressDialog = new QProgressDialog(tr("Loading files"), tr("Stop"), aMin, aMax,this, Qt::ToolTip);
		_Engine->setUpdateSignaler(new ProgressDialogUpdateSignaler(*_ProgressDialog));
	}

	_ProgressDialog->setRange(aMin, aMax);
	_ProgressDialog->setValue(aMin);
	_ProgressDialog->setWindowModality(Qt::WindowModal);
	_ProgressDialog->setCancelButton(NULL);
	_ProgressDialog->setMinimumDuration(500);

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
}

void SaisieQtWindow::runProgressDialog(QFuture<void> aFuture, int aMin, int aMax)
{
    bool bShowMsgs = _ui->actionShow_messages->isChecked();
    on_actionShow_messages_toggled(false);

    //~ _FutureWatcher.setFuture(aFuture);
	activateLoadImageProgressDialog(aMin, aMax);
    aFuture.waitForFinished();
    on_actionShow_messages_toggled(bShowMsgs);
}

bool SaisieQtWindow::loadPly(const QStringList& filenames)
{
    runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadClouds,filenames), 0, filenames.size());
    return true;
}

bool SaisieQtWindow::loadImages(const QStringList& filenames)
{
	#ifdef USE_MIPMAP_HANDLER
		_Engine->loadImages(filenames);
	#else
    	runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadImages, filenames), 0, filenames.size());
	#endif

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

    runProgressDialog(QtConcurrent::run(_Engine, &cEngine::loadCameras,filenames), 0, filenames.size());
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
                QMessageBox::critical(this, tr("Error"), QString(tr("File [%1] does not exist (or bad argument)")).arg(filenames[i]));
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
						#ifdef USE_MIPMAP_HANDLER
							setDataToGLWidget(aK, _Engine->getGLData(aK));
						#else
							getWidget(aK)->setGLData(_Engine->getGLData(aK), _ui->actionShow_messages->isChecked(), _ui->actionShow_cams->isChecked(),true,true,_params->eNavigation());
						#endif

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

void SaisieQtWindow::on_actionShow_Zoom_window_toggled(bool show)
{
    _ui->QFrame_zoom->setVisible(show);
}


void SaisieQtWindow::on_actionShow_3D_view_toggled(bool show)
{
    _ui->frame_preview3D->setVisible(show);
}

void SaisieQtWindow::on_actionShow_list_polygons_toggled(bool show)
{
    tableView_Objects()->setVisible(show);

    if(show)
    {
        QList<int> sizeS;
        sizeS << 1 << 1;
        _ui->splitter_Tools->show();
        _ui->splitter_GLWid_Tools->setSizes(sizeS);
        resizeTables();
    }

    else
    {
        QList<int> sizeS;
        sizeS << 1 << 0;
        _ui->splitter_Tools->hide();
        _ui->splitter_GLWid_Tools->setSizes(sizeS);
    }

}

void SaisieQtWindow::selectionObjectChanged(const QItemSelection& select, const QItemSelection& unselect)
{
    if(select.indexes().size()>0)
    {
        cPolygon* polygon	= getWidget(0)->getGLData()->currentPolygon();
        selectInfos info	= getWidget(0)->getHistoryManager()->getSelectInfo(select.indexes()[0].row());
        polygon->setVector(info.poly);
        polygon->close();
    }
    else
    {
        getWidget(0)->getGLData()->currentPolygon()->clear();
    }

    getWidget(0)->update();
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

void SaisieQtWindow::on_actionWorkbench_toggled(bool mode)
{
   if(!_workBench)
   {
       _workBench = new cWorkBenchWidget;
       _workBench->setDIOCamera(_devIOCamera);
       _workBench->setDIOTieFile(_devIOTieFile);
       _workBench->setDIOImage(_Engine->Loader()->devIOImageAlter());

   }

   if(mode)
       _workBench->show();
   else
       _workBench->hide();

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
    }
    if ((_appMode == MASK3D) || (_appMode == MASK2D))
    {
        shortcuts.push_back(Ctrl + "O");
        actions.push_back(tr("open image file"));
        shortcuts.push_back(Ctrl + "+S");
        actions.push_back(tr("save mask"));
        shortcuts.push_back(Ctrl + "Maj+S");
        actions.push_back(tr("save file as"));
        shortcuts.push_back(Ctrl + "X");
        actions.push_back(tr("close files"));
    }
    if (_appMode != BOX2D)
    {
        shortcuts.push_back(Ctrl + "T");
        actions.push_back(tr("settings"));
    }
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

    float shiftStep = _params->getShiftStep();

    if (_appMode == MASK3D)
    {
        shortcuts.push_back(tr("Navigation 3D"));
        actions.push_back("");

        shortcuts.push_back(tr("camera rotate x and y"));
        actions.push_back(tr("Left button \t+ move mouse"));

        shortcuts.push_back(tr("camera rotate z"));
        actions.push_back(tr("Right button \t+ move mouse (only ball navigation)"));

        shortcuts.push_back(tr("Zoom"));
        actions.push_back(tr("wheel or shift + middle button"));

        shortcuts.push_back(tr("move"));
        actions.push_back(tr("middle button + move mouse"));

        shortcuts.push_back(tr("move on vertex"));
        actions.push_back(tr("Double click on vertex"));

        shortcuts.push_back("");
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
        if (_appMode != BOX2D)
        {
            shortcuts.push_back(tr("Left click"));
            actions.push_back(tr("add a vertex to polygon"));
            shortcuts.push_back(tr("Right click"));
            actions.push_back(tr("close polygon or delete nearest vertex"));
            shortcuts.push_back(tr("Echap"));
            actions.push_back(tr("delete polygon"));
            shortcuts.push_back(tr("W+drag"));
            actions.push_back(tr("move polygon"));

    #ifdef ELISE_Darwin
            shortcuts.push_back(tr("Fn+Space bar"));
            shortcuts.push_back("Fn+D");
            shortcuts.push_back("Fn+U");
            shortcuts.push_back("Fn+Y");
            fillStringList(actions, _appMode);
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
            actions.push_back(tr("move selected vertex") + " (" + QString::number(shiftStep).toStdString().c_str() +" px)" + tr(" - see Settings"));
            shortcuts.push_back(tr("Alt+arrow keys"));
            actions.push_back(tr("move selected vertex") + " (" + QString::number(10.f*shiftStep).toStdString().c_str() + " px)");
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
            shortcuts.push_back(tr("Click+drag"));
            actions.push_back(tr("draw box, or edit box"));
        }
    }
    else
    {
        shortcuts.push_back(tr("Left click"));
        actions.push_back(tr("add point"));
        shortcuts.push_back(tr("Right click"));
        actions.push_back(tr("show state menu or window menu"));
        shortcuts.push_back(tr("Drag & drop"));
        actions.push_back(tr("move selected point"));
        shortcuts.push_back(tr("Arrow keys"));
        actions.push_back(tr("move selected point") + " (" + QString::number(shiftStep).toStdString().c_str() +" px)" + tr(" - see Settings"));
        shortcuts.push_back(tr("Alt+arrow keys"));
        actions.push_back(tr("move selected point") + " (" + QString::number(10.f*shiftStep).toStdString().c_str() + " px)");
    }
    if ((_appMode == MASK3D) || (_appMode == MASK2D)) //TEMP: TODO corriger le undo Elise - Pas disponible en mode Box2D
    {
        shortcuts.push_back(Ctrl +"Z");
        actions.push_back(tr("undo last action"));
        shortcuts.push_back(Ctrl + "Shift+Z");
        actions.push_back(tr("redo last action"));
    }

    shortcuts.push_back("");
    actions.push_back("");

    shortcuts.push_back(tr("G / H / J keys"));
    actions.push_back(tr("Increase / decrease / reset gamma"));

    _helpDialog->populateTableView(shortcuts, actions);
}

void SaisieQtWindow::on_actionAbout_triggered()
{
    QFont font("Courier New", 9, QFont::Normal);

    QMessageBox *msgBox = new QMessageBox(this);

    QString qStr(_banniere);
#if (ELISE_windows || (defined ELISE_Darwin))
        qStr.replace( "**", "  " );
#endif

    QString adressbit(" " + QString::number(sizeof(int*)*8) + " bits");

    GlExtensions glExtensions;
    qStr += "\n" + tr("Application") + "\t" + QApplication::applicationName() + adressbit +
            + "\n" +  tr("Built with \tQT ") + QT_VERSION_STR  +
            + "\n" +  "OpenGL     \t[" + glExtensions.version().c_str() + "] [" + glExtensions.vendor().c_str() + "]" +
            + "\n" +  tr("Revision\t") + _git_revision + "\n";

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
    for (int i = 0; i < nbWidgets(); ++i)
    {
        if(getWidget(i)->getGLData())
            {
                if(getWidget(i)->getGLData()->polygonCount() == 1)
                {
                    cPolygon* polyg = new cPolygon(2,1.0,Qt::yellow,Qt::yellow, Geom_cross);
                    polyg->setPointSize(10);
                    getWidget(i)->getGLData()->addPolygon(polyg);
                }
                getWidget(i)->getGLData()->setCurrentPolygonIndex(check ? 1 : 0);
                //getWidget(i)->getGLData()->polygon(1)->setAllVisible(check);

                getWidget(i)->getGLData()->polygon(1)->setAllVisible(true);

                if(check)
                    _ui->label_ImagePosition_2->show();
                else
                    _ui->label_ImagePosition_2->hide();
            }
    }

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

void SaisieQtWindow::on_actionConfirm_changes_triggered()
{
    QItemSelection select	= tableView_Objects()->selectionModel()->selection();
    if(select.indexes().size())
    {
        int id = select.indexes()[0].row();
        selectInfos &	info	= getWidget(0)->getHistoryManager()->getSelectInfo(id);
        cPolygon * currentPoly	= currentWidget()->getGLData()->polygon();
        info.poly				= currentPoly->getVector();
        currentPoly->clear();
        updateMask();
        tableView_Objects()->clearSelection();
    }
}

void SaisieQtWindow::on_actionRemove_inside_triggered()
{

    if (_appMode > MASK3D)
        currentWidget()->polygon(0)->removeSelectedPoint();  //TODO: actuellement on ne garde pas le point selectionné (ajouter une action)
    else
        currentWidget()->Select(SUB_INSIDE);
}

void SaisieQtWindow::on_actionRemove_outside_triggered()
{
    currentWidget()->Select(SUB_OUTSIDE);
}

void SaisieQtWindow::on_actionUndo_triggered(){

    if (_appMode <= MASK3D)
    {
        currentWidget()->undo();
        updateMask(true);
    }
    else

        emit undoSgnl(true);

}

void SaisieQtWindow::on_actionRedo_triggered()
{
    if (_appMode <= MASK3D)
    {
        currentWidget()->redo();
        updateMask(false);
    }
    else

        emit undoSgnl(false);
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
deviceIOTieFile* SaisieQtWindow::devIOTieFile() const
{
    return _devIOTieFile;
}

void SaisieQtWindow::setDevIOTieFile(deviceIOTieFile* devIOTieFile)
{
    _devIOTieFile = devIOTieFile;
}

QString SaisieQtWindow::textToolBar() const
{
    return _textToolBar;
}

void SaisieQtWindow::setTextToolBar(const QString& textToolBar)
{
    _textToolBar = textToolBar;
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

    bool isModeMask = _appMode == MASK3D || _appMode == MASK2D || _appMode == BOX2D;
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
    _ui->menuWindows->menuAction()->setVisible(_appMode != BOX2D);
    _ui->menuTools->menuAction()->setVisible(_appMode != BOX2D);
    _ui->menuSelection->menuAction()->setVisible(_appMode == MASK2D || _appMode == MASK3D);

    hideAction(_ui->actionSettings, _appMode != BOX2D);

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
        _ui->actionRemove_inside->setShortcut(Qt::Key_D);
        _ui->actionRemove_outside->setShortcut(Qt::Key_Y);
        _ui->actionAdd_outside->setShortcut(Qt::Key_U);
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
        else if (_appMode == BASC)             setWindowTitle("Micmac - SaisieQT");

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

        _ui->tableView_Objects->close();
       // _ui->splitter_Tools->setContentsMargins(2,0,0,0);
    }
    else
    {
        _ui->QFrame_zoom->close();
        _ui->tableView_PG->close();
        _ui->tableView_Images->close();
        _ui->frame_preview3D->close();
        _ui->tableView_Objects->hide();
        _ui->splitter_Tools->hide();

        QList<int> sizeS;
        sizeS << 1 << 0;
        _ui->splitter_GLWid_Tools->setSizes(sizeS);

        _ui->splitter_GLWid_Tools->setStretchFactor(0,5);
        _ui->splitter_GLWid_Tools->setStretchFactor(1,2);
    }

    //TEMP:
    //hideAction(_ui->menuTools->menuAction(), false);

    if (_appMode <= MASK3D)
    {
        hideAction(_ui->actionShow_Zoom_window, false);
        hideAction(_ui->actionShow_3D_view, false);
    }
    else
        hideAction(_ui->actionShow_list_polygons, false);
}

#ifdef USE_MIPMAP_HANDLER
	void SaisieQtWindow::keyPressEvent( QKeyEvent *aEvent )
	{
		switch (aEvent->key())
		{
		case Qt::Key_Return:
			on_actionConfirm_changes_triggered();
			break;
		case Qt::Key_PageUp:
			changeDisplayedImages(false); // false = aForward
			break;
		case Qt::Key_PageDown:
			changeDisplayedImages(true); // true = aForward
			break;
		}
	}
#else
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

	void SaisieQtWindow::keyPressEvent(QKeyEvent *event)
	{
		switch(event->key())
		{
		case Qt::Key_Return:
		    on_actionConfirm_changes_triggered();
		    break;
		default:
		    return;
		}
	}
#endif

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

void SaisieQtWindow::setModelObject(QAbstractItemModel *model_Objects)
{
    tableView_Objects()->setModel(model_Objects);
}

void SaisieQtWindow::setModel(QAbstractItemModel *model_Pg, QAbstractItemModel *model_Images)
{
    tableView_PG()->setModel(model_Pg);
    tableView_Images()->setModel(model_Images);
}

void SaisieQtWindow::selectPointAllWGL(QString pointName)
{
    emit selectPoint(pointName);
}

void SaisieQtWindow::setDataToGLWidget(int idGLW, cGLData *glData)
{
	if (glData == NULL) return;

	#ifdef USE_MIPMAP_HANDLER
		if (glData->glImageMasked().hasSrcImage()) _Engine->Loader()->loadImage(glData->glImageMasked().srcImage());
		if (glData->glImageMasked().hasSrcMask() && !_Engine->Loader()->loadImage(glData->glImageMasked().srcMask()))
		{
			cout << "ignoring mask image [" << glData->glImageMasked().srcMask().mFilename << "]: cannot be loaded" << endl;
			glData->glImageMasked().removeSrcMask();
		}
	#endif

	GLWidget * glW = getWidget(idGLW);
	glW->setGLData(glData, glData->stateOption(cGLData::OpShow_Mess), glData->stateOption(cGLData::OpShow_Cams));
	glW->setParams(getParams());
}

void SaisieQtWindow::loadPlyIn3DPrev(const QStringList &filenames, cData *dataCache)
{
    if (filenames.size())
    {
        for (int i=0; i< filenames.size();++i)
        {
            if(!QFile(filenames[i]).exists())
            {
                QMessageBox::critical(this, tr("Error"), tr("File does not exist (or bad argument)"));
                return;
            }
        }

        QString suffix = QFileInfo(filenames[0]).suffix();

        if (suffix == "ply")
        {
            loadPly(filenames);
            threeDWidget()->getGLData()->clearClouds();
            dataCache->computeCenterAndBBox(1);
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

void SaisieQtWindow::setAutoName(QString val)
{
    emit setName(val);
}

void SaisieQtWindow::setImagePosition(QPointF pt)
{
    QString text(tr("Image position : "));
    QString textRule;

    if (pt.x() >= 0.f && pt.y() >= 0.f)
    {
        GLWidget* glW = currentWidget();
        if(glW)
            if ( glW->hasDataLoaded() && !glW->getGLData()->is3D() && (glW->isPtInsideIm(pt)))
            {
				ELISE_DEBUG_ERROR(glW->getGLData()->glImageMasked()._m_image == NULL, " SaisieQtWindow::setImagePosition", "glW->getGLData()->glImageMasked()._m_image == NULL");
                int imHeight = glW->getGLData()->glImageMasked()._m_image->height();

                text = QString(text + QString::number(pt.x(),'f',1) + ", " + QString::number((imHeight - pt.y()),'f',1)+" px");

                if(glW->getGLData()->getCurrentPolygonIndex() == 1)
                {
                    textRule = QString(text + " \t ") + tr("Image length : ") + QString::number(glW->getGLData()->currentPolygon()->length()) + QString(" px");
                }
            }
    }

    _ui->label_ImagePosition_1->setText(text);
    _ui->label_ImagePosition_2->setText(textRule + QString(" ") + textToolBar());
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

void SaisieQtWindow::updateMask(bool reloadMask)
{
    // TODO seg fault dans le undo � cause de la destruction des images...

    if (currentWidget()->getHistoryManager()->size())
    {

        if ((_appMode != MASK3D) && reloadMask)
        {
            int idx = currentWidgetIdx();

            bool showMessage = _ui->actionShow_messages->isChecked();
            bool show_cams	 = _ui->actionShow_cams->isChecked();

            currentWidget()->setGLData(NULL, showMessage, show_cams,false,false);

            _Engine->reloadMask(_appMode, idx);

            currentWidget()->setGLData(_Engine->getGLData(idx), showMessage, show_cams, false,false);
        }

        currentWidget()->applyInfos();
        _bSaved = false;
    }

    resizeTables();

}
QString SaisieQtWindow::banniere() const
{
    return _banniere;
}

void SaisieQtWindow::setBanniere(const QString& banniere)
{
    _banniere = banniere;
}

QString SaisieQtWindow::git_revision() const
{
    return _git_revision;
}

void SaisieQtWindow::setGit_revision(QString git_revision)
{
    _git_revision = git_revision;
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

#ifdef USE_MIPMAP_HANDLER
	#ifdef __DEBUG
		void SaisieQtWindow::dumpGLDataIdSet( const vector<int> &aIdSet, const string &aPrefix, ostream &aStream ) const
		{
			for (size_t i = 0; i < aIdSet.size(); i++)
			{
				if (aIdSet[i] == -1)
				{
					aStream << aPrefix << i << ": -1" << endl;
					continue;
				}

				cGLData &data = *_Engine->getGLData(aIdSet[i]);
				aStream << aPrefix << i << ": " << aIdSet[i];
				data.dump(" ", aStream);
			}
		}

		void SaisieQtWindow::dumpAllGLData( const string &aPrefix, ostream &aStream ) const
		{
			vector<int> idSet((size_t)_Engine->nbGLData());
			for (int i = 0; i < _Engine->nbGLData(); i++)
			{
				aStream << aPrefix << i;
				_Engine->getGLData(i)->dump(": ", aStream);
			}
		}
	#endif

	void SaisieQtWindow::loadGLDataIdSet( const vector<int> &aIdSet )
	{
		ELISE_DEBUG_ERROR(aIdSet.size() != (size_t)nbWidgets(), "SaisieQtWindow::loadGLDataIdSet", "aIdSet.size() = " << aIdSet.size() << " != nbWidgets() = " << nbWidgets());

		for (int i = 0; i < nbWidgets(); i++)
			if (aIdSet[i] != -1) getWidget(i)->setGLData(NULL);

		for (size_t i = 0; i < aIdSet.size(); i++)
		{
			if (aIdSet[i] == -1) continue;

			const int newDataId = aIdSet[i];
			#ifdef __DEBUG
				const int nbGLData = _Engine->nbGLData();
				ELISE_DEBUG_ERROR(newDataId < 0 || newDataId >= nbGLData, "SaisieQtWindow::loadGLDataIdSet", "invalid aIdSet[" << i << "] = " << newDataId << ", nbGLData = " << nbGLData);
			#endif
			cGLData &data = *_Engine->getGLData(newDataId);

			ELISE_DEBUG_ERROR( !data.glImageMasked().hasSrcImage(), "SaisieQtWindow::loadGLDataIdSet", "!data.glImageMasked().hasSrcImage()");
			_Engine->Loader()->loadImage(data.glImageMasked().srcImage());

			if (data.glImageMasked().hasSrcMask()) _Engine->Loader()->loadImage(data.glImageMasked().srcMask());

			getWidget((int)i)->setGLData(&data);
		}
	}

	void SaisieQtWindow::changeDisplayedImages( bool aForward )
	{
		ELISE_DEBUG_ERROR(_Engine == NULL, "SaisieQtWindow::changeImages", "_Engine == NULL");

		if (nbWidgets() >= _Engine->nbImages()) return;

		const int nbGLData = _Engine->nbGLData();
		vector<int> loadedIds;
		_Engine->getGLDataIdSet(0, nbGLData - 1, true, nbWidgets(), loadedIds); // true = aIsLoaded
		ELISE_DEBUG_ERROR(loadedIds.size() != (size_t)nbWidgets(), "SaisieQtWindow::changeImages", "loadedIds.size() = " << loadedIds.size() << " != nbWidgets() = " << nbWidgets());

		vector<int> toLoadIds;
		int i0, i1;
		if (aForward)
		{
			i0 = _Engine->minLoadedGLDataId() + 1;
			i1 = nbGLData - 1;
		}
		else
		{
			i0 = _Engine->minLoadedGLDataId() - 1;
			i1 = 0;
		}
		_Engine->getGLDataIdSet(i0, i1, false, nbWidgets(), toLoadIds); // false = aIsLoaded

		if (toLoadIds.size() < (size_t)nbWidgets())
		{
			size_t nbToLoad = toLoadIds.size();
			toLoadIds.resize((size_t)nbWidgets());
			size_t nbToKeep = toLoadIds.size() - nbToLoad;

			if (aForward)
				memcpy(toLoadIds.data() + nbToLoad, loadedIds.data() + nbToLoad, nbToKeep * sizeof(int));
			else
				memcpy(toLoadIds.data() + nbToLoad, loadedIds.data(), nbToKeep * sizeof(int));
		}

		sort(toLoadIds.begin(), toLoadIds.end());

		// prevent unchanged widget association to be unload
		for (size_t i = 0; i < loadedIds.size(); i++)
			if (toLoadIds[i] == loadedIds[i]) toLoadIds[i] = -1; // -1 is as special value for loadGLDataIdSet meaning do not unload and load GLData

		loadGLDataIdSet(toLoadIds);
	}
#endif

ModelObjects::ModelObjects(QObject *parent, HistoryManager* hMag)
    :QAbstractTableModel(parent),
      _hMag(hMag)
{

}

int ModelObjects::rowCount(const QModelIndex & /*parent*/) const
{
    return _hMag->size();
}

int ModelObjects::columnCount(const QModelIndex &parent) const
{
    return 3;
}

QVariant ModelObjects::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole)
    {
        if (orientation == Qt::Horizontal)
        {
            switch (section)
            {
                case 0:
                    return QString(tr("id"));
                case 1:
                    return QString(tr("nb pts"));
                case 2:
                    return QString(tr("Mode"));
            }
        }
    }
    return QVariant();
}

bool ModelObjects::setData(const QModelIndex& index, const QVariant& value, int role)
{

    if (role == Qt::EditRole)
    {
        if(index.column() == 2)
        {
            int id = index.row();
            _hMag->getSelectInfo(id).selection_mode = value.toInt();
            QModelIndex topLeft		= this->index(index.row(),index.column());
            QModelIndex bottomRight = topLeft;
            emit dataChanged( topLeft, bottomRight );
        }
    }

    return true;
}

Qt::ItemFlags ModelObjects::flags(const QModelIndex &index) const
{

    switch (index.column())
    {
        case 2:
        //if(index.row() < PG_Count())
            return QAbstractTableModel::flags(index) | Qt::ItemIsEditable;
    }

    return QAbstractTableModel::flags(index);
}
QVariant ModelObjects::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        int aK = index.row();

        if(aK < _hMag->size())
        {

            QString nonS;
            QVector <selectInfos> sInfo = _hMag->getSelectInfos();
            selectInfos info = sInfo[aK];

            switch (index.column())
            {
                case 0:
                {
                    return nonS.number(aK);
                }
                case 1:
                {

                    return nonS.number(info.poly.size());
                }
                case 2:
                {
                    return QVariant(getSelectionMode()[info.selection_mode]);
                }
            }
        }
    }

    if (role == Qt::BackgroundColorRole)
    {
        if(_hMag->getActionIdx()- 1 == index.row())
            return QColor(Qt::darkCyan);
    }
    return QVariant();
}

bool ModelObjects::insertRows(int row, int count, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row+count-1);
    endInsertRows();
    return true;
}

QStringList ModelObjects::getSelectionMode()
{
    return (QStringList()
            << tr("subtract inside")
            << tr("add inside")
            << tr("subtract outside")
            << tr("add outside")
            << tr("invert selection")
            << tr("select all")
            << tr("select none"));
}

bool ObjectsSFModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{
    //TODO:
    return true;
}

ComboBoxDelegate::ComboBoxDelegate(QStringList const &listCombo, int size, QObject* parent)
    : QStyledItemDelegate(parent),
      _size(size),
      _enumString(listCombo)
{
}

QWidget *ComboBoxDelegate::createEditor(QWidget *parent,
    const QStyleOptionViewItem &/* option */,
    const QModelIndex &/* index */) const
{
    QComboBox *editor = new QComboBox(parent);

    for (int i = 0; i < _size; ++i)
        editor->addItem(_enumString[i]);

    return editor;
}

void ComboBoxDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
    int value = index.model()->data(index, Qt::EditRole).toInt();

    QComboBox *comboBox = static_cast<QComboBox*>(editor);
    comboBox->setCurrentIndex(value);
}

void ComboBoxDelegate::setModelData(QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
    QComboBox *comboBox = static_cast<QComboBox*>(editor);

      int value = comboBox->currentIndex();

      model->setData(index, value, Qt::EditRole);
}

void ComboBoxDelegate::updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    editor->setGeometry(option.rect);
}
