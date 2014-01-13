#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(Pt2di aSzW, Pt2di aNbFen, bool mode2D, QWidget *parent) :
        QMainWindow(parent),
        GLWidgetSet(aNbFen.x*aNbFen.y,colorBG0,colorBG1),
        _ui(new Ui::MainWindow),
        _Engine(new cEngine),
        _layout(new QGridLayout)
{
    _ui->setupUi(this);

    QString style = "border: 2px solid gray;"
            "border-radius: 1px;"
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgb(%1,%2,%3), stop:1 rgb(%4,%5,%6));";

    style = style.arg(colorBG0.red()).arg(colorBG0.green()).arg(colorBG0.blue());
    style = style.arg(colorBG1.red()).arg(colorBG1.green()).arg(colorBG1.blue());

    _ui->OpenglLayout->setStyleSheet(style);

    _ProgressDialog = new QProgressDialog("Loading files","Stop",0,100,this);

    connect(&_FutureWatcher, SIGNAL(finished()),_ProgressDialog,SLOT(cancel()));

    _nbFen = QPoint(aNbFen.x,aNbFen.y);
    _szFen = QPoint(aSzW.x,aSzW.y);

    resize(_szFen.x(), _szFen.y());

    setMode2D(mode2D);

    int cpt=0;
    for (int aK = 0; aK < aNbFen.x;++aK)
        for (int bK = 0; bK < aNbFen.y;++bK, cpt++)
            _layout->addWidget(getWidget(cpt), bK, aK);

    _signalMapper = new QSignalMapper (this);
    connectActions();
    _ui->OpenglLayout->setLayout(_layout);

    createMenus();
}

MainWindow::~MainWindow()
{
    delete _ui;
    delete _Engine;
    delete _RFMenu;
    delete _layout;
    delete _signalMapper;
}

void MainWindow::connectActions()
{
    for (uint aK = 0; aK < NbWidgets();++aK)
        connect(getWidget(aK),	SIGNAL(filesDropped(const QStringList&)), this,	SLOT(addFiles(const QStringList&)));

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

void MainWindow::createMenus()
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

        _Engine->setFilenamesIn(filenames);

        if (_bMode2D == true) closeAll();
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
            // TODO ENCAPSULER LA PROGRESS BAR
            QTimer *timer_test = new QTimer(this);
            _incre = new int(0);
            connect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            timer_test->start(10);
            QFuture<void> future = QtConcurrent::run(_Engine, &cEngine::loadClouds,filenames,_incre);

            _FutureWatcher.setFuture(future);
            _ProgressDialog->setWindowModality(Qt::WindowModal);
            _ProgressDialog->exec();

            timer_test->stop();
            disconnect(timer_test, SIGNAL(timeout()), this, SLOT(progression()));
            delete _incre;
            delete timer_test;                     

            future.waitForFinished();            
            // FIN DE CHARGEMENT ET PROGRESS BAR

            _Engine->setFilename();
            _Engine->setFilenamesOut();
        }
        else if (fi.suffix() == "xml")
        {
            // TODO ENCAPSULER LA PROGRESS BAR
            QFuture<void> future = QtConcurrent::run(_Engine, &cEngine::loadCameras, filenames);

            _FutureWatcher.setFuture(future);
            _ProgressDialog->setWindowModality(Qt::WindowModal);
            _ProgressDialog->exec();

            future.waitForFinished();
            // FIN DE CHARGEMENT ET PROGRESS BAR

            _ui->actionShow_cams->setChecked(true);
        }
        else // LOAD IMAGE
        {
            setMode2D(true);
            closeAll();            

            _Engine->loadImages(filenames);
            _Engine->setFilenamesOut();
        }

        _Engine->allocAndSetGLData();
        for (uint aK = 0; aK < NbWidgets();++aK)
            getWidget(aK)->setGLData(_Engine->getGLData(aK),_ui->actionShow_messages);

        for (int aK=0; aK< filenames.size();++aK) setCurrentFile(filenames[aK]);
    }
}

void MainWindow::on_actionFullScreen_toggled(bool state)
{   
    return state ? showFullScreen() : showNormal();
}

void MainWindow::on_actionShow_ball_toggled(bool state)
{
    if (!_bMode2D)
    {
        CurrentWidget()->setOption(cGLData::OpShow_Ball,state);

        if (state && _ui->actionShow_axis->isChecked())
        {
            CurrentWidget()->setOption(cGLData::OpShow_BBox,!state);
            _ui->actionShow_axis->setChecked(!state);
        }
    }
}

void MainWindow::on_actionShow_bbox_toggled(bool state)
{
    if(!_bMode2D)
        CurrentWidget()->setOption(cGLData::OpShow_BBox,state);
}

void MainWindow::on_actionShow_axis_toggled(bool state)
{
    if (!_bMode2D)
    {
        CurrentWidget()->setOption(cGLData::OpShow_Axis,state);

        if (state && _ui->actionShow_ball->isChecked())
        {
            CurrentWidget()->setOption(cGLData::OpShow_Ball,!state);
            _ui->actionShow_ball->setChecked(!state);
        }
    }
}

void MainWindow::on_actionShow_cams_toggled(bool state)
{
    if (!_bMode2D)
        CurrentWidget()->setOption(cGLData::OpShow_Cams,state);
}

void MainWindow::on_actionShow_messages_toggled(bool state)
{
    CurrentWidget()->setOption(cGLData::OpShow_Mess,state);
}

void MainWindow::on_actionToggleMode_toggled(bool mode)
{
    if (!_bMode2D)
        CurrentWidget()->setInteractionMode(mode ? SELECTION : TRANSFORM_CAMERA,_ui->actionShow_messages->isChecked());
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
    text += "F8: \t"+tr("2D mode / 3D mode") +"\n";

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

    text += "Shift+R: \t"+tr("reset view") +"\n\n";


    text += tr("Selection menu:") +"\n\n";
    if (!_bMode2D)
    {
        text += "F9: \t"+tr("move mode / selection mode (only 3D)") +"\n\n";
    }
    text += tr("Left click : \tadd a vertex to polyline") +"\n";
    text += tr("Right click: \tclose polyline or delete nearest vertex") +"\n";
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
    text += tr("Ctrl+right click: remove last vertex") +"\n";
    text += tr("Drag & drop: move polyline vertex") +"\n";
    text += "Ctrl+A: \t"+tr("select all") +"\n";
    text += "Ctrl+D: \t"+tr("select none") +"\n";
    text += "Ctrl+R: \t"+tr("reset") +"\n";
    text += "Ctrl+I: \t"+tr("invert selection") +"\n";
    text += "Ctrl+Z: \t"+tr("undo last selection") +"\n";

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
    CurrentWidget()->Select(ADD);
}

void MainWindow::on_actionSelect_none_triggered()
{
    CurrentWidget()->Select(NONE);
}

void MainWindow::on_actionInvertSelected_triggered()
{
    CurrentWidget()->Select(INVERT);
}

void MainWindow::on_actionSelectAll_triggered()
{
    CurrentWidget()->Select(ALL);
}

void MainWindow::on_actionReset_triggered()
{
    if (_bMode2D)
    {
        closeAll();

        addFiles(_Engine->getFilenamesIn());
    }
    else
    {
        CurrentWidget()->Select(ALL);
    }
}

void MainWindow::on_actionRemove_triggered()
{
    CurrentWidget()->Select(SUB);
}

void MainWindow::on_actionUndo_triggered()
{   
    QVector <selectInfos> vInfos = CurrentWidget()->getSelectInfos();

    if (vInfos.size())
    {
        if (_bMode2D)
        {
            int idx = CurrentWidgetIdx();

            _Engine->reloadImage(idx);

            CurrentWidget()->setGLData(_Engine->getGLData(idx),_ui->actionShow_messages);
        }

        vInfos.pop_back();
        CurrentWidget()->applyInfos(vInfos);
    }
}

void MainWindow::on_actionRedo_triggered()
{

}

void MainWindow::on_actionSetViewTop_triggered()
{
    if (!_bMode2D)
        CurrentWidget()->setView(TOP_VIEW);
}

void MainWindow::on_actionSetViewBottom_triggered()
{
    if (!_bMode2D)
        CurrentWidget()->setView(BOTTOM_VIEW);
}

void MainWindow::on_actionSetViewFront_triggered()
{
    if (!_bMode2D)
        CurrentWidget()->setView(FRONT_VIEW);
}

void MainWindow::on_actionSetViewBack_triggered()
{
    if (!_bMode2D)
        CurrentWidget()->setView(BACK_VIEW);
}

void MainWindow::on_actionSetViewLeft_triggered()
{
    if (!_bMode2D)
        CurrentWidget()->setView(LEFT_VIEW);
}

void MainWindow::on_actionSetViewRight_triggered()
{
    if (!_bMode2D)
        CurrentWidget()->setView(RIGHT_VIEW);
}

void MainWindow::on_actionReset_view_triggered()
{
    CurrentWidget()->resetView(true,true,true);
}

void MainWindow::on_actionZoom_Plus_triggered()
{
    CurrentWidget()->setZoom(CurrentWidget()->getZoom()*1.5f);
}

void MainWindow::on_actionZoom_Moins_triggered()
{
    CurrentWidget()->setZoom(CurrentWidget()->getZoom()/1.5f);
}

void MainWindow::on_actionZoom_fit_triggered()
{
    CurrentWidget()->zoomFit();
}

void MainWindow::zoomFactor(int aFactor)
{
    CurrentWidget()->zoomFactor(aFactor);
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
    _Engine->saveMask(CurrentWidgetIdx());
}

void MainWindow::on_actionSave_as_triggered()
{
    QString fname = QFileDialog::getSaveFileName(NULL, tr("Save mask Files"),QString(), tr("Files (*.*)"));

    if (!fname.isEmpty())
    {
        _Engine->setFilenameOut(fname);

        _Engine->saveMask(CurrentWidgetIdx());
    }
}

void MainWindow::on_actionSave_selection_triggered()
{
    _Engine->saveSelectInfos(CurrentWidget()->getSelectInfos());
}

void MainWindow::closeAll()
{
    _Engine->unloadAll();

    for (uint aK=0; aK < NbWidgets(); ++aK)
        getWidget(aK)->reset();
}

void MainWindow::openRecentFile()
{
    // A TESTER en multi images
    QAction *action = qobject_cast<QAction *>(sender());
    if (action)
    {
        _Engine->setFilenamesIn(QStringList(action->data().toString()));

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
    _Engine->setGamma(aGamma);
}
