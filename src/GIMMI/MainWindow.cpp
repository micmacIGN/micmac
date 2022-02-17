#include "MainWindow.h"
using namespace std;


MainWindow::MainWindow()
{
    GI_MicMacDir = qApp->applicationDirPath();

    QPalette pal = palette();

    // set black background
    pal.setColor(QPalette::Background, Qt::white);
    this->setAutoFillBackground(true);
    this->setPalette(pal);
    this->show();

    this->setWindowIcon(QIcon("/home/atruffier/Bureau/micmac.jpg"));

    resize(1900,1100);

    createMenu();
    createToolBar();
    createDock();
    createCentralWidget();
}


// STRUCTURE DE LA MAIN WINDOW
void MainWindow::createMenu()
{
    pathsActions = readPaths();

    // Creation of the menu pathProject
    QMenu *menuProject = menuBar()->addMenu("&File");
    QAction *actionLoadProject = menuProject->addAction("&New projet");

    //    actionLoadProject->setShortcut(QKeySequence("Ctrl+N"));
    QObject::connect(actionLoadProject, SIGNAL(triggered(bool)), this, SLOT(loadProject()));
    
    QAction *actionRecentProject = menuProject->addAction("&Recent Project");
    //    actionRecentProject->setShortcut(QKeySequence("Ctrl+Q"));
    QMenu *menuP = menuBar()->addMenu("");
    QAction *ac1 = menuP->addAction(pathsActions[0]);
    QAction *ac2 = menuP->addAction(pathsActions[1]);
    QAction *ac3 = menuP->addAction(pathsActions[2]);
    QAction *ac4 = menuP->addAction(pathsActions[3]);
    QAction *ac5 = menuP->addAction(pathsActions[4]);

    actionRecentProject->setMenu(menuP);
    QObject::connect(ac1, SIGNAL(triggered(bool)), this, SLOT(loadExistingProject1()));
    QObject::connect(ac2, SIGNAL(triggered(bool)), this, SLOT(loadExistingProject2()));
    QObject::connect(ac3, SIGNAL(triggered(bool)), this, SLOT(loadExistingProject3()));
    QObject::connect(ac4, SIGNAL(triggered(bool)), this, SLOT(loadExistingProject4()));
    QObject::connect(ac5, SIGNAL(triggered(bool)), this, SLOT(loadExistingProject5()));
    
    QAction *actionQuit = menuProject->addAction("&Quit");
    //    actionQuit->setShortcut(QKeySequence("Ctrl+Q"));
    QObject::connect(actionQuit, SIGNAL(triggered(bool)), qApp, SLOT(quit()));


    
    // Creation of the menu SATELITE WORKFLOW
    QMenu *menuWorkflowSatellite = menuBar()->addMenu("&Satellite WorkFlow");
    QAction *actionTapioca = menuWorkflowSatellite->addAction("&Tapioca : Tie points computation");
    //    actionTapioca->setShortcut(QKeySequence("Ctrl+H"));
    actionTapioca->setToolTip("Ceci est une info bulle");
    QObject::connect(actionTapioca, SIGNAL(triggered(bool)), this, SLOT(tapioca()));
    
    QAction *actionConvert2GenBunddle = menuWorkflowSatellite->addAction("&Convert2GenBunddle :  Import RPC or other to MicMac format, for adjustment, matching");
    //    actionConvert2GenBunddle->setShortcut(QKeySequence("Ctrl+O"));
    actionConvert2GenBunddle->setToolTip("Ceci est une info bulle");
    QObject::connect(actionConvert2GenBunddle, SIGNAL(triggered(bool)), this, SLOT(convert()));
    
    QAction *actionCampari = menuWorkflowSatellite->addAction("&Campari : Lever arm estimation");
    //    actionCampari->setShortcut(QKeySequence("Ctrl+3"));
    actionCampari->setToolTip("Ceci est une info bulle");
    QObject::connect(actionCampari, SIGNAL(triggered(bool)), this, SLOT(campari()));
    
    QAction *actionMalt = menuWorkflowSatellite->addAction("&Malt : Matching in ground geometry");
    //    actionMalt->setShortcut(QKeySequence("Ctrl+H"));
    actionMalt->setToolTip("Ceci est une info bulle");
    QObject::connect(actionMalt, SIGNAL(triggered(bool)), this, SLOT(malt()));
    
    
    
    
    // Creation of the menu CLASSIC WORKFLOW
    QMenu *menuWorklowDrone = menuBar()->addMenu("&Classic WorkFlow");
    QAction *actionTapiocaDr = menuWorklowDrone->addAction("&Tapioca : Tie points computation");
    //    actionTapiocaDr->setShortcut(QKeySequence("Ctrl+H"));
    actionTapiocaDr->setToolTip("Ceci est une info bulle");
    QObject::connect(actionTapiocaDr, SIGNAL(triggered(bool)), this, SLOT(tapioca()));

    QAction *actionTapasDr = menuWorklowDrone->addAction("&Tapas : relative orientation and calibration");
    //    actionTapasDr->setShortcut(QKeySequence("Ctrl+O"));
    actionTapasDr->setToolTip("Ceci est une info bulle");
    QObject::connect(actionTapasDr, SIGNAL(triggered(bool)), this, SLOT(tapas()));
    
    QAction *actionMaltDr = menuWorklowDrone->addAction("&Malt : Matching in ground geometry");
    //    actionMaltDr->setShortcut(QKeySequence("Ctrl+3"));
    actionMaltDr->setToolTip("Ceci est une info bulle");
    QObject::connect(actionMaltDr, SIGNAL(triggered(bool)), this, SLOT(malt()));
    
    QAction *actionC3DCDr = menuWorklowDrone->addAction("&C3DC : Point cloud computation from a set of oriented images");
    //    actionC3DCDr->setShortcut(QKeySequence("Ctrl+H"));
    actionC3DCDr->setToolTip("Ceci est une info bulle");
    QObject::connect(actionC3DCDr, SIGNAL(triggered(bool)), this, SLOT(c3dc()));
    
    
    
    
    // Creation of the menu Visualization Tools
    QMenu *menuVisualizationTools = menuBar()->addMenu("&Visualization Tools");
    QAction *actionSEL = menuVisualizationTools->addAction("&Sel");
    //    actionSEL->setShortcut(QKeySequence("Ctrl+H"));
    actionSEL->setToolTip("Ceci est une info bulle");
    QObject::connect(actionSEL, SIGNAL(triggered(bool)), this, SLOT(sel()));
    
    
    QAction *actionMeshlab = menuVisualizationTools->addAction("&Meshlab");
    //    actionMeshlab->setShortcut(QKeySequence("Ctrl+O"));
    actionMeshlab->setToolTip("Ceci est une info bulle");
    QObject::connect(actionMeshlab, SIGNAL(triggered(bool)), this, SLOT(meshlab()));
    
    QAction *actionAperi = menuVisualizationTools->addAction("&AperiCloud");
    //    actionAperi->setShortcut(QKeySequence("Ctrl+3"));
    actionAperi->setToolTip("Ceci est une info bulle");
    QObject::connect(actionAperi, SIGNAL(triggered(bool)), this, SLOT(aperi()));
    
    
    
}

void MainWindow::createDock()
{
    listFileFoldersProject.clear();

    dock = new QDockWidget("", this);
    QPalette pal = palette();
    dock->setMaximumWidth(580);

    // set black background
    pal.setColor(QPalette::Background, Qt::white);
    dock->setAutoFillBackground(true);
    dock->setPalette(pal);
    dock->show();
    dockLayoutv = new QVBoxLayout;
    addDockWidget(Qt::LeftDockWidgetArea, dock);
    QWidget *contenuDock = new QWidget;
    dock->setWidget(contenuDock);
    contenuDock->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    
    //Label "Orientation :: ......"
    QLabel *labelCurrentOri = new QLabel(this);
    labelCurrentOri->setStyleSheet("font-weight: bold; color: orange");
    std::string OriResidu="Orientation : " + readResiduOri();
    QString qstrOriResidu = QString::fromStdString(OriResidu);
    labelCurrentOri->setText(qstrOriResidu);
    QPushButton *refresh = new QPushButton();
    refresh->setFixedWidth(105);
    refresh->setText("Refresh Dock");
    connect(refresh, SIGNAL(clicked(bool)), this, SLOT(dockUpdate()));

    dockLayoutv->addWidget(refresh);
    dockLayoutv->addWidget(labelCurrentOri);

    //ZONE WOKRKSPACE
    treeViewFloders = new QTreeView();
    model = new QDirModel;
    model->setFilter(QDir::NoDotAndDotDot | QDir::AllDirs | QDir::AllEntries);

    if(pathProject!=NULL){
        
        treeViewFloders->setModel(model);
        treeViewFloders->setRootIndex(model->index(pathProject));
        treeViewFloders->setSortingEnabled(true);
        treeViewFloders->hideColumn(1);
        treeViewFloders->hideColumn(2);
        treeViewFloders->hideColumn(3);
        parentIndex = model->index(pathProject);
        int numRows = model->rowCount(parentIndex);

        for (int row = 0; row < numRows; ++row) {
            QModelIndex childIndex = model->index(row, 0, parentIndex);
            QString path = model->data(childIndex).toString();
            if(path!=NULL){
                listFileFoldersProject.append(path);
            }
        }
        listFileFoldersProject_all = listFileFoldersProject;
                qDebug() << listFileFoldersProject;
    }
    
    connect(treeViewFloders, SIGNAL(clicked(QModelIndex)), this, SLOT(setTextFileReader()));
    connect(treeViewFloders, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(vino2click()));

    
    treeViewImages = new QTableView();
    treeViewImages->setSelectionBehavior( QAbstractItemView::SelectItems );
    treeViewImages->setSelectionMode(QAbstractItemView::SingleSelection);
    mod = new QStandardItemModel();

    numColumn = 0;
    

    if(!pathProject.isEmpty()){
        
        std::string pathimages;
        pathimages=pathProject.toStdString()+"/listeimages.xml";
        
        FILE * fp = fopen(pathimages.c_str() , "rb");
        if(fp == NULL)
        { std::cout << "Empty in Dock";}
        else{
            
            
            cListOfName aL0 =  StdGetFromPCP(pathProject.toStdString()+"/listeimages.xml",ListOfName);
            std::list< std::string > aLS = aL0.Name();
            std::list< std::string >::iterator it;
            for(it = aLS.begin(); it!=aLS.end(); ++it)
            {
                std::string itToStrg = *it;
                QString qstr = QString::fromStdString(itToStrg);
                QStandardItem *item0 = new QStandardItem();
                item0->setText(qstr);
                mod->setItem(numColumn,0,item0);
                std::string size = pathProject.toStdString()+"/"+itToStrg;
                //     std::cout <<size;

                findResidu();

                if(!nameresidu2.isEmpty())
                {
                    for(int a = 0; a<nameresidu2.size(); a++)
                    {
                        if(itToStrg==nameresidu2.at(a).toStdString())
                        {

                            QStandardItem *itemResidu = new QStandardItem();
                            itemResidu->setText(nameresidu2.at(a+1));
                            mod->setItem(numColumn,1,itemResidu);
                            itemResidu->setFlags(Qt::NoItemFlags);

                            QStandardItem *itemPercOk = new QStandardItem();
                            itemPercOk->setText(nameresidu2.at(a+2));
                            mod->setItem(numColumn,2,itemPercOk);
                            itemPercOk->setFlags(Qt::NoItemFlags);

                            QStandardItem *itemNbPts = new QStandardItem();
                            itemNbPts->setText(nameresidu2.at(a+3));
                            mod->setItem(numColumn,3,itemNbPts);
                            itemNbPts->setFlags(Qt::NoItemFlags);

                            QStandardItem *itemNbPtsMul = new QStandardItem();
                            itemNbPtsMul->setText(nameresidu2.at(a+4));
                            mod->setItem(numColumn,4,itemNbPtsMul);
                            itemNbPtsMul->setFlags(Qt::NoItemFlags);

                        }
                    }
                }
                
                numColumn=numColumn+1;
            }
            
            treeViewImages->setModel(mod);
            QModelIndex index = treeViewImages->model()->index(0, 0);
            treeViewImages->setColumnWidth(0,150);
            mod->setHeaderData(0, Qt::Horizontal, tr("Images"));
            mod->setHeaderData(1, Qt::Horizontal, tr("Résidu"));
            mod->setHeaderData(2, Qt::Horizontal, tr("PercOk"));
            mod->setHeaderData(3, Qt::Horizontal, tr("NbPts"));
            mod->setHeaderData(4, Qt::Horizontal, tr("NbPtsMul"));
            treeViewImages->selectionModel()->select(index, QItemSelectionModel::Select);
            connect(treeViewImages, SIGNAL(clicked(QModelIndex)), this, SLOT(doSomethingDock(QModelIndex)));
        }
        
    }

    QSplitter *split2 = new QSplitter;
    split2->addWidget(treeViewImages);
    split2->addWidget(treeViewFloders);
    split2->setOrientation(Qt::Vertical);
    dockLayoutv->addWidget(split2);
    contenuDock->setLayout(dockLayoutv);
}

void MainWindow::createToolBar()
{

    QToolBar *toolBar = addToolBar("ToolBar");

    QAction *actionLoadProjectTB = toolBar->addAction("&Load Project");
    toolBar->addAction(actionLoadProjectTB);
    actionLoadProjectTB->setToolTip("Select a folder containing the images you want to treat");
    QObject::connect(actionLoadProjectTB, SIGNAL(triggered(bool)), this, SLOT(loadProject()));
    toolBar->addSeparator();

    QAction *actionSELTB = toolBar->addAction("&SEL");
    toolBar->addAction(actionSELTB);
    actionSELTB->setToolTip("Tie points visualization");
    QObject::connect(actionSELTB, SIGNAL(triggered(bool)), this, SLOT(sel()));

    QAction *actionAperiTB = toolBar->addAction("&Aperi");
    toolBar->addAction(actionAperiTB);
    actionAperiTB->setToolTip("Generate a visualization of camera position");
    QObject::connect(actionAperiTB, SIGNAL(triggered(bool)), this, SLOT(aperi()));

    QAction *actionMeshlabTB = toolBar->addAction("&Meshlab");
    toolBar->addAction(actionMeshlabTB);
    actionMeshlabTB->setToolTip("Open a file with Meshlab");
    QObject::connect(actionMeshlabTB, SIGNAL(triggered(bool)), this, SLOT(meshlab()));

    QAction *actionVinoTB = toolBar->addAction("&Vino");
    toolBar->addAction(actionVinoTB);
    actionVinoTB->setToolTip("Image visualization");
    QObject::connect(actionVinoTB, SIGNAL(triggered(bool)), this, SLOT(vino()));

    toolBar->addSeparator();
    
    QAction *actionBasculeTB = toolBar->addAction("&Bascule");
    toolBar->addAction(actionBasculeTB);
    actionBasculeTB->setToolTip("Transform the global orientation");
    QObject::connect(actionBasculeTB, SIGNAL(triggered(bool)), this, SLOT(bascule()));

    QAction *actionSaisiePtsTB = toolBar->addAction("&SaisiePts");
    toolBar->addAction(actionSaisiePtsTB);
    actionSaisiePtsTB->setToolTip("Set 2D coordinates");
    QObject::connect(actionSaisiePtsTB, SIGNAL(triggered(bool)), this, SLOT(saisiePts()));
    
    
    toolBar->addSeparator();

    
    QAction *actionCommandeTB = toolBar->addAction("&Commande");
    toolBar->addAction(actionCommandeTB);
    actionCommandeTB->setToolTip("Manual command launcher");
    QObject::connect(actionCommandeTB, SIGNAL(triggered(bool)), this, SLOT(cmd()));
    
    
    toolBar->addSeparator();
    
    
    QAction *actionTaramaTB = toolBar->addAction("&Tarama");
    toolBar->addAction(actionTaramaTB);
    actionTaramaTB->setToolTip("Compute a rectified image");
    QObject::connect(actionTaramaTB, SIGNAL(triggered(bool)), this, SLOT(tarama()));
    
    QAction *actionSaisieMasqTB = toolBar->addAction("&SaisieMasq");
    toolBar->addAction(actionSaisieMasqTB);
    actionSaisieMasqTB->setToolTip("Create a mask upon a selected image");
    QObject::connect(actionSaisieMasqTB, SIGNAL(triggered(bool)), this, SLOT(saisieMasq()));
    
    QAction *actionto8bitsTB = toolBar->addAction("&to8Bits");
    toolBar->addAction(actionto8bitsTB);
    actionto8bitsTB->setToolTip("Convert a 16 or 32 bit image in an 8 bit image");
    QObject::connect(actionto8bitsTB, SIGNAL(triggered(bool)), this, SLOT(to8Bits()));
    
    QAction *actionGrShadeTB = toolBar->addAction("&GrShade ");
    toolBar->addAction(actionGrShadeTB);
    actionGrShadeTB->setToolTip("Compute shading from depth image");
    QObject::connect(actionGrShadeTB, SIGNAL(triggered(bool)), this, SLOT(grShade()));

    QAction *actionmm2dPosSismTB = toolBar->addAction("&mm2dPosSism ");
    toolBar->addAction(actionmm2dPosSismTB);
    actionmm2dPosSismTB->setToolTip("Search for homolgous pixels and highilghts earth displacements");
    QObject::connect(actionmm2dPosSismTB, SIGNAL(triggered(bool)), this, SLOT(mm2dPosSim()));
    
    toolBar->addSeparator();

    this->addToolBarBreak();
    QToolBar *toolBar2 = addToolBar("ToolBar");
    this->addToolBar(toolBar2);

    QAction *actionExportConsoleTB = toolBar2->addAction("&ExportConsole");
    toolBar2->addAction(actionExportConsoleTB);
    actionExportConsoleTB->setToolTip("Export console contents");
    QObject::connect(actionExportConsoleTB, SIGNAL(triggered(bool)), this, SLOT(expor()));

    QAction *actionChSysTB = toolBar2->addAction("&ChSys");
    toolBar2->addAction(actionChSysTB);
    actionChSysTB->setToolTip("ChSys file generation");
    QObject::connect(actionChSysTB, SIGNAL(triggered(bool)), this, SLOT(chsys()));

    QAction *actionCropRPCTB = toolBar2->addAction("&CropRPC");
    toolBar2->addAction(actionCropRPCTB);
    actionCropRPCTB->setToolTip("Crop generation");
    QObject::connect(actionCropRPCTB, SIGNAL(triggered(bool)), this, SLOT(croprpc()));

}

void MainWindow::createCentralWidget()
{

    zoneCentrale = new QWidget;
    QPalette pal = palette();

    // set black background
    pal.setColor(QPalette::Background, Qt::white);
    zoneCentrale->setAutoFillBackground(true);
    zoneCentrale->setPalette(pal);
    zoneCentrale->show();

    //    zoneCentrale->setMinimumWidth(1000);
    QVBoxLayout* l = new QVBoxLayout;
    zoneCentrale->setLayout(l);

    onglets = new QTabWidget(zoneCentrale);
    onglets->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    ongletConsole = new QWidget;
    onglets->addTab(ongletConsole, "Console");
    lab = new QTextEdit;
    QVBoxLayout* VBox = new QVBoxLayout;
    ongletConsole->setLayout(VBox);
    VBox->addWidget(lab);



    ongletFileReader = new QWidget;
    onglets->addTab(ongletFileReader, "File Reader");
    lab2 = new QTextEdit;
    QVBoxLayout* VBox2 = new QVBoxLayout;
    ongletFileReader->setLayout(VBox2);
    VBox2->addWidget(lab2);

    ongletVISU = new QWidget;
    onglets->addTab(ongletVISU, "VISU");


    onglets1 = new QTabWidget(zoneCentrale);
    onglets1->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
    ongletPhotos = new QListWidget;
    onglets1->addTab(ongletPhotos, "Photos");
    ongletPhotos->setViewMode(QListWidget::IconMode);
    ongletPhotos->setIconSize(QSize(192,192));
    ongletPhotos->setResizeMode(QListWidget::Adjust);


    QSplitter *split1 = new QSplitter;
    split1->setOrientation(Qt::Vertical);
    split1->addWidget(onglets);
    split1->addWidget(onglets1);
    l->addWidget(split1);

    
    if(!listFileFoldersProject.empty()){
        
        std::string pathimages;
        pathimages=pathProject.toStdString()+"/listeimages.xml";
        
        FILE * fp = fopen(pathimages.c_str() , "rb");
        if(fp == NULL)
        { std::cout << "Empty in Dock";}
        else{
            cListOfName aL0 =  StdGetFromPCP(pathProject.toStdString()+"/listeimages.xml",ListOfName);
            std::list< std::string > aLS = aL0.Name();
            std::list< std::string >::iterator it;
            
            for(it = aLS.begin(); it!=aLS.end(); ++it)
            {

                std::string itToStrg = *it;
                imgIcone = pathProject.toStdString() + "/icone/" + itToStrg+".jpg";
                item = new QListWidgetItem(QIcon(imgIcone.c_str()),itToStrg.c_str());
                item->setFlags(item->flags() | Qt::ItemIsUserCheckable); // set checkable flag

                cListOfName aL0C =  StdGetFromPCP(pathProject.toStdString()+"/listeimageschecked.xml",ListOfName);
                std::list< std::string > aLSC = aL0C.Name();
                std::list< std::string >::iterator itC;

                if(!aLSC.empty()){
                    for(itC = aLSC.begin(); itC!=aLSC.end(); ++itC)
                    {
                        std::string itToStrgC = *itC;
                        if(itToStrgC==itToStrg){
                            item->setCheckState(Qt::Checked);
                            item->setBackgroundColor(QColor(178, 255, 102, 200));
                            item->setTextColor(QColor(0, 0, 0, 255));
                            break;
                        }else{
                            item->setCheckState(Qt::Unchecked);
                            item->setBackgroundColor(QColor(255, 0, 0, 170));
                            item->setTextColor(QColor(255, 255, 255, 255));

                        }
                    }
                }
                else
                {
                    item->setCheckState(Qt::Unchecked);
                    item->setBackgroundColor(QColor(255, 0, 0, 170));
                    item->setTextColor(QColor(255, 255, 255, 255));

                }
                ongletPhotos->setDragEnabled(false);
                ongletPhotos->addItem(item);
                ongletPhotos->setSpacing(10);
           }
            
            bool checked1;
            checkedItems.clear();
            
            
            for (int i = 0; i < ongletPhotos->count(); i++){
                checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
                if(checked1==1){
                    checkedItems.append(ongletPhotos->item(i)->text());}
                else{
                    checkedItems.append("NONESELECTEDITEM");}}
            
            
            setCentralWidget(zoneCentrale);
            
        }
        
        
        
    }else{
        lab->setText("PUT INSTRUCTIONS "
                     "\n Instru 1"
                     "\n Instru 2"
                     "\n Instru 3"
                     "\n Instru 4"
                     "\n Instru 5"
                     "\n Instru 6"
                     "\n Instru 7"
                     "\n Instru 8");
                     
                     
                     setCentralWidget(zoneCentrale);}
    connect(ongletPhotos, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(doSomething2(QListWidgetItem*)));
    connect(ongletPhotos, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(doSomething(QListWidgetItem*)));
    ongletPhotos->setContextMenuPolicy((Qt::CustomContextMenu));
    connect(ongletPhotos, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(showContextMenu(QPoint)));

    //  connect(ongletPhotos, SIGNAL(currentItemChanged(QListWidgetItem*,QListWidgetItem*)), this, SLOT(doSomethingCentral(QListWidgetItem*)));
    

}


// OUVERTURE DES FENETRES DE FONCTIONS



void MainWindow::tapioca(){

    qDebug() << listFileFoldersProject;
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        bool checked1;
        checkedItems.clear();
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        Tapioca *tapio = new Tapioca(listFileFoldersProject, checkedItems,pathProject);
        tapio->show();
        tapio->isActiveWindow();
        
        
        while( tapio->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        while(tapio->qDi.isVisible() ){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        
        word = tapio->msg1();
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);
        }
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Tapioca Error");
        msgBox1.setText("You can't launch Tapioca without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }
}

void MainWindow::tapas(){

    std::string OriResidu;

    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        Tapas *tapas = new Tapas(listFileFoldersProject, checkedItems, pathProject);
        tapas->show();
        while(tapas->qDi.isVisible() || tapas->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
            
        }
        // Ligne à ajouter dans toutes les fonctions qui changeront le résidu
        OriResidu = tapas->sendOri();
        //+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°°°+°+°+°+°+°+°+°+°°

        word = tapas->msg1();
        
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            // Ligne à ajouter dans toutes les fonctions qui changeront le résidu
            writeResiduOri(OriResidu);
            //+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°+°°°+°+°+°+°+°+°+°+°°

            onglets->setCurrentWidget(ongletConsole);
        }











    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Tapas Error");
        msgBox1.setText("You can't launch Tapas without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::campari(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        Campari *campari = new Campari(listFileFoldersProject_all, listFileFoldersProject, checkedItems, pathProject);
        campari->show();
        while(campari->qDi.isVisible() || campari->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);}
        word = campari->msg1();
        
        
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);

        }    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Campari Error");
        msgBox1.setText("You can't launch Campari without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::bascule(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        bool checked1;
        checkedItems.clear();


        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}


        Bascule *bascule = new Bascule(listFileFoldersProject_all, listFileFoldersProject, checkedItems, pathProject);
        bascule->show();
        while(bascule->qDi.isVisible() || bascule->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);}
        word = bascule->msg1();


        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);

        }    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Bascule Error");
        msgBox1.setText("You can't launch Bascule without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::malt(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        
        
        Malt *malt = new Malt(listFileFoldersProject_all, listFileFoldersProject, checkedItems, pathProject);
        malt->show();
        while(malt->qDi.isVisible() || malt->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);}
        
        word = malt->msg1();
        
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);

        }
    }
    else{
        QMessageBox msgBox1;
        msgBox1.setWindowTitle("Malt Error");
        msgBox1.setText("You can't launch Malt without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }
}

void MainWindow::saisiePts(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        bool checked1;
        checkedItems.clear();


        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}




        SaisiePts *saisie = new SaisiePts(listFileFoldersProject_all, listFileFoldersProject, checkedItems, pathProject);
        saisie->show();
        while(saisie->qDi.isVisible() || saisie->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);}

        word = saisie->msg1();

        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);

        }
    }
    else{
        QMessageBox msgBox1;
        msgBox1.setWindowTitle("saisiePts Error");
        msgBox1.setText("You can't launch saisiePtsInit without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }
}

void MainWindow::sel(){
    
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        
        
        Sel *sel = new Sel(listFileFoldersProject, checkedItems,pathProject);
        sel->show();
        while(sel->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        
        word = sel->msg1();
        
        if(word!=""){lab->setText(word);
            dock->deleteLater();
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);
        }
        
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Sel Error");
        msgBox1.setText("You can't launch Sel without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::aperi(){
    
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        
        
        Aperi *aperi = new Aperi(listFileFoldersProject_all, listFileFoldersProject, checkedItems,pathProject);
        aperi->show();
        aperi->isActiveWindow();


        while( aperi->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        while(aperi->qDi.isVisible() ){

            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        
        word = aperi->msg1();
        
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);
        }
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("AperiCloud Error");
        msgBox1.setText("You can't launch AperiCloud without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::meshlab(){

    qDebug() << "meshlab";

    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        otherPath = QFileDialog::getOpenFileName(0, ("Select Output Folder"),  pathProject.toStdString().c_str(),
                                                 "(*.ply)");
        QString cmd;
        QString cmd2;
        QString cmd1;



        if(otherPath!=""){
            if(killProc!=""){
                cmd2="kill "+QString::fromStdString(killProc);
                Proc3.startDetached("sh", QStringList()<<"-c"<< cmd2);
                qDebug() << cmd2;
                Proc3.start(cmd2);
                Proc3.close();

                Proc1.close();
                Proc2.close();
                Proc.close();
            }

            cmd = "meshlab "+otherPath;

            Proc.start(cmd);

            int num = Proc.pid();

            stringstream ss;
            ss << num;
            string str = ss.str();
            std::cout << str << std::endl;
            killProc = str;
            cmd1 = "xwininfo -root -tree | grep MeshLab";

            usleep(1000000); // TROUVER UN AUTRE MOYEN // mettre un while de tant qu'on a pas trouvé meshlab ...

            Proc1.start("sh", QStringList()<<"-c"<< cmd1);
            Proc1.waitForFinished(); // sets current thread to sleep and waits for pingProcess end
            QString output1(Proc1.readAllStandardOutput());
            output=output1;
            if(output.isEmpty()){qDebug() << "MeshLab not installed";

                QMessageBox msgBox1;
                        msgBox1.setWindowTitle("Error MeshLab");
                        msgBox1.setText("MeshLab is not installed. If you want to use this function, please, install MeshLab first.");
                        msgBox1.setStandardButtons(QMessageBox::Ok);
                        msgBox1.setDefaultButton(QMessageBox::Ok);
                        if(msgBox1.exec() == QMessageBox::Ok){
                        }
            }
            else{
                qDebug() << output;
                Proc1.kill();



                str = output1.toStdString();
                std::cout << "NOT CUT "+str +" NOT CUT " << std::endl;

                str = str.substr(13,7);
                std::cout << "YES CUT "+str +" YES CUT " << std::endl;

                std::cout << str << std::endl;
                std::stringstream str2;
                str2 << str;

                int value;
                str2 >> std::hex >> value;
                std::cout << value << std::endl;

                window =  QWindow::fromWinId(value);
                QWidget *widget = QWidget::createWindowContainer(window);
                vboxx = new QVBoxLayout();
                vboxx->addWidget(widget);
                ongletVISU->deleteLater();
                onglets->setCurrentWidget(ongletVISU);
                ongletVISU = new QWidget;
                onglets->addTab(ongletVISU, "VISU");
                ongletVISU->setLayout(vboxx);

                dockUpdate();

            }


        }
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Meshlab Error");
        msgBox1.setText("You can't launch Meshlab without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }}

void MainWindow::meshlab2(QString otherPath2){

    qDebug() << "meshlab2";

    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        QString cmd;
        QString cmd2;
        QString cmd1;

        qDebug() << Proc1.pid();
        qDebug() << Proc2.pid();
        qDebug() << Proc3.pid();
        qDebug() << Proc.pid();


        if(otherPath2!=""){
            if(killProc!=""){
                cmd2="kill "+QString::fromStdString(killProc);
                Proc3.startDetached("sh", QStringList()<<"-c"<< cmd2);
                qDebug() << cmd2;
                Proc3.start(cmd2);
                Proc1.kill();
                Proc2.kill();
                Proc3.kill();
                Proc.kill();
                qDebug() << Proc1.pid();
                qDebug() << Proc2.pid();
                qDebug() << Proc3.pid();
                qDebug() << Proc.pid();
                Proc3.close();
                Proc1.close();
                Proc2.close();
                Proc.close();

            }
            //            usleep(500000); // TROUVER UN AUTRE MOYEN // mettre un while de tant qu'on a pas trouvé meshlab ...

            //            QStringList ListVE=QProcess::systemEnvironment();
            //            qDebug()<<ListVE;
            cmd = "meshlab "+otherPath2;
            Proc.start(cmd);
            usleep(1000000);

            int num = Proc.pid();

            stringstream ss;
            ss << num;
            string str = ss.str();
            std::cout << str << std::endl;
            killProc = str;
            cmd1 = "xwininfo -root -tree | grep MeshLab";

            // TROUVER UN AUTRE MOYEN // mettre un while de tant qu'on a pas trouvé meshlab ...

            Proc1.start("sh", QStringList()<<"-c"<< cmd1);
            Proc1.waitForFinished(); // sets current thread to sleep and waits for pingProcess end
            QString output1(Proc1.readAllStandardOutput());
            output=output1;
            if(output.isEmpty()){qDebug() << "MeshLab not installed";
                QMessageBox msgBox1;
                        msgBox1.setWindowTitle("Error MeshLab");
                        msgBox1.setText("MeshLab is not installed. If you want to use this function, please, install MeshLab first.");
                        msgBox1.setStandardButtons(QMessageBox::Ok);
                        msgBox1.setDefaultButton(QMessageBox::Ok);
                        if(msgBox1.exec() == QMessageBox::Ok){
                        }

            }
            else{
            qDebug() << output;
            //            connect(&Proc1,SIGbugNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
            //            connect(&Proc1, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));  //Add this line will cause error at runtime
            Proc1.kill();
            qDebug() << " F I N I S H E D";



            str = output1.toStdString();
            std::cout << "NOT CUT "+str +" NOT CUT " << std::endl;

            str = str.substr(13,7);
            std::cout << "YES CUT "+str +" YES CUT " << std::endl;

            std::cout << str << std::endl;
            std::stringstream str2;
            str2 << str;

            int value;
            str2 >> std::hex >> value;
            std::cout << value << std::endl;

            window =  QWindow::fromWinId(value);

            QWidget *widget = QWidget::createWindowContainer(window);
            vboxx = new QVBoxLayout();
            vboxx->addWidget(widget);
            ongletVISU->deleteLater();
            ongletVISU = new QWidget;
            onglets->addTab(ongletVISU, "VISU");

            onglets->setCurrentWidget(ongletVISU);

            ongletVISU->setLayout(vboxx);





            dockUpdate();
            }
        }
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Meshlab Error");
        msgBox1.setText("You can't launch Meshlab without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }}
void MainWindow::vino2(QString otherPath2){

    qDebug() << "vino2";

    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        QString cmd;
        QString cmd2;
        QString cmd1;



        if(otherPath2!=""){
            if(killProc!=""){
                cmd2="kill "+QString::fromStdString(killProc);
                Proc3.startDetached("sh", QStringList()<<"-c"<< cmd2);
                qDebug() << cmd2;
                Proc3.start(cmd2);
                Proc3.close();

                Proc1.close();
                Proc2.close();
                Proc.close();
            }


            cmd = "mm3d Vino "+otherPath2 ;
            Proc.start(cmd);



        }
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Meshlab Error");
        msgBox1.setText("You can't launch Meshlab without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }}

void MainWindow::cmd(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();



        bool checked1;
        checkedItems.clear();


        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}




        Cmd *cmd = new Cmd(pathProject);
        cmd->show();
        cmd->isActiveWindow();


        while( cmd->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        while(cmd->qDi.isVisible() ){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }

        word = cmd->msg1();
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);

        }


    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Cmd Error");
        msgBox1.setText("You can't launch cmd without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::c3dc(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        C3dc *c3dc = new C3dc(listFileFoldersProject_all, listFileFoldersProject, checkedItems, pathProject);
        c3dc->show();
        while(c3dc->qDi.isVisible() || c3dc->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
            
        }
        word = c3dc->msg1();
        
        if(word!=""){
            lab->setText(word);
            dockUpdate();
            onglets->setCurrentWidget(ongletConsole);
        }
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("C3dc Error");
        msgBox1.setText("You can't launch C3dc without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::tarama()
{    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        
        
        Tarama *tarama = new Tarama(listFileFoldersProject_all,listFileFoldersProject, checkedItems, pathProject);
        tarama->show();
        while(tarama->qDi.isVisible() || tarama->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);}
        
        word = tarama->msg1();
        
        if(word!=""){
            lab->setText(word);
            refreshInterface();
            onglets->setCurrentWidget(ongletConsole);
        }
    }
    else{
        QMessageBox msgBox1;
        msgBox1.setWindowTitle("Tarama Error");
        msgBox1.setText("You can't launch Tarama without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }
}

void MainWindow::saisieMasq()
{
    
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();


        otherPath = QFileDialog::getOpenFileName(0, ("Select Output Folder"),  pathProject.toStdString().c_str(),
                                                 "Images (*.png *.xpm *.jpg *.tif *.JPG *.TIF *.PNG)");
        QString cmd;

        if(otherPath!=""){

            cmd = "mm3d SaisieMasq "+otherPath+ " @ExitOnBrkp";

            p.start(cmd);
            p.waitForFinished(-1);

            p.kill();
            dockUpdate();
        }

    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("saisieMasq Error");
        msgBox1.setText("You can't launch saisieMasq without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }}

void MainWindow::to8Bits()
{
    if(!listFileFoldersProject.empty()){

        listFileFoldersProject_update();
        otherPath = QFileDialog::getOpenFileName(0, ("Select Output Folder"),  pathProject.toStdString().c_str(),
                                                 "Images (*.png *.xpm *.jpg *.tif *.JPG *.TIF *.PNG)");
        QString cmd;

        if(otherPath!=""){
            cmd = "mm3d to8Bits "+otherPath+ " @ExitOnBrkp";

            p.start(cmd);
            p.waitForFinished(-1);
            p.kill();
            dockUpdate();
        }
        
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("to8Bits Error");
        msgBox1.setText("You can't launch to8Bits without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }


}

void MainWindow::grShade()
{
    
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();


        otherPath = QFileDialog::getOpenFileName(0, ("Select Output Folder"),  pathProject.toStdString().c_str(),
                                                 "Images (*.png *.xpm *.jpg *.tif *.JPG *.TIF *.PNG)");

        if(otherPath!="")
        {

            GrShade *grshade = new GrShade(otherPath);



            while( grshade->isVisible() || grshade->qDi.isVisible()){
                qApp->processEvents(QEventLoop::WaitForMoreEvents);
            }

            word = grshade->msg1();

            if(word!=""){
                lab->setText(word);
                dockUpdate();
            }
            onglets->setCurrentWidget(ongletConsole);
        }





    }

    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("grShade Error");
        msgBox1.setText("You can't launch grShade without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }

    
}

void MainWindow::vino()
{
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();


        otherPath = QFileDialog::getOpenFileName(0, ("Select Output Folder"),  pathProject.toStdString().c_str(),
                                                 "Images (*.png *.xpm *.jpg *.tif *.JPG *.TIF *.PNG)");
        QString cmd;



        cmd = "mm3d Vino "+otherPath+ " @ExitOnBrkp";

        p.start(cmd);
        p.waitForFinished(-1);

        p.kill();
        dockUpdate();
        qDebug()<<cmd;

    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Vino Error");
        msgBox1.setText("You can't launch Vino without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }


    }}

void MainWindow::expor(){
    if(!listFileFoldersProject.empty() && lab->toPlainText()!=""){
        listFileFoldersProject_update();
        
        QQ = new QWidget;
        // QQ->setMinimumHeight(40);
        //QQ->setMinimumWidth(40);
        QVBoxLayout *boxLayoutV = new QVBoxLayout;
        QLabel *name = new QLabel;
        name->setText("Enter the name you want to give to your exported file :");
        boxLayoutV->addWidget(name);
        Edit = new QTextEdit;
        Edit->setFixedHeight(25);
        boxLayoutV->addWidget(Edit);
        QPushButton *QOk = new QPushButton();
        QOk->setFixedWidth(50);
        QOk->setText("Ok");
        connect(QOk, SIGNAL(clicked()), this, SLOT(confirm()));
        boxLayoutV->addWidget(QOk);
        QQ->setLayout(boxLayoutV);
        QQ->show();
    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("Export Error");
        msgBox1.setText("Your console is empty.");
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::convert(){
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();
        
        bool checked1;
        checkedItems.clear();
        
        
        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}
        
        
        
        
        Convert2GenBundle *conv = new Convert2GenBundle(listFileFoldersProject_all, listFileFoldersProject, checkedItems, pathProject);
        conv->show();
        while(conv->qDi.isVisible() || conv->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);}
        
        word = conv->msg1();
        
        if(word!=""){
            lab->setText(word);
            dockUpdate();

            onglets->setCurrentWidget(ongletConsole);
        }
    }
    else{
        QMessageBox msgBox1;
        msgBox1.setWindowTitle("Convert2GenBundle Error");
        msgBox1.setText("You can't launch Convert2GenBundle without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }
}

void MainWindow::confirm(){
    
    if(Edit->toPlainText()!=""){
        listFileFoldersProject_update();
        
        QString words_expor = lab->toPlainText();
        QString name_file;
        name_file=Edit->toPlainText();
        std::string path_expor = pathProject.toStdString()+"/"+name_file.toStdString()+".txt";
        const char *path = path_expor.c_str()
                ;
        ofstream otherPath(path, ios::out | ios::trunc);  // ouverture en écriture avec effacement du pathProject ouvert
        if(otherPath)
        {
            
            otherPath << words_expor.toStdString();
            otherPath.close();
            QQ->close();
        }
        else{
            cerr << "Impossible d'ouvrir le pathProject !" << endl;}}
    else{
        
        QMessageBox msgBox1;
        msgBox1.setWindowTitle("Error");
        msgBox1.setText("ENTER A VALID NAME");
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }
    
    dockUpdate();
    
}

void MainWindow::loadProject()
{

    QString pathProjectSaved;
    pathProjectSaved=pathProject;
    pathProject = QFileDialog::getExistingDirectory(0, ("Select Output Folder"));



    if(!pathProject.isEmpty()){
        FILE * fp = fopen(pathProject.toStdString().c_str() , "rb");
        if(fp == NULL)
        {
            for (int y = 0; y < listFileFoldersProject.count(); y++)
            {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                        (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                    aList.push_back(listFileFoldersProject[y].toStdString());

                imToJpg(listFileFoldersProject[y].toStdString());

qDebug() << "YES";
            }

        }





        QMessageBox msgBox;
        msgBox.setWindowTitle("Load Project");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if(msgBox.exec() == QMessageBox::Yes){
            std::string pathWithSpaces;
            pathWithSpaces = pathProject.toStdString();
            if(pathWithSpaces.find(" ")<1000){
                msgBox.setWindowTitle("Error");
                msgBox.setText("Space(s) have been found int the path of the folder you selected. Please rename the foler or delete the spaces, then load it again \n ");
                msgBox.setStandardButtons(QMessageBox::Ok);
                msgBox.setDefaultButton(QMessageBox::Ok);
                if(msgBox.exec() == QMessageBox::Ok){
                }
            }else{


                pathWithSpaces = pathProject.toStdString();
                writePaths(pathProject);
                dockUpdate();
                Proc1.kill();
                Proc2.kill();
                Proc.kill();
                killProc ="";
                aList.clear();
                aListC.clear();
                nameresidu.clear();
                nameresidu2.clear();
                checkedItems.clear();
                std::string pathimages;
                pathimages=pathProject.toStdString()+"/listeimages.xml";

                std::string cmd = "mkdir "+pathWithSpaces+"/icone";

                p1.setWorkingDirectory(pathProject.toStdString().c_str());
                p1.waitForFinished(-1);
                p1.start(cmd.c_str());
                p1.waitForFinished(-1);
                p1.kill();


                FILE * fp = fopen(pathimages.c_str() , "rb");
                if(fp == NULL)
                {
                    for (int y = 0; y < listFileFoldersProject.count(); y++)
                    {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                            aList.push_back(listFileFoldersProject[y].toStdString());

                        imToJpg(listFileFoldersProject[y].toStdString());


                    }

                    //Création d'un pathProject xml avec tous les noms des images
                    aLON.Name()=aList;
                    MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");

                }

                std::string pathimageschecked;
                pathimageschecked=pathProject.toStdString()+"/listeimageschecked.xml";


                FILE * fpc = fopen(pathimageschecked.c_str() , "rb");
                if(fpc == NULL)
                {
                    for (int y = 0; y < listFileFoldersProject.count(); y++)
                    {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                                (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                            aListC.push_back(listFileFoldersProject[y].toStdString());
                    }


                    //Création d'un pathProject xml avec tous les noms des images
                    aLONC.Name()=aListC;
                    MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");

                }

                else

                {



                }
                refreshInterface();

            } }else {

            pathProject=pathProjectSaved;
        }
    }else{
        pathProject=pathProjectSaved;

    }




}


// Fonctions qui permettent d'ouvrir rapidement les projets récents
void MainWindow::loadExistingProject1()
{
    pathProject = pathsActions[0];

    if(QFileInfo(pathProject).exists()){

    if(!pathProject.isEmpty()){


        QMessageBox msgBox;
        msgBox.setWindowTitle("Load Project");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if(msgBox.exec() == QMessageBox::Yes){

            refreshInterface();
            aList.clear();
            aListC.clear();
            nameresidu.clear();
            nameresidu2.clear();
            checkedItems.clear();
            std::string pathimages;
            pathimages=pathProject.toStdString()+"/listeimages.xml";


            std::string cmd = "mkdir "+pathProject.toStdString()+"/icone";
            std::cout << cmd +" this is the command" << std::endl;
            p1.setWorkingDirectory(pathProject.toStdString().c_str());
            p1.waitForFinished(-1);
            p1.start(cmd.c_str());
            p1.waitForFinished(-1);
            p1.kill();



            FILE * fp = fopen(pathimages.c_str() , "rb");
            if(fp == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aList.push_back(listFileFoldersProject[y].toStdString());
                    imToJpg(listFileFoldersProject[y].toStdString());


                }

                //Création d'un pathProject xml avec tous les noms des images
                aLON.Name()=aList;
                MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");

            }

            std::string pathimageschecked;
            pathimageschecked=pathProject.toStdString()+"/listeimageschecked.xml";


            FILE * fpc = fopen(pathimageschecked.c_str() , "rb");
            if(fpc == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aListC.push_back(listFileFoldersProject[y].toStdString());
                }


                //Création d'un pathProject xml avec tous les noms des images
                aLONC.Name()=aListC;
                MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");

            }

            else

            {



            }
            refreshInterface();
        }else {pathProject=QString::fromStdString(imgIcone);
        }
        writePaths(pathProject);

    }else{pathProject=QString::fromStdString(imgIcone);

    }

    }else{ QMessageBox msgBox1;
        msgBox1.setWindowTitle("Path error");
        msgBox1.setText("The Path " + pathProject +" that you selected is no longer available. Please, make sure that you didn't move or delete this folder. You can still load this project by using the Load Project button.' \n");


        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }}


}
void MainWindow::loadExistingProject2()
{
    pathProject = pathsActions[1];

        //The file exists

    if(QFileInfo(pathProject).exists()){

    if(!pathProject.isEmpty()){
        QMessageBox msgBox;
        msgBox.setWindowTitle("Load Project");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if(msgBox.exec() == QMessageBox::Yes){

            refreshInterface();
            aList.clear();
            aListC.clear();
            nameresidu.clear();
            nameresidu2.clear();
            checkedItems.clear();
            std::string pathimages;
            pathimages=pathProject.toStdString()+"/listeimages.xml";


            std::string cmd = "mkdir "+pathProject.toStdString()+"/icone";
            std::cout << cmd +" this is the command" << std::endl;
            p1.setWorkingDirectory(pathProject.toStdString().c_str());
            p1.waitForFinished(-1);
            p1.start(cmd.c_str());
            p1.waitForFinished(-1);
            p1.kill();



            FILE * fp = fopen(pathimages.c_str() , "rb");
            if(fp == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aList.push_back(listFileFoldersProject[y].toStdString());
                    imToJpg(listFileFoldersProject[y].toStdString());


                }

                //Création d'un pathProject xml avec tous les noms des images
                aLON.Name()=aList;
                MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");

            }

            std::string pathimageschecked;
            pathimageschecked=pathProject.toStdString()+"/listeimageschecked.xml";


            FILE * fpc = fopen(pathimageschecked.c_str() , "rb");
            if(fpc == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aListC.push_back(listFileFoldersProject[y].toStdString());
                }


                //Création d'un pathProject xml avec tous les noms des images
                aLONC.Name()=aListC;
                MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");

            }

            else

            {



            }
            refreshInterface();

        }else{pathProject=QString::fromStdString(imgIcone);
        }
        writePaths(pathProject);

    }else{pathProject=QString::fromStdString(imgIcone);

    }

}else{ QMessageBox msgBox1;
        msgBox1.setWindowTitle("Path error");
        msgBox1.setText("The Path " + pathProject +" that you selected is no longer available. Please, make sure that you didn't move or delete this folder. You can still load this project by using the Load Project button.' \n");
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }}



}
void MainWindow::loadExistingProject3()
{

    pathProject = pathsActions[2];
    if(QFileInfo(pathProject).exists()){

    if(!pathProject.isEmpty()){


        QMessageBox msgBox;
        msgBox.setWindowTitle("Load Project");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if(msgBox.exec() == QMessageBox::Yes){

            refreshInterface();
            aList.clear();
            aListC.clear();
            nameresidu.clear();
            nameresidu2.clear();
            checkedItems.clear();
            std::string pathimages;
            pathimages=pathProject.toStdString()+"/listeimages.xml";


            std::string cmd = "mkdir "+pathProject.toStdString()+"/icone";
            std::cout << cmd +" this is the command" << std::endl;
            p1.setWorkingDirectory(pathProject.toStdString().c_str());
            p1.waitForFinished(-1);
            p1.start(cmd.c_str());
            p1.waitForFinished(-1);
            p1.kill();



            FILE * fp = fopen(pathimages.c_str() , "rb");
            if(fp == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aList.push_back(listFileFoldersProject[y].toStdString());
                    imToJpg(listFileFoldersProject[y].toStdString());


                }

                //Création d'un pathProject xml avec tous les noms des images
                aLON.Name()=aList;
                MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");

            }

            std::string pathimageschecked;
            pathimageschecked=pathProject.toStdString()+"/listeimageschecked.xml";


            FILE * fpc = fopen(pathimageschecked.c_str() , "rb");
            if(fpc == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aListC.push_back(listFileFoldersProject[y].toStdString());
                }


                //Création d'un pathProject xml avec tous les noms des images
                aLONC.Name()=aListC;
                MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");

            }

            else

            {



            }
            refreshInterface();


        }else {pathProject=QString::fromStdString(imgIcone);
        }


        writePaths(pathProject);

    }else{pathProject=QString::fromStdString(imgIcone);

    }



    }else{ QMessageBox msgBox1;
        msgBox1.setWindowTitle("Path error");
        msgBox1.setText("The Path " + pathProject +" that you selected is no longer available. Please, make sure that you didn't move or delete this folder. You can still load this project by using the Load Project button.' \n");
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }}

}
void MainWindow::loadExistingProject4()
{

    pathProject = pathsActions[3];
    if(QFileInfo(pathProject).exists()){

    if(!pathProject.isEmpty()){


        QMessageBox msgBox;
        msgBox.setWindowTitle("Load Project");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if(msgBox.exec() == QMessageBox::Yes){

            refreshInterface();
            aList.clear();
            aListC.clear();
            nameresidu.clear();
            nameresidu2.clear();
            checkedItems.clear();
            std::string pathimages;
            pathimages=pathProject.toStdString()+"/listeimages.xml";


            std::string cmd = "mkdir "+pathProject.toStdString()+"/icone";
            std::cout << cmd +" this is the command" << std::endl;
            p1.setWorkingDirectory(pathProject.toStdString().c_str());
            p1.waitForFinished(-1);
            p1.start(cmd.c_str());
            p1.waitForFinished(-1);
            p1.kill();



            FILE * fp = fopen(pathimages.c_str() , "rb");
            if(fp == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aList.push_back(listFileFoldersProject[y].toStdString());
                    imToJpg(listFileFoldersProject[y].toStdString());


                }

                //Création d'un pathProject xml avec tous les noms des images
                aLON.Name()=aList;
                MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");

            }

            std::string pathimageschecked;
            pathimageschecked=pathProject.toStdString()+"/listeimageschecked.xml";


            FILE * fpc = fopen(pathimageschecked.c_str() , "rb");
            if(fpc == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aListC.push_back(listFileFoldersProject[y].toStdString());
                }


                //Création d'un pathProject xml avec tous les noms des images
                aLONC.Name()=aListC;
                MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");

            }

            else

            {



            }
            refreshInterface();


        }else {pathProject=QString::fromStdString(imgIcone);
        }

        writePaths(pathProject);

    }else{pathProject=QString::fromStdString(imgIcone);

    }



    }else{ QMessageBox msgBox1;
        msgBox1.setWindowTitle("Path error");
        msgBox1.setText("The Path " + pathProject +" that you selected is no longer available. Please, make sure that you didn't move or delete this folder. You can still load this project by using the Load Project button.' \n");
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }}


}
void MainWindow::loadExistingProject5()
{

    pathProject = pathsActions[4];
    if(QFileInfo(pathProject).exists()){

    if(!pathProject.isEmpty()){


        QMessageBox msgBox;
        msgBox.setWindowTitle("Load Project");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);

        if(msgBox.exec() == QMessageBox::Yes){

            refreshInterface();
            aList.clear();
            aListC.clear();
            nameresidu.clear();
            nameresidu2.clear();
            checkedItems.clear();
            std::string pathimages;
            pathimages=pathProject.toStdString()+"/listeimages.xml";


            std::string cmd = "mkdir "+pathProject.toStdString()+"/icone";
            std::cout << cmd +" this is the command" << std::endl;
            p1.setWorkingDirectory(pathProject.toStdString().c_str());
            p1.waitForFinished(-1);
            p1.start(cmd.c_str());
            p1.waitForFinished(-1);
            p1.kill();



            FILE * fp = fopen(pathimages.c_str() , "rb");
            if(fp == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aList.push_back(listFileFoldersProject[y].toStdString());
                    imToJpg(listFileFoldersProject[y].toStdString());


                }

                //Création d'un pathProject xml avec tous les noms des images
                aLON.Name()=aList;
                MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");

            }

            std::string pathimageschecked;
            pathimageschecked=pathProject.toStdString()+"/listeimageschecked.xml";


            FILE * fpc = fopen(pathimageschecked.c_str() , "rb");
            if(fpc == NULL)
            {
                for (int y = 0; y < listFileFoldersProject.count(); y++)
                {if((listFileFoldersProject[y].toStdString().find(".JPG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".JPEG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".jpeg")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".png")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".PNG")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".bmp")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".BMP")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".TIF")<10000) ||
                            (listFileFoldersProject[y].toStdString().find(".tif")<10000))
                        aListC.push_back(listFileFoldersProject[y].toStdString());
                }


                //Création d'un pathProject xml avec tous les noms des images
                aLONC.Name()=aListC;
                MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");

            }

            else

            {



            }
            refreshInterface();


        }else {pathProject=QString::fromStdString(imgIcone);
        }

        writePaths(pathProject);

    }else{pathProject=QString::fromStdString(imgIcone);

    }




}else{

        QMessageBox msgBox1;
                msgBox1.setWindowTitle("Path error");
                msgBox1.setText("The Path " + pathProject +" that you selected is no longer available. Please, make sure that you didn't move or delete this folder. You can still load this project by using the Load Project button.' \n");
                msgBox1.setStandardButtons(QMessageBox::Ok);
                msgBox1.setDefaultButton(QMessageBox::Ok);
                if(msgBox1.exec() == QMessageBox::Ok){
                }
    }

}


// Fonctions qui permettent d'ouvrir rapidement les projets récents (ouverture/écriture dans les fichiers)
void MainWindow::writePaths(QString pathToWrite){



    if(pathToWrite.toStdString()!=""){
        QString contenu0, contenu1, contenu2, contenu3, contenu4;

        if(pathToWrite!=pathsWritten[0] &&
                pathToWrite!=pathsWritten[1] &&
                pathToWrite!=pathsWritten[2] &&
                pathToWrite!=pathsWritten[3] &&
                pathToWrite!=pathsWritten[4] )
        {
            contenu0 = pathToWrite;
            contenu1 = pathsWritten[0];
            contenu2 = pathsWritten[1];
            contenu3 = pathsWritten[2];
            contenu4 = pathsWritten[3];
            pathsWritten.clear();

            pathsWritten.append(contenu0);
            pathsWritten.append(contenu1);
            pathsWritten.append(contenu2);
            pathsWritten.append(contenu3);
            pathsWritten.append(contenu4);

            std::string GI_M= GI_MicMacDir.toStdString()+ "/paths.txt";
            std::cout << GI_M << std::endl;

            ofstream fichier(GI_M.c_str(), ios::out | ios::trunc);

            if(fichier)
            {
                fichier << pathsWritten[0].toStdString();
                fichier << "\n";
                fichier << pathsWritten[1].toStdString();
                fichier << "\n";
                fichier << pathsWritten[2].toStdString();
                fichier << "\n";
                fichier << pathsWritten[3].toStdString();
                fichier << "\n";
                fichier << pathsWritten[4].toStdString();
                fichier << "\n";
                fichier.close();

            }
            else  // sinon
                cerr << "Erreur à l'ouverture !" << endl;}
        else
        { std::cout << "ALREADY EXISTING" << std::endl;}

    }}
QList<QString> MainWindow::readPaths(){

    //§!§!§changed§!§!§
    string contenu0, contenu1, contenu2, contenu3, contenu4;
    std::string GI_M = GI_MicMacDir.toStdString()+ "/paths.txt";

    ifstream fichier(GI_M.c_str(), ios::in);

    if(fichier)
    {

        fichier >> contenu0 >> contenu1 >> contenu2 >> contenu3 >> contenu4;
        fichier.close();

    }
    else
    {
        cerr << "Error à l'ouverture !" << endl;
    }
    QString contenu00 = QString::fromStdString(contenu0);
    QString contenu11 = QString::fromStdString(contenu1);
    QString contenu22 = QString::fromStdString(contenu2);
    QString contenu33 = QString::fromStdString(contenu3);
    QString contenu44 = QString::fromStdString(contenu4);


    pathsWritten.append(contenu00);
    pathsWritten.append(contenu11);
    pathsWritten.append(contenu22);
    pathsWritten.append(contenu33);
    pathsWritten.append(contenu44);
    qDebug() << pathsWritten;
    return pathsWritten;

}


// Fonction qui rafraichit le dock
void MainWindow::refreshInterface(){

    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    int widthdock = dock->width();
    int heightdock = dock->height();

    int widthonglets1 = onglets1->width();
    int heightonglets1 = onglets1->height();

    int widthzoneCentrale = zoneCentrale->width();
    int heighthzoneCentrale = zoneCentrale->height();

    int widthonglets = onglets->width();
    int heighthonglets = onglets->height();

    int widthongletVISU = ongletVISU->width();
    int heighthongletVISU = ongletVISU->height();

    int widthongletConsole = ongletConsole->width();
    int heighthongletConsole = ongletConsole->height();

    int widthongletFileReader = ongletFileReader->width();
    int heighthongletFileReader = ongletFileReader->height();

    int widthtreeViewImages = treeViewImages->width();
    int heighttreeViewImages =treeViewImages->height();

    int widthtreeViewFloders =treeViewFloders->width();
    int heighttreeViewFloders =treeViewFloders->height();

    int widthOngPhoto = ongletPhotos->width();
    int heightOngPhoto = ongletPhotos->height();


    dockUpdate();
    centralUpdate();

    dock->resize(widthdock,heightdock);
    onglets->resize(widthonglets,heighthonglets);
    onglets1->resize(widthonglets1,heightonglets1);
    zoneCentrale->resize(widthzoneCentrale,heighthzoneCentrale);
    ongletVISU->resize(widthongletVISU,heighthongletVISU);
    ongletConsole->resize(widthongletConsole,heighthongletConsole);
    ongletFileReader->resize(widthongletFileReader,heighthongletFileReader);
    ongletPhotos->resize(widthOngPhoto,heightOngPhoto);
    treeViewFloders->resize(widthtreeViewFloders,heighttreeViewFloders);
    treeViewImages->resize(widthtreeViewImages,heighttreeViewImages);


    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
}
void MainWindow::dockUpdate(){
    dock->deleteLater();
    createDock();
    lab->setText(word);}
void MainWindow::centralUpdate(){
    zoneCentrale->deleteLater();
    createCentralWidget();
}

// Fonctions qui peremttent de surligner les différents éléments du dock/photos
void MainWindow::doSomething(QListWidgetItem *itemE)

{

    qDebug() << "doSomething";

    std::string itemS = itemE->text().toStdString();
    
    for(int i = 0; i < numColumn; i++){
        QModelIndex index = treeViewImages->model()->index(i,0);
        QString imgCent = index.data().toString();
        std::string imgCCent = imgCent.toStdString();
        
        if(imgCCent==itemS){

            treeViewImages->selectionModel()->setCurrentIndex(index, QItemSelectionModel::SelectCurrent);
        }
        
    }
    
    
    bool checked = itemE->checkState() == Qt::Checked;
    if(checked==1){
        itemE->setBackgroundColor(QColor(255, 0, 0, 170));
        itemE->setCheckState(Qt::Unchecked);
        itemChecked.removeItemWidget(itemE);
        itemE->setTextColor(QColor(255, 255, 255, 255));
        
        
        cListOfName aL0C =  StdGetFromPCP(pathProject.toStdString()+"/listeimageschecked.xml",ListOfName);
        std::list< std::string > aLSC = aL0C.Name();
        
        aLSC.remove(itemS.c_str());
        
        aLONC.Name()=aLSC;
        MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");
        
        
        
        
        checked = 0;
        
    }else{
        itemE->setBackgroundColor(QColor(178, 255, 102, 200));
        itemE->setCheckState(Qt::Checked);
        itemChecked.addItem(itemE);
        itemE->setTextColor(QColor(0, 0, 0, 255));
        
        
        cListOfName aL0C =  StdGetFromPCP(pathProject.toStdString()+"/listeimageschecked.xml",ListOfName);
        std::list< std::string > aLSC = aL0C.Name();
        
        aLSC.push_back(itemS.c_str());
        
        aLONC.Name()=aLSC;
        MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");
        
        
        
        checked = 1;
        
    }
}

void MainWindow::doSomethingDock(QModelIndex itemE)
{
    qDebug() << "doSomethingDock";


    QList<QModelIndex> QM = treeViewImages->selectionModel()->selectedIndexes();
    QModelIndex qvv = QM.at(0);
    qDebug() << qvv;
    
    QString imgDock = qvv.data().toString();
    std::string imgDDock = imgDock.toStdString();
    
    if(!ongletPhotos->findItems(imgDDock.c_str(), Qt::MatchExactly).isEmpty())
    {
        a2List= ongletPhotos->findItems(imgDDock.c_str(), Qt::MatchExactly);
        QListWidgetItem* selected = a2List.at(0);
        
        qDebug() << selected->text();
        ongletPhotos->setCurrentItem(selected);
    }}

void MainWindow::doSomething2(QListWidgetItem *item)
{
    qDebug() << "doSomething2";


    qDebug() <<  pathProject ;
    p.setWorkingDirectory(pathProject);
    p.waitForFinished(-1);
    qDebug() <<  QDir::currentPath();
    qDebug()<< item->text();
    
    QString cmd = "mm3d Vino "+ pathProject +"/"+ item->text();
    p.start(cmd);
    p.waitForFinished(-1);
    p.kill();
    
    
}
void MainWindow::showContextMenu(const QPoint &pos)
{
    // Handle global position
    QPoint globalPos = ongletPhotos->mapToGlobal(pos);
    qDebug() << globalPos;
    // Create menu and insert some actions
    QMenu myMenu;
    myMenu.addAction("Vino", this, SLOT(addItem()));
    myMenu.addAction("GrShade",  this, SLOT(eraseItem()));

    // Show context menu at handling position
    myMenu.exec(globalPos);
}

void MainWindow::chsys()
{
    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        bool checked1;
        checkedItems.clear();


        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}




        ChSys *chsys = new ChSys(pathProject);
        chsys->show();
        while(chsys->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        dockUpdate();



    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("chsys Error");
        msgBox1.setText("You can't launch chsys without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }




}

void MainWindow::croprpc(){

    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();



        bool checked1;
        checkedItems.clear();


        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}




        CropRPC *croprpc = new CropRPC(listFileFoldersProject, checkedItems,pathProject);
        croprpc->show();


        while( croprpc->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }
        while(croprpc->qDi.isVisible() ){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }

        word = croprpc->msg1();
        if(word!=""){
            lab->setText(word);
            dock->deleteLater();
            createDock();
            onglets->setCurrentWidget(ongletConsole);

        }


    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("cropRPC Error");
        msgBox1.setText("You can't launch cropRPC without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }}

void MainWindow::mm2dPosSim(){

    if(!listFileFoldersProject.empty()){
        listFileFoldersProject_update();

        bool checked1;
        checkedItems.clear();


        for (int i = 0; i < ongletPhotos->count(); i++){
            checked1 = ongletPhotos->item(i)->checkState() == Qt::Checked;
            if(checked1==1){
                checkedItems.append(ongletPhotos->item(i)->text());}
            else{
                checkedItems.append("NONESELECTEDITEM");}}




        MM2DPosSism *mm2d = new MM2DPosSism(listFileFoldersProject, checkedItems,pathProject);
        mm2d->show();
        while(mm2d->isVisible()){
            qApp->processEvents(QEventLoop::WaitForMoreEvents);
        }

        word = mm2d->msg1();

        if(word!=""){lab->setText(word);
            refreshInterface();

            onglets->setCurrentWidget(ongletConsole);
        }

    }
    else
    {  QMessageBox msgBox1;
        msgBox1.setWindowTitle("mm2d Error");
        msgBox1.setText("You can't launch mm2d without having loaded a folder containing images before ! Please, select a folder. \n" + pathProject);
        msgBox1.setStandardButtons(QMessageBox::Ok);
        msgBox1.setDefaultButton(QMessageBox::Ok);
        if(msgBox1.exec() == QMessageBox::Ok){
        }
    }

}

void MainWindow::surbri()
{
    if(!listFileFoldersProject.empty()){
        QModelIndex index = treeViewImages->currentIndex();
        QVariant data = treeViewImages->model()->data(index);
        QString text = data.toString();
        std::string imgSup = text.toStdString();
        qDebug() << text;
        
        
        
        
        
        
        QMessageBox msgBox2;
        
        msgBox2.setWindowTitle("Delete file");
        
        msgBox2.setText("Here is the file/folder you're about to delete' : \n " + text );
        msgBox2.setStandardButtons(QMessageBox::Yes);
        msgBox2.addButton(QMessageBox::No);
        msgBox2.setDefaultButton(QMessageBox::No);
        if(msgBox2.exec() == QMessageBox::Yes){
            
            cListOfName aL0 =  StdGetFromPCP(pathProject.toStdString()+"/listeimages.xml",ListOfName);
            std::list< std::string > aLS = aL0.Name();
            
            aLS.remove(imgSup);
            
            
            cListOfName aL0C =  StdGetFromPCP(pathProject.toStdString()+"/listeimageschecked.xml",ListOfName);
            std::list< std::string > aLSC = aL0C.Name();
            
            aLSC.remove(imgSup);
            
            
            
            
            
            aLON.Name()=aLS;
            MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");
            
            aLONC.Name()=aLSC;
            MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");
            
            
            
            
            dockUpdate();
            
            createCentralWidget();
        }
        
        
        
        
        
        
        
    }}

void MainWindow::ajoutf()
{
    if(!listFileFoldersProject.empty()){
        otherPath = QFileDialog::getOpenFileName(0, ("Select Output Folder"));
        QString name;
        name=QFileInfo(otherPath).fileName();
        qDebug() << name;
        
        listFileFoldersProject_update();
        
        int dejaexistant=0;
        
        for(int a=0; a<listFileFoldersProject.size(); a+=1){
            
            if(name==listFileFoldersProject[a]){
                dejaexistant=1;}}
        
        if(dejaexistant==0){
            if(!otherPath.isEmpty()){
                QMessageBox msgBox;
                msgBox.setWindowTitle("title");
                msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + pathProject);
                msgBox.setStandardButtons(QMessageBox::Yes);
                msgBox.addButton(QMessageBox::No);
                msgBox.setDefaultButton(QMessageBox::No);
                if(msgBox.exec() == QMessageBox::Yes){
                    if((name.toStdString().find(".JPG")<10000) ||
                            (name.toStdString().find(".jpg")<10000) ||
                            (name.toStdString().find(".JPEG")<10000)||
                            (name.toStdString().find(".jpeg")<10000)||
                            (name.toStdString().find(".png")<10000) ||
                            (name.toStdString().find(".PNG")<10000) ||
                            (name.toStdString().find(".bmp")<10000) ||
                            (name.toStdString().find(".BMP")<10000) ||
                            (name.toStdString().find(".TIF")<10000) ||
                            (name.toStdString().find(".tif")<10000))
                    {
                        
                        aList.clear();
                        for (int y = 0; y < listFileFoldersProject.count(); y++)
                        {
                            aList.push_back(listFileFoldersProject[y].toStdString());
                        }


                        
                        aList.push_back(name.toStdString().c_str());
                        
                        //Création d'un pathProject xml avec tous les noms des images
                        aLON.Name()=aList;
                        MakeFileXML(aLON,pathProject.toStdString()+"/listeimages.xml");
                        
                        
                        
                        cListOfName aL0C =  StdGetFromPCP(pathProject.toStdString()+"/listeimageschecked.xml",ListOfName);
                        std::list< std::string > aLSC = aL0C.Name();
                        
                        aLSC.push_back(name.toStdString().c_str());
                        
                        aLONC.Name()=aLSC;
                        MakeFileXML(aLONC,pathProject.toStdString()+"/listeimageschecked.xml");
                        
                        
                        
                        
                        
                        dejaexistant=0;}
                    else{
                        std::cout << "déjà existin";
                    }}
                
            }
            
            
            else{
                
                QMessageBox msgBox2;
                
                msgBox2.setWindowTitle("Delete file");
                
                msgBox2.setText("This folder or file has already been loaded");
                msgBox2.setStandardButtons(QMessageBox::Ok);
                msgBox2.setDefaultButton(QMessageBox::Ok);
                if(msgBox2.exec() == QMessageBox::Ok){
                    
                    
                }
                
                
            }
            
            
            
            
        }}
    
    listFileFoldersProject_update();
    dockUpdate();
    createCentralWidget();
    
}

void MainWindow::ajoutF()
{
    if(!listFileFoldersProject.empty()){
        QString imgIcone = pathProject;
        otherPath = QFileDialog::getExistingDirectory(0, ("Select Output Folder"));
        QString cmd;
        
        cmd = "cp -R "+otherPath+" "+imgIcone;

        
        p.start(cmd);
        p.waitForFinished(-1);
        
        p.kill();
        dockUpdate();
        
    }}

void MainWindow::listFileFoldersProject_update(){
    
    qDebug() <<listFileFoldersProject;

    listFileFoldersProject.clear();
    
    
    cListOfName aL0 =  StdGetFromPCP(pathProject.toStdString()+"/listeimages.xml",ListOfName);
    std::list< std::string > aLS = aL0.Name();
    std::list< std::string >::iterator it;
    for(it = aLS.begin(); it!=aLS.end(); ++it)
    {
        std::string strcmp = *it;
        QString imgSupQ = QString::fromStdString(strcmp);
        
        
        listFileFoldersProject.append(imgSupQ);
        qDebug() <<listFileFoldersProject;

    }
}

//Creation des icones
void MainWindow::imToJpg(std::string img){
    std::string cmd = "convert "+img+" -quality 60 -resize 1000 -contrast-stretch 5% icone/"+img+".jpg";
    p1.setWorkingDirectory(pathProject.toStdString().c_str());
    p1.waitForFinished(-1);
    p1.start(cmd.c_str());
    p1.waitForFinished(-1);
    p1.kill();


}

void MainWindow::setTextFileReader(){




    QList<QModelIndex> QM = treeViewFloders->selectionModel()->selectedIndexes();
    QModelIndex qvv = QM.at(0);

    QString modelIndex = model->filePath(qvv);


    QString imgDock = qvv.data().toString();

    std::string imgDDock = imgDock.toStdString();


    std::string imgDDockPath = modelIndex.toStdString();

    std::string text;


    std::cout << imgDDockPath + "IIMGDOCK" << std::endl;

    ifstream otherPath(imgDDockPath.c_str());  // on ouvre en lecture
    if(otherPath && ((imgDDockPath.find(".txt")<10000) ||
                     (imgDDockPath.find(".xml")<10000) ||
                     (imgDDockPath.find(".TXT")<10000) ||
                     (imgDDockPath.find(".XML")<10000)))  // si l'ouverture a fonctionné

    {
        onglets->setCurrentWidget(ongletFileReader);
        std::string contenu;  // déclaration d'une chaîne qui contiendra la ligne lue
        while(getline(otherPath, contenu))  // tant que l'on peut mettre la ligne dans "contenu"

        {
            cout << contenu << endl;
            text.append(contenu + "\n");
        }
        otherPath.close();
        lab2->setText(text.c_str());


    }




    else if(otherPath && (imgDDock.find("Ori-")<10000))  // si l'ouverture a fonctionné

    {
        std::string OriWord = imgDDock.erase(0,4);
        std::cout << "ORI >"+ OriWord +"< ORI"  << std::endl;
        writeResiduOri(OriWord);
        dockUpdate();
    }

    else
    {
        cerr << "Impossible d'ouvrir le pathProject !" << endl;
    }




}

void MainWindow::vino2click(){
    qDebug() << "vino2click";




    QList<QModelIndex> QM = treeViewFloders->selectionModel()->selectedIndexes();
    QModelIndex qvv = QM.at(0);

    QString modelIndex = model->filePath(qvv);


    QString imgDock = qvv.data().toString();

    std::string imgDDock = imgDock.toStdString();


    std::string imgDDockPath = modelIndex.toStdString();
    std::cout << imgDDock;
    std::cout << imgDDockPath;
    std::string text;


    std::cout << imgDDockPath + "IIMGDOCK" << std::endl;

    ifstream otherPath(imgDDockPath.c_str());  // on ouvre en lecture
    if(otherPath && ((imgDDockPath.find(".JPG")<10000) ||
                     (imgDDockPath.find(".jpg")<10000) ||
                     (imgDDockPath.find(".JPEG")<10000) ||
                     (imgDDockPath.find(".jpeg")<10000) ||
                     (imgDDockPath.find(".png")<10000) ||
                     (imgDDockPath.find(".PNG")<10000) ||
                     (imgDDockPath.find(".bmp")<10000) ||
                     (imgDDockPath.find(".BMP")<10000) ||
                     (imgDDockPath.find(".TIF")<10000) ||
                     (imgDDockPath.find(".tif")<10000))
            && (imgDDockPath.find(".tif.xml")>10000))  // si l'ouverture a fonctionné

    {


        vino2(QString::fromStdString(imgDDockPath));

    }

    if(otherPath && ((imgDDockPath.find(".ply")<10000) ||
                     (imgDDockPath.find(".PLY")<10000)))  // si l'ouverture a fonctionné

    {


        meshlab2(QString::fromStdString(imgDDockPath));
    }

}

void MainWindow::findResidu(){
    qDebug() << "findResidu";

    std::string OriResidu=readResiduOri();
    //std::cout << OriResidu << std::endl;

    if(OriResidu!="")
    {
        QFile file(pathProject+"/Ori-"+OriResidu.c_str()+"/Residus.xml");
        int inc = 0;
        if(!file.open(QFile::ReadOnly | QFile::Text)){
            qDebug() << "Cannot read file" << file.errorString();
        }else{

            QXmlStreamReader reader(&file);

            if (reader.readNextStartElement()) {
                if (reader.name() == "XmlSauvExportAperoGlob"){
                    while(reader.readNextStartElement()){
                        if(reader.name() == "Iters"){
                            while(reader.readNextStartElement()){
                                if(reader.name() == "OneIm"){
                                    while(reader.readNextStartElement()){
                                        if(reader.name() == "Name"){
                                            QString s = reader.readElementText();
                                            //                                                       qDebug() << s + " Name";
                                            nameresidu.append(s);
                                            inc = inc+1;

                                        }

                                        else if(reader.name() == "Residual"){
                                            QString s2 = reader.readElementText();
                                            //                                                       qDebug() << s2 + " Residual";
                                            nameresidu.append(s2);
                                            inc = inc+1;
                                        }

                                        else if(reader.name() == "PercOk"){
                                            QString s3 = reader.readElementText();
                                            //                                                       qDebug() << s3 + " PercOk";
                                            nameresidu.append(s3);
                                            inc = inc+1;
                                        }

                                        else if(reader.name() == "NbPts"){
                                            QString s4 = reader.readElementText();
                                            //                                                       qDebug() << s4 + " NbPts";
                                            nameresidu.append(s4);
                                            inc = inc+1;
                                        }

                                        else if(reader.name() == "NbPtsMul"){
                                            QString s5 = reader.readElementText();
                                            //                                                       qDebug() << s5 + " NbPtsMul";
                                            nameresidu.append(s5);
                                            inc = inc+1;
                                        }


                                        else {
                                            reader.skipCurrentElement();
                                        }
                                    }

                                }
                                else {
                                    reader.skipCurrentElement();
                                }

                            }

                            nameresidu2=nameresidu;
                            nameresidu.clear();
                            inc=0;

                        }
                        else {
                            reader.skipCurrentElement();
                        }
                    }
                }
                else {
                    reader.skipCurrentElement();
                }

            }
            else {

                reader.skipCurrentElement();
            }

        }
    }
    else
    {


    }
}


void MainWindow::writeResiduOri(std::string OriResidu){

    qDebug() << "writeResiduOri";
    std::string pathResidu = pathProject.toStdString()+"/.residu.txt";
    ofstream txtFileResidu(pathResidu.c_str(), ios::out | ios::trunc);  // ouverture en écriture avec effacement du pathProject ouvert
    if(txtFileResidu)
    {

        txtFileResidu << OriResidu.c_str();
        txtFileResidu.close();
    }
    else{
        cerr << "Impossible d'ouvrir le pathProject !" << endl;
    }

    treeViewFloders->update();

}



std::string MainWindow::readResiduOri(){

    qDebug() << "readResiduOri";

    string contenu;  // déclaration d'une chaîne qui contiendra la ligne lue


    std::string pathResidu = pathProject.toStdString()+"/.residu.txt";
    ifstream fichier(pathResidu.c_str(), ios::in);  // on ouvre le fichier en lecture

    if(fichier)
    {

        getline(fichier, contenu);  // on met dans "contenu" la ligne


    }
    else
    {
        cerr << "Impossible d'ouvrir le fichier là !" << endl;
        contenu="";
    }

    return contenu;

}
