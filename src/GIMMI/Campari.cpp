#include "Campari.h"

Campari::Campari(QList <QString> list_all, QList <QString> list, QList <QString> listCheckedItems, QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    path_s = path.toStdString();
    setWindowTitle("Campari");
    qDebug()<< path;
    QVBoxLayout *boxlayoutv = new QVBoxLayout;


    QLabel *labelMode1 = new QLabel(this);
    labelMode1->setText("Input-Orientation :");
    labelMode1->setStyleSheet("font-weight: bold");



    QLabel *outPutCampari = new QLabel(this);
    outPutCampari->setText("Output folder :");
    outPutCampari->setStyleSheet("font-weight: bold");



    QLabel *labelOptions = new QLabel(this);
    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");



    QLabel *GCP = new QLabel(this);
    GCP->setText("GCP");
    GCPQt = new QTextEdit();
    QHBoxLayout *boxlayouth0 = new QHBoxLayout;
    boxlayouth0->addWidget(GCP);
    boxlayouth0->addWidget(GCPQt);
    GCPQt->setFixedWidth(300);
    GCPQt->setFixedHeight(27);

    QLabel *EmGPS = new QLabel(this);
    EmGPS->setText("EmGPS");
    EmGPSQt = new QTextEdit();
    QHBoxLayout *boxlayouth1 = new QHBoxLayout;
    boxlayouth1->addWidget(EmGPS);
    boxlayouth1->addWidget(EmGPSQt);
    EmGPSQt->setFixedWidth(300);
    EmGPSQt->setFixedHeight(27);


    QLabel *GpsLa = new QLabel(this);
    GpsLa->setText("GpsLa");
    GpsLaQt = new QTextEdit();
    GpsLaQt2 = new QTextEdit();
    GpsLaQt3 = new QTextEdit();
    QHBoxLayout *boxlayouth2 = new QHBoxLayout;
    QHBoxLayout *boxlayouth22 = new QHBoxLayout;
    boxlayouth2->addWidget(GpsLa);
    boxlayouth22->addWidget(GpsLaQt);
    boxlayouth22->addWidget(GpsLaQt2);
    boxlayouth22->addWidget(GpsLaQt3);
    GpsLaQt->setFixedWidth(98);
    GpsLaQt->setFixedHeight(27);
    GpsLaQt2->setFixedWidth(95);
    GpsLaQt2->setFixedHeight(27);
    GpsLaQt3->setFixedWidth(95);
    GpsLaQt3->setFixedHeight(27);
    boxlayouth2->addLayout(boxlayouth22);


    QLabel *SigmaTieP = new QLabel(this);
    SigmaTieP->setText("SigmaTieP");
    SigmaTiePQt = new QTextEdit();
    QHBoxLayout *boxlayouth3 = new QHBoxLayout;
    boxlayouth3->addWidget(SigmaTieP);
    boxlayouth3->addWidget(SigmaTiePQt);
    SigmaTiePQt->setFixedWidth(300);
    SigmaTiePQt->setFixedHeight(27);

    QLabel *FactElimTieP = new QLabel(this);
    FactElimTieP->setText("FactElimTieP");
    FactElimTiePQt = new QTextEdit();
    QHBoxLayout *boxlayouth4 = new QHBoxLayout;
    boxlayouth4->addWidget(FactElimTieP);
    boxlayouth4->addWidget(FactElimTiePQt);
    FactElimTiePQt->setFixedWidth(300);
    FactElimTiePQt->setFixedHeight(27);

    QLabel *CPI1 = new QLabel(this);
    CPI1->setText("CPI1");
    CPI1Qt = new QCheckBox();
    QHBoxLayout *boxlayouth5 = new QHBoxLayout;
    boxlayouth5->addWidget(CPI1);
    boxlayouth5->addWidget(CPI1Qt);
    CPI1Qt->setFixedWidth(300);
    CPI1Qt->setFixedHeight(27);

    QLabel *CPI2 = new QLabel(this);
    CPI2->setText("CPI2");
    CPI2Qt = new QCheckBox();
    QHBoxLayout *boxlayouth6 = new QHBoxLayout;
    boxlayouth6->addWidget(CPI2);
    boxlayouth6->addWidget(CPI2Qt);
    CPI2Qt->setFixedWidth(300);
    CPI2Qt->setFixedHeight(27);

    QLabel *FocFree = new QLabel(this);
    FocFree->setText("FocFree");
    FocFreeQt = new QCheckBox();
    QHBoxLayout *boxlayouth7 = new QHBoxLayout;
    boxlayouth7->addWidget(FocFree);
    boxlayouth7->addWidget(FocFreeQt);
    FocFreeQt->setFixedWidth(300);
    FocFreeQt->setFixedHeight(27);

    QLabel *PPFree = new QLabel(this);
    PPFree->setText("PPFree");
    PPFreeQt = new QCheckBox();
    QHBoxLayout *boxlayouth8 = new QHBoxLayout;
    boxlayouth8->addWidget(PPFree);
    boxlayouth8->addWidget(PPFreeQt);
    PPFreeQt->setFixedWidth(300);
    PPFreeQt->setFixedHeight(27);

    QLabel *AffineFree = new QLabel(this);
    AffineFree->setText("AffineFree");
    AffineFreeQt = new QCheckBox();
    QHBoxLayout *boxlayouth9 = new QHBoxLayout;
    boxlayouth9->addWidget(AffineFree);
    boxlayouth9->addWidget(AffineFreeQt);
    AffineFreeQt->setFixedWidth(300);
    AffineFreeQt->setFixedHeight(27);

    QLabel *AllFree = new QLabel(this);
    AllFree->setText("AllFree");
    AllFreeQt = new QCheckBox();
    QHBoxLayout *boxlayouth10 = new QHBoxLayout;
    boxlayouth10->addWidget(AllFree);
    boxlayouth10->addWidget(AllFreeQt);
    AllFreeQt->setFixedWidth(300);
    AllFreeQt->setFixedHeight(27);


    QLabel *ExpTxt = new QLabel(this);
    ExpTxt->setText("ExpTxt");
    ExpTxtQt = new QCheckBox();
    QHBoxLayout *boxlayouth11 = new QHBoxLayout;
    boxlayouth11->addWidget(ExpTxt);
    boxlayouth11->addWidget(ExpTxtQt);
    ExpTxtQt->setFixedWidth(300);
    ExpTxtQt->setFixedHeight(27);

    for (int i = 0; i < list.count(); i++)
    {
        std::string mot = list[i].toStdString();
        if((mot.find(".JPG")<10000) ||
                (mot.find(".jpg")<10000) ||
                (mot.find(".JPEG")<10000) ||
                (mot.find(".jpeg")<10000) ||
                (mot.find(".png")<10000) ||
                (mot.find(".PNG")<10000) ||
                (mot.find(".bmp")<10000) ||
                (mot.find(".BMP")<10000) ||
                (mot.find(".TIF")<10000) ||
                (mot.find(".tif")<10000)){
            listImages.append(list[i]);
        }
    }

    for (int i = 0; i < list_all.count(); i++)
    {
        std::string mot = list_all[i].toStdString();
        if(mot.find("Ori-")<10000){
            list2.append(list_all[i]);
            qDebug()<<list2;
        }}

    for (int i = 0; i < listImages.count(); i++)
    {
        imagesCheckBox = new QCheckBox(listImages[i]);
        listeCasesImages.append(imagesCheckBox);


        for (int y = 0; y < listImages.count(); y++)
            if(listImages[i]==listCheckedItems[y]){
                qDebug() << listImages[i] + "      listImages";
                qDebug() << listCheckedItems[i] + "listCheckedItems";
                imagesCheckBox->setChecked(true);
                y=listImages.count();
            }
    }








    groupBox1 = new QComboBox();

    for (int i = 0; i < list2.count(); i++)
    {
        groupBox1->addItem(list2[i]);
    }

    boxlayoutv->addWidget(labelMode1);
    boxlayoutv->addWidget(groupBox1);


    boxlayoutv->addWidget(outPutCampari);
    outPut = new QTextEdit();
    outPut->setFixedHeight(25);
    boxlayoutv->addWidget(outPut);

    boxlayoutv->addWidget(labelOptions);



    QScrollArea *scrollArea = new QScrollArea;
    scrollArea->setFixedSize(430, 300);
    QWidget *layoutWidget = new QWidget(this);
    QVBoxLayout *boxlayoutscroll = new QVBoxLayout;
    layoutWidget->setLayout(boxlayoutscroll);

    boxlayoutscroll->addLayout(boxlayouth0);
    boxlayoutscroll->addLayout(boxlayouth1);
    boxlayoutscroll->addLayout(boxlayouth2);
    boxlayoutscroll->addLayout(boxlayouth3);
    boxlayoutscroll->addLayout(boxlayouth4);
    boxlayoutscroll->addLayout(boxlayouth5);
    boxlayoutscroll->addLayout(boxlayouth6);
    boxlayoutscroll->addLayout(boxlayouth7);
    boxlayoutscroll->addLayout(boxlayouth8);
    boxlayoutscroll->addLayout(boxlayouth9);
    boxlayoutscroll->addLayout(boxlayouth10);
    boxlayoutscroll->addLayout(boxlayouth11);

    boxlayoutv->addWidget(scrollArea);
    scrollArea->setWidget(layoutWidget);
    scrollArea->show();

    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxlayoutv->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
    this->setLayout(boxlayoutv);
}


void Campari::mm3d(){

    int a = 0;
    images.erase();

    std::cout <<  "images = "  << std::endl;
    for( int i = 0; i< listeCasesImages.size(); i++){
        if(listeCasesImages.at(i)->isChecked())
        {a+=1;
            images.append(listImages.at(i).toStdString()+"|");
        }
    }


    if(a<2){
        msgBox2.setWindowTitle("Warning");
        msgBox2.setText("Veuillez sélectionner au moins deux images à traiter \n ");
        msgBox2.setStandardButtons(QMessageBox::Ok);
        msgBox2.setDefaultButton(QMessageBox::Ok);
        if(msgBox2.exec() == QMessageBox::Ok){
        }

    }else{
        images = images.erase(images.size()-1);

        std::cout <<  images << std::endl;
        std::cout <<  "mode = " + mode << std::endl;


        outPutStd = outPut->toPlainText().toStdString();
        std::cout <<  "Size M = " + outPutStd << std::endl;


        rPCStd = groupBox1->currentText().toStdString();
rPCStd.erase(0,4);


        if(GCPQt->toPlainText()!=""){
            GCPVar="GCP="+GCPQt->toPlainText().toStdString();
        }

        if(EmGPSQt->toPlainText()!=""){
            EmGPSVar="EmGPS="+EmGPSQt->toPlainText().toStdString();
        }


        if(GpsLaQt->toPlainText()!="" && GpsLaQt2->toPlainText()!="" && GpsLaQt3->toPlainText()!=""){
            GpsLaVar="GpsLaQ=["+GpsLaQt->toPlainText().toStdString()+";"+GpsLaQt2->toPlainText().toStdString()+";"+GpsLaQt3->toPlainText().toStdString()+"]";
        }

        if(SigmaTiePQt->toPlainText()!=""){
            SigmaTiePVar="SigmaTieP="+SigmaTiePQt->toPlainText().toStdString();
        }

        if(FactElimTiePQt->toPlainText()!=""){
            FactElimTiePVar="FactElimTieP="+FactElimTiePQt->toPlainText().toStdString();
        }

        if(CPI1Qt->isChecked()){
            CPI1Var="CPI1=1";
        }

        if(CPI2Qt->isChecked()){
            CPI2Var="CPI2=1";
        }

        if(FocFreeQt->isChecked()){
            FocFreeVar="FocFree=1";
        }

        if(PPFreeQt->isChecked()){
            PPFreeVar="PPFree=1";
        }

        if(AffineFreeQt->isChecked()){
            AffineFreeVar="AffineFree=1";
        }

        if(AllFreeQt->isChecked()){
            AllFreeVar="AllFree=1";
        }


        if(ExpTxtQt->isChecked()){
            ExpTxtVar="ExpTxt=1";
        }


        cmd = "mm3d Campari \""+ images +"\" "+ rPCStd +" "+ outPutStd +" "
                + GCPVar.c_str()+" "+
                + EmGPSVar.c_str()+" "+
                + GpsLaVar.c_str()+" "+
                + SigmaTiePVar.c_str()+" "+
                + FactElimTiePVar.c_str()+" "+
                + CPI1Var.c_str()+" "+
                + CPI2Var.c_str()+" "+
                + FocFreeVar.c_str()+" "+
                + PPFreeVar.c_str()+" "+
                + AffineFreeVar.c_str()+" "+
                + AllFreeVar.c_str()+" "+
                + ExpTxtVar.c_str()+" "+
                + " @ExitOnBrkp";
        std::cout <<  cmd << std::endl;




        QMessageBox msgBox2;
        msgBox2.setWindowTitle("Campari");
        const QString str = QString::fromStdString(cmd);
        msgBox2.setText("Here is the commande you're about to launch : \n " + str);
        msgBox2.setStandardButtons(QMessageBox::Yes);
        msgBox2.addButton(QMessageBox::No);
        msgBox2.setDefaultButton(QMessageBox::No);
        if(msgBox2.exec() == QMessageBox::Yes){
            this->close();
            p.setWorkingDirectory(path_s.c_str());
            p.waitForFinished(-1);
            qDebug() <<  QDir::currentPath();
            qDebug() <<  cmd.c_str() ;
            p.start(cmd.c_str());
            p.setReadChannel(QProcess::StandardOutput);
            connect(&p,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
            connect(&p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));  //Add this line will cause error at runtime
            msg();

        }else {
            images.clear();
            GCPVar=" ";
            EmGPSVar=" ";
            GpsLaVar=" ";
            GpsLa2Var=" ";
            GpsLa3Var=" ";
            SigmaTiePVar=" ";
            FactElimTiePVar=" ";
            CPI1Var=" ";
            CPI2Var=" ";
            FocFreeVar=" ";
            PPFreeVar=" ";
            AffineFreeVar=" ";
            AllFreeVar=" ";
            ExpTxtVar=" ";}


    }



}





QString Campari::getTxt(){
    return p_stdout;
}


void Campari::msg(){

    qDi.setWindowTitle("Campari state");

    QLabel *wait = new QLabel(this);
    wait->setText(" Wait until the end of the process, or click stop to cancel it ");
    wait->setStyleSheet("font-weight: bold");


    Qtr = new QTextEdit();
    Qtr->setFixedHeight(500);
    Qtr->setFixedWidth(700);




    QVBoxLayout *QV = new QVBoxLayout();
    QV->addWidget(wait);

    QV->addWidget(Qtr);


    QPushButton *stop = new QPushButton();
    stop->setText("STOP");
    stop->setFixedHeight(30);
    stop->setFixedWidth(60);

    QObject::connect(stop, SIGNAL(clicked()), this, SLOT(stopped()));
    QV->addWidget(stop);
    qDi.setLayout(QV);
    qDi.show();}





QString Campari::msg1(){
    return sendCons;
}


void Campari::rpcFile(){
    fichier = QFileDialog::getExistingDirectory(0, ("Select Output Folder"), QDir::currentPath());

    QString name = fichier.fileName();
    QMessageBox msgBox;
    msgBox.setWindowTitle("Campari");
    msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + name);
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if(msgBox.exec() == QMessageBox::Yes){
        rPC->setText(name);
    }else {        }
}


void Campari::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();
}


void Campari::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}






