#include "C3dc.h"

C3dc::C3dc(QList <QString> list_all, QList <QString> list, QList <QString> listCheckedItems, QString path) : QWidget()
{

    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal);
    sendCons="";
    path_s = path.toStdString();
    setWindowTitle("C3DC");
    qDebug()<< path;
    QVBoxLayout *boxlayoutv = new QVBoxLayout;

    QLabel *labelMode = new QLabel(this);
    labelMode->setText("Input-Orientation :");
    labelMode->setStyleSheet("font-weight: bold");

    QLabel *labelMode1 = new QLabel(this);
    labelMode1->setText("Mode :");
    labelMode1->setStyleSheet("font-weight: bold");


    QLabel *labelOptions = new QLabel(this);
    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");




    //      Masq3D INT
    QLabel *Masq3D = new QLabel(this);
    Masq3D->setText("Masq3D");
    Masq3DQt = new QTextEdit();
    QHBoxLayout *boxlayouth2 = new QHBoxLayout;
    boxlayouth2->addWidget(Masq3D);
    boxlayouth2->addWidget(Masq3DQt);
    Masq3DQt->setFixedWidth(300);
    Masq3DQt->setFixedHeight(27);
    //     Out INT
    QLabel *Out = new QLabel(this);
    Out->setText("Out");
    OutQt = new QTextEdit();
    QHBoxLayout *boxlayouth3 = new QHBoxLayout;
    boxlayouth3->addWidget(Out);
    boxlayouth3->addWidget(OutQt);
    OutQt->setFixedWidth(300);
    OutQt->setFixedHeight(27);
    //      SzNorm INT :: {Do Compensation}
    QLabel *SzNorm = new QLabel(this);
    SzNorm->setText("SzNorm");
    SzNormQt = new QTextEdit();
    QHBoxLayout *boxlayouth4 = new QHBoxLayout;
    boxlayouth4->addWidget(SzNorm);
    boxlayouth4->addWidget(SzNormQt);
    SzNormQt->setFixedWidth(300);
    SzNormQt->setFixedHeight(27);
    //      PlyCoul] INT :: {Is for calibration (Change def value of LMV and prop diag)?}
    QLabel *PlyCoul = new QLabel(this);
    PlyCoul->setText("PlyCoul");
    PlyCoulQt = new QCheckBox();
    QHBoxLayout *boxlayouth5 = new QHBoxLayout;
    boxlayouth5->addWidget(PlyCoul);
    boxlayouth5->addWidget(PlyCoulQt);
    PlyCoulQt->setFixedWidth(300);
    PlyCoulQt->setFixedHeight(27);
    //      Tuning] Pt2dr :: {Keep images with focal length inside range [A,B] (A,B in mm) (Def=keep all)}
    QLabel *Tuning = new QLabel(this);
    Tuning->setText("Tuning");
    TuningQt = new QCheckBox();
    QHBoxLayout *boxlayouth6 = new QHBoxLayout;
    boxlayouth6->addWidget(Tuning);
    boxlayouth6->addWidget(TuningQt);
    TuningQt->setFixedWidth(300);
    TuningQt->setFixedHeight(27);
    //      Purge] INT
    QLabel *Purge = new QLabel(this);
    Purge->setText("Purge");
    PurgeQt = new QCheckBox();
    QHBoxLayout *boxlayouth7 = new QHBoxLayout;
    boxlayouth7->addWidget(Purge);
    boxlayouth7->addWidget(PurgeQt);
    PurgeQt->setFixedWidth(300);
    PurgeQt->setFixedHeight(27);
    //      DownScale] Pt2dr :: {Principal point shift}
    QLabel *DownScale = new QLabel(this);
    DownScale->setText("DownScale");
    DownScaleQt = new QTextEdit();
    QHBoxLayout *boxlayouth8 = new QHBoxLayout;
    boxlayouth8->addWidget(DownScale);
    boxlayouth8->addWidget(DownScaleQt);
    DownScaleQt->setFixedWidth(300);
    DownScaleQt->setFixedHeight(27);
    //      ZoomF] INT :: {Principal point is shifted (Def=false)}
    QLabel *ZoomF = new QLabel(this);
    ZoomF->setText("ZoomF");
    ZoomFQt = new QTextEdit();
    QHBoxLayout *boxlayouth9 = new QHBoxLayout;
    boxlayouth9->addWidget(ZoomF);
    boxlayouth9->addWidget(ZoomFQt);
    ZoomFQt->setFixedWidth(300);
    ZoomFQt->setFixedHeight(27);
    //      UseGpu] REAL :: {Hemi-spherik fisheye diameter to diagonal ratio}
    QLabel *UseGpu = new QLabel(this);
    UseGpu->setText("UseGpu");
    UseGpuQt = new QCheckBox();
    QHBoxLayout *boxlayouth10 = new QHBoxLayout;
    boxlayouth10->addWidget(UseGpu);
    boxlayouth10->addWidget(UseGpuQt);
    UseGpuQt->setFixedWidth(300);
    UseGpuQt->setFixedHeight(27);
    //      DefCor] string :: {Save intermediary results to, Set NONE if dont want any}
    QLabel *DefCor = new QLabel(this);
    DefCor->setText("DefCor");
    DefCorQt = new QTextEdit();
    QHBoxLayout *boxlayouth11= new QHBoxLayout;
    boxlayouth11->addWidget(DefCor);
    boxlayouth11->addWidget(DefCorQt);
    DefCorQt->setFixedWidth(300);
    DefCorQt->setFixedHeight(27);
    //      ZReg] string :: {Force first image}
    QLabel *ZReg = new QLabel(this);
    ZReg->setText("ZReg");
    ZRegQt = new QTextEdit();
    QHBoxLayout *boxlayouth12= new QHBoxLayout;
    boxlayouth12->addWidget(ZReg);
    boxlayouth12->addWidget(ZRegQt);
    ZRegQt->setFixedWidth(300);
    ZRegQt->setFixedHeight(27);
    //      ExpTxt] bool :: {MOI}
    QLabel *ExpTxt = new QLabel(this);
    ExpTxt->setText("ExpTxt");
    ExpTxtQt = new QCheckBox();
    QHBoxLayout *boxlayouth13= new QHBoxLayout;
    boxlayouth13->addWidget(ExpTxt);
    boxlayouth13->addWidget(ExpTxtQt);
    ExpTxtQt->setFixedWidth(300);
    ExpTxtQt->setFixedHeight(27);
    //      FilePair] INT :: {Debug (internal use : DebugPbCondFaisceau=true) }
    QLabel *FilePair = new QLabel(this);
    FilePair->setText("FilePair");
    FilePairQt = new QTextEdit();
    QHBoxLayout *boxlayouth14 = new QHBoxLayout;
    boxlayouth14->addWidget(FilePair);
    boxlayouth14->addWidget(FilePairQt);
    FilePairQt->setFixedWidth(300);
    FilePairQt->setFixedHeight(27);
    //      DebugMMByP] bool :: {Partial file for debug}
    QLabel *DebugMMByP = new QLabel(this);
    DebugMMByP->setText("DebugMMByP");
    DebugMMByPQt = new QCheckBox();
    QHBoxLayout *boxlayouth15 = new QHBoxLayout;
    boxlayouth15->addWidget(DebugMMByP);
    boxlayouth15->addWidget(DebugMMByPQt);
    DebugMMByPQt->setFixedWidth(300);
    DebugMMByPQt->setFixedHeight(27);
    //      Bin] INT :: {Max degree of radial, default model dependent}
    QLabel *Bin = new QLabel(this);
    Bin->setText("Bin");
    BinQt = new QCheckBox();
    QHBoxLayout *boxlayouth16 = new QHBoxLayout;
    boxlayouth16->addWidget(Bin);
    boxlayouth16->addWidget(BinQt);
    BinQt->setFixedWidth(300);
    BinQt->setFixedHeight(27);
    //      ExpImSec] INT :: {Max degree of general polynome, default model dependent (generally 0 or 1)}
    QLabel *ExpImSec = new QLabel(this);
    ExpImSec->setText("ExpImSec");
    ExpImSecQt = new QCheckBox();
    QHBoxLayout *boxlayouth17= new QHBoxLayout;
    boxlayouth17->addWidget(ExpImSec);
    boxlayouth17->addWidget(ExpImSecQt);
    ExpImSecQt->setFixedWidth(300);
    ExpImSecQt->setFixedHeight(27);
    //      OffsetPly] bool :: {Free affine parameter, Def=true}
    QLabel *OffsetPly = new QLabel(this);
    OffsetPly->setText("OffsetPly");
    OffsetPlyQt = new QTextEdit();
    OffsetPlyQt2 = new QTextEdit();
    OffsetPlyQt3 = new QTextEdit();
    QHBoxLayout *boxlayouth18= new QHBoxLayout;
    QHBoxLayout *boxlayouth188= new QHBoxLayout;

    boxlayouth18->addWidget(OffsetPly);
    boxlayouth188->addWidget(OffsetPlyQt);
    boxlayouth188->addWidget(OffsetPlyQt2);
    boxlayouth188->addWidget(OffsetPlyQt3);

    OffsetPlyQt->setFixedWidth(99);
    OffsetPlyQt->setFixedHeight(27);
    OffsetPlyQt2->setFixedWidth(99);
    OffsetPlyQt2->setFixedHeight(27);
    OffsetPlyQt3->setFixedWidth(99);
    OffsetPlyQt3->setFixedHeight(27);
    boxlayouth18->addLayout(boxlayouth188);


    //    boxlayoutv->addWidget(labelImages);

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
    groupBox1->addItem("Ground");
    groupBox1->addItem("Statue");
    groupBox1->addItem("Forest");
    groupBox1->addItem("TestIGN");
    groupBox1->addItem("QuickMac");
    groupBox1->addItem("MicMac");
    groupBox1->addItem("BigMac");
    groupBox1->addItem("MTDTmp");

    boxlayoutv->addWidget(labelMode1);
    boxlayoutv->addWidget(groupBox1);
    boxlayoutv->addWidget(labelMode);

    groupBox2 = new QComboBox();

    for (int i = 0; i < list2.count(); i++)
    {
        groupBox2->addItem(list2[i]);
    }
    boxlayoutv->addWidget(groupBox2);
    boxlayoutv->addWidget(labelOptions);

    QScrollArea *scrollArea = new QScrollArea;
    scrollArea->setFixedSize(445, 300);
    QWidget *layoutWidget = new QWidget(this);
    QVBoxLayout *boxlayoutscroll = new QVBoxLayout;
    layoutWidget->setLayout(boxlayoutscroll);

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
    boxlayoutscroll->addLayout(boxlayouth12);
    boxlayoutscroll->addLayout(boxlayouth13);
    boxlayoutscroll->addLayout(boxlayouth14);
    boxlayoutscroll->addLayout(boxlayouth15);
    boxlayoutscroll->addLayout(boxlayouth16);
    boxlayoutscroll->addLayout(boxlayouth17);
    boxlayoutscroll->addLayout(boxlayouth18);
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


void C3dc::mm3d(){

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
        modo = groupBox1->currentText().toStdString();
        std::cout <<  "mode = " + modo << std::endl;

        outPutCampStd = groupBox2->currentText().toStdString();
        if(Masq3DQt->toPlainText()!=""){
            Masq3DVar="Masq3D="+Masq3DQt->toPlainText().toStdString();
        }

        if(OutQt->toPlainText()!=""){
            OutVar="Out="+OutQt->toPlainText().toStdString();
        }
        if(SzNormQt->toPlainText()!=""){
            SzNormVar="SzNorm="+SzNormQt->toPlainText().toStdString();
        }
        if(PlyCoulQt->isChecked()){
            PlyCoulVar="PlyCoul=1";
        }
        if(TuningQt->isChecked()){
            TuningVar="Tuning=1";
        }
        if(PurgeQt->isChecked()){
            PurgeVar="Purge=1";
        }
        if(DownScaleQt->toPlainText()!=""){
            DownScaleVar="DownScale="+DownScaleQt->toPlainText().toStdString();
        }
        if(ZoomFQt->toPlainText()!=""){
            ZoomFVar="ZoomF="+ZoomFQt->toPlainText().toStdString();
        }
        if(UseGpuQt->isChecked()){
            UseGpuVar="UseGpu=1";
            DefCorVar="DefCor="+DefCorQt->toPlainText().toStdString();
        }
        if(ZRegQt->toPlainText()!=""){
            ZRegVar="ZReg="+ZRegQt->toPlainText().toStdString();
        }
        if(ExpTxtQt->isChecked()){
            ExpTxtVar="ExpTxt=1";
        }
        if(FilePairQt->toPlainText()!=""){
            FilePairVar="FilePair="+FilePairQt->toPlainText().toStdString();
        }
        if(DebugMMByPQt->isChecked()){
            DebugMMByPVar="DebugMMByP=1";
        }
        if(BinQt->isChecked()){
            BinVar="Bin=1";
        }
        if(ExpImSecQt->isChecked()){
            ExpImSecVar="ExpImSec=1";
        }

        if(OffsetPlyQt->toPlainText()!="" && OffsetPlyQt2->toPlainText()!="" && OffsetPlyQt3->toPlainText()!=""){
            OffsetPlyVar="OffsetPly=["+OffsetPlyQt->toPlainText().toStdString()+";"+OffsetPlyQt2->toPlainText().toStdString()+";"+OffsetPlyQt3->toPlainText().toStdString()+"]";
        }






        cmd = "mm3d C3DC " + modo +" \""+ images +"\" "+ outPutCampStd +" "+
                + Masq3DVar.c_str()+" "+
                + OutVar.c_str()+" "+
                + SzNormVar.c_str()+" "+
                + PlyCoulVar.c_str()+" "+
                + TuningVar.c_str()+" "+
                + PurgeVar.c_str()+" "+
                + DownScaleVar.c_str()+" "+
                + ZoomFVar.c_str()+" "+
                + UseGpuVar.c_str()+" "+
                + DefCorVar.c_str()+" "+
                + ZRegVar.c_str()+" "+
                + ExpTxtVar.c_str()+" "+
                + FilePairVar.c_str()+" "+
                + DebugMMByPVar.c_str()+" "+
                + BinVar.c_str()+" "+
                + ExpImSecVar.c_str()+" "+
                + OffsetPlyVar.c_str()+" "+
                + " @ExitOnBrkp";

        std::cout <<  cmd << std::endl;

        QMessageBox msgBox2;
        msgBox2.setWindowTitle("C3DC");
        const QString str = QString::fromStdString(cmd);
        msgBox2.setText("Here is the commande you're about to launch : \n " + str);
        msgBox2.setStandardButtons(QMessageBox::Yes);
        msgBox2.addButton(QMessageBox::No);
        msgBox2.setDefaultButton(QMessageBox::No);
        if(msgBox2.exec() == QMessageBox::Yes){

            this->close();
            //       cons.show();
            p.setWorkingDirectory(path_s.c_str());
            p.waitForFinished(-1);
            qDebug() <<  QDir::currentPath();
            qDebug() <<  cmd.c_str() ;
            p.start(cmd.c_str());
            p.setReadChannel(QProcess::StandardOutput);
            connect(&p,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
            //        p.waitForFinished(-1);
            //        p.kill();
            connect(&p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));  //Add this line will cause error at runtime
            msg();


        }else{ images.clear();
            outPutCampStd="";
            Masq3DVar="";
            OutVar="";
            SzNormVar="";
            PlyCoulVar="";
            TuningVar="";
            PurgeVar="";
            DownScaleVar="";
            ZoomFVar="";
            UseGpuVar="";
            DefCorVar="";
            ZRegVar="";
            ExpTxtVar="";
            FilePairVar="";
            DebugMMByPVar="";
            BinVar="";
            ExpImSecVar="";
            OffsetPlyVar = "";


        }

        // do something else



    }}





//************************************************************************************************************************************
// CODE POUR ENVOYER UN LE CONTENU DU COMPTE RENDU  TRAITEMENT IMAGE

QString C3dc::getTxt(){

    return p_stdout;

    //************************************************************************************************************************************



}


//************************************************************************************************************************************
// CODE POUR ENVOYER UNE BOX APRES TRAITEMENT IMAGE FINI
void C3dc::msg(){
    qDi.setWindowTitle("C3DC state");

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

//************************************************************************************************************************************



void C3dc::rpcFile(){
    fichier = QFileDialog::getExistingDirectory(0, ("Select Output Folder"), QDir::currentPath());

    QString name = fichier.fileName();
    QMessageBox msgBox;
    msgBox.setWindowTitle("title");
    msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + name);
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if(msgBox.exec() == QMessageBox::Yes){
        outPutCampT->setText(name);
    }else {        }
}





QString C3dc::msg1(){
    return sendCons;
}


void C3dc::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();


}

void C3dc::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}

