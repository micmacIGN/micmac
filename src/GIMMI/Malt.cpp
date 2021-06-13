#include "Malt.h"

Malt::Malt(QList <QString> list_all ,QList <QString> list, QList <QString> listCheckedItems, QString path) : QWidget()
{

    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    sendCons="";
    path_s = path.toStdString();
    setWindowTitle("Malt");
    qDebug()<< path;
    QVBoxLayout *boxlayoutv = new QVBoxLayout;



    QLabel *labelMode = new QLabel(this);
    labelMode->setText("Mode :");
    labelMode->setStyleSheet("font-weight: bold");

    QLabel *labelMode1 = new QLabel(this);
    labelMode1->setText("Input-Orientation :");
    labelMode1->setStyleSheet("font-weight: bold");

    QLabel *labelOptions = new QLabel(this);
    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");


    QLabel *labelSizeWindow = new QLabel(this);
    labelSizeWindow->setText("SizeWindow");
    labelSizeWindowQt = new QTextEdit();
    QHBoxLayout *boxlayouth0 = new QHBoxLayout;
    boxlayouth0->addWidget(labelSizeWindow);
    boxlayouth0->addWidget(labelSizeWindowQt);
    labelSizeWindowQt->setFixedWidth(300);
    labelSizeWindowQt->setFixedHeight(27);

    QLabel *labelDefCord = new QLabel(this);
    labelDefCord->setText("DefCor");
    labelDefCordQt = new QTextEdit();
    QHBoxLayout *boxlayouth1 = new QHBoxLayout;
    boxlayouth1->addWidget(labelDefCord);
    boxlayouth1->addWidget(labelDefCordQt);
    labelDefCordQt->setFixedWidth(300);
    labelDefCordQt->setFixedHeight(27);


    QLabel *labelZoomF = new QLabel(this);
    labelZoomF->setText("ZoomF");
    labelZoomFQt = new QTextEdit();
    QHBoxLayout *boxlayouth2 = new QHBoxLayout;
    boxlayouth2->addWidget(labelZoomF);
    boxlayouth2->addWidget(labelZoomFQt);
    labelZoomFQt->setFixedWidth(300);
    labelZoomFQt->setFixedHeight(27);

    QLabel *labelNbVI = new QLabel(this);
    labelNbVI->setText("NbVI");
    labelNbVIQt = new QTextEdit();
    QHBoxLayout *boxlayouth3 = new QHBoxLayout;
    boxlayouth3->addWidget(labelNbVI);
    boxlayouth3->addWidget(labelNbVIQt);
    labelNbVIQt->setFixedWidth(300);
    labelNbVIQt->setFixedHeight(27);

    QLabel *labelDirMEC = new QLabel(this);
    labelDirMEC->setText("DirMEC");
    labelDirMECQt = new QTextEdit();
    QHBoxLayout *boxlayouth4 = new QHBoxLayout;
    boxlayouth4->addWidget(labelDirMEC);
    boxlayouth4->addWidget(labelDirMECQt);
    labelDirMECQt->setFixedWidth(300);
    labelDirMECQt->setFixedHeight(27);

    QLabel *labelRegul = new QLabel(this);
    labelRegul->setText("Regul");
    labelRegulQt = new QTextEdit();
    QHBoxLayout *boxlayouth5 = new QHBoxLayout;
    boxlayouth5->addWidget(labelRegul);
    boxlayouth5->addWidget(labelRegulQt);
    labelRegulQt->setFixedWidth(300);
    labelRegulQt->setFixedHeight(27);

    QLabel *labelZMoy = new QLabel(this);
    labelZMoy->setText("ZMoy");
    labelZMoyQt = new QTextEdit();
    QHBoxLayout *boxlayouth6 = new QHBoxLayout;
    boxlayouth6->addWidget(labelZMoy);
    boxlayouth6->addWidget(labelZMoyQt);
    labelZMoyQt->setFixedWidth(300);
    labelZMoyQt->setFixedHeight(27);

    QLabel *labelZinc = new QLabel(this);
    labelZinc->setText("Zinc");
    labelZincQt = new QTextEdit();
    QHBoxLayout *boxlayouth7 = new QHBoxLayout;
    boxlayouth7->addWidget(labelZinc);
    boxlayouth7->addWidget(labelZincQt);
    labelZincQt->setFixedWidth(300);
    labelZincQt->setFixedHeight(27);

    QLabel *labelEza = new QLabel(this);
    labelEza->setText("Eza");
    labelEzaQt = new QCheckBox();
    QHBoxLayout *boxlayouth8 = new QHBoxLayout;
    boxlayouth8->addWidget(labelEza);
    boxlayouth8->addWidget(labelEzaQt);
    labelEzaQt->setFixedWidth(300);
    labelEzaQt->setFixedHeight(27);

    QLabel *labelforDeform = new QLabel(this);
    labelforDeform->setText("Deform");
    labelforDeformQt = new QCheckBox();
    QHBoxLayout *boxlayouth9 = new QHBoxLayout;
    boxlayouth9->addWidget(labelforDeform);
    boxlayouth9->addWidget(labelforDeformQt);
    labelforDeformQt->setFixedWidth(300);
    labelforDeformQt->setFixedHeight(27);

    QLabel *labelVSND = new QLabel(this);
    labelVSND->setText("VSND");
    labelVSNDQt = new QTextEdit();
    QHBoxLayout *boxlayouth10 = new QHBoxLayout;
    boxlayouth10->addWidget(labelVSND);
    boxlayouth10->addWidget(labelVSNDQt);
    labelVSNDQt->setFixedWidth(300);
    labelVSNDQt->setFixedHeight(27);

    QLabel *labelImOrtho = new QLabel(this);
    labelImOrtho->setText("ImOrtho");
    labelImOrthoQt = new QTextEdit();
    QHBoxLayout *boxlayouth11 = new QHBoxLayout;
    boxlayouth11->addWidget(labelImOrtho);
    boxlayouth11->addWidget(labelImOrthoQt);
    labelImOrthoQt->setFixedWidth(300);
    labelImOrthoQt->setFixedHeight(27);

    QLabel *labelDoOrtho = new QLabel(this);
    labelDoOrtho->setText("DoOrtho");
    labelDoOrthoQt = new QTextEdit();
    QHBoxLayout *boxlayouth12 = new QHBoxLayout;
    boxlayouth12->addWidget(labelDoOrtho);
    boxlayouth12->addWidget(labelDoOrthoQt);
    labelDoOrthoQt->setFixedWidth(300);
    labelDoOrthoQt->setFixedHeight(27);

    QLabel *label2Ortho = new QLabel(this);
    label2Ortho->setText("2Ortho");
    label2OrthoQt = new QCheckBox();
    QHBoxLayout *boxlayouth13 = new QHBoxLayout;
    boxlayouth13->addWidget(label2Ortho);
    boxlayouth13->addWidget(label2OrthoQt);
    label2OrthoQt->setFixedWidth(300);
    label2OrthoQt->setFixedHeight(27);



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

    groupBox = new QButtonGroup();

    UrbanMNE = new QRadioButton(tr("UrbanMNE"));
    Ortho = new QRadioButton(tr("Ortho"));
    GeomImage = new QRadioButton(tr("GeomImage"));


    UrbanMNE->setChecked(true);



    groupBox1 = new QComboBox();

    for (int i = 0; i < list2.count(); i++)
    {
        groupBox1->addItem(list2[i]);
    }

    boxlayoutv->addWidget(labelMode1);
    boxlayoutv->addWidget(groupBox1);


    boxlayoutv->addWidget(labelMode);

    boxlayoutv->addWidget(UrbanMNE);
    boxlayoutv->addWidget(Ortho);
    boxlayoutv->addWidget(GeomImage);




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
    boxlayoutscroll->addLayout(boxlayouth12);
    boxlayoutscroll->addLayout(boxlayouth13);

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


void Malt::mm3d(){


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

        if(UrbanMNE->isChecked())
        {
            mode="UrbanMNE";}
        if(Ortho->isChecked())
        {
            mode="Ortho";}
        if(GeomImage->isChecked())
        {
            mode="GeomImage";}



        std::cout <<  "mode = " + mode << std::endl;

//        outPutCampStd = outPutCampT->toPlainText().toStdString();
//        std::cout <<  "Size m = " + outPutCampStd << std::endl;


        modo = groupBox1->currentText().toStdString();

        std::cout <<  "mode = " + modo << std::endl;




        if(labelSizeWindowQt->toPlainText()!=""){

          labelSizeWindowVar = "SzW="+labelSizeWindowQt->toPlainText().toStdString();
        }

        if(labelZoomFQt->toPlainText()!=""){

            labelZoomFVar = "ZoomF="+ labelZoomFQt->toPlainText().toStdString();

        }

        if(labelNbVIQt->toPlainText()!=""){

            labelNbVIVar = "NbVI="+ labelNbVIQt->toPlainText().toStdString();

        }

        if(labelDirMECQt->toPlainText()!=""){

            labelDirMECVar = "DirMEC="+labelDirMECQt->toPlainText().toStdString();

        }

        if(labelforDeformQt->isChecked()){

            labelforDeformVar = "ForDeform=1";

        }

        if(labelRegulQt->toPlainText()!=""){

            labelRegulVar = "Regul="+labelRegulQt->toPlainText().toStdString();

        }

        if(labelZMoyQt->toPlainText()!=""){

            labelZMoyVar = "ZMoy="+labelZMoyQt->toPlainText().toStdString();

        }
        if(labelZincQt->toPlainText()!=""){

            labelZincVar = "ZInc="+labelZincQt->toPlainText().toStdString();

        }

        if(labelEzaQt->isChecked()){

            labelEzaVar = "EZA=1";

        }

        if(labelVSNDQt->toPlainText()!=""){

            labelVSNDVar = "VSND="+labelVSNDQt->toPlainText().toStdString();

        }

        if(labelImOrthoQt->toPlainText()!=""){

            labelImOrthoVar = "ImOrtho="+labelImOrthoQt->toPlainText().toStdString();

        }

        if(labelDoOrthoQt->toPlainText()!=""){

            labelOrthoVar = "DoOrtho="+labelDoOrthoQt->toPlainText().toStdString();

        }

        if(label2OrthoQt->isChecked()){

            labelDeOrthoVar = "2Ortho=1";

        }

        if(labelDefCordQt->toPlainText()!=""){

            labelDefCordVar = "DefCor="+labelDefCordQt->toPlainText().toStdString();

        }














        //cmd = "mm3d Malt " + mode +" \""+ images +"\" "+ outPutCampStd +" "+ modo +" "+ txt;
        cmd = "mm3d Malt " + mode +" \""+ images +"\" "+ modo +" " +" "
                + labelSizeWindowVar.c_str()+ " " +
                +  labelDefCordVar.c_str()+ " " +
                +  labelZoomFVar.c_str()+ " " +
                +  labelNbVIVar.c_str()+ " " +
                +  labelDirMECVar.c_str()+ " " +
                +  labelRegulVar.c_str()+ " " +
                +  labelZMoyVar.c_str()+ " " +
                +  labelZincVar.c_str()+ " " +
                +  labelEzaVar.c_str()+ " " +
                +  labelforDeformVar.c_str()+ " " +
                +  labelVSNDVar.c_str()+ " " +
                +  labelImOrthoVar.c_str()+ " " +
                +  labelOrthoVar.c_str()+ " " +
                +  labelDeOrthoVar.c_str()+ " " +


                +"@ExitOnBrkp";

        std::cout <<  cmd << std::endl;




        QMessageBox msgBox2;
        msgBox2.setWindowTitle("Malt");
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
             labelSizeWindowVar="";
              labelDefCordVar="";
              labelZoomFVar="";
              labelNbVIVar="";
              labelDirMECVar="";
              labelRegulVar="";
              labelZMoyVar="";
              labelZincVar="";
              labelEzaVar="";
              labelforDeformVar="";
              labelVSNDVar="";
              labelImOrthoVar="";
              labelOrthoVar="";
              labelDeOrthoVar="";}

        // do something else



    }}






//************************************************************************************************************************************
// CODE POUR ENVOYER UN LE CONTENU DU COMPTE RENDU  TRAITEMENT IMAGE

QString Malt::getTxt(){

    return p_stdout;

    //************************************************************************************************************************************



}


//************************************************************************************************************************************
// CODE POUR ENVOYER UNE BOX APRES TRAITEMENT IMAGE FINI
void Malt::msg(){
    qDi.setWindowTitle("Malt state");

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



void Malt::rpcFile(){
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





QString Malt::msg1(){
    return sendCons;
}


void Malt::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();


}

void Malt::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}

