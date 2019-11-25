
#include "MainWindow.h"
#include "NewProject.h"
#include "Tapioca.h"
#include "Console.h"



Tapioca::Tapioca(QList <QString> list,QList <QString> listCheckedItems, QString path) : QWidget()
{

    this->show();


    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    sendCons="";
    setWindowTitle("Tapioca");
    path_s = path.toStdString();
    boxLayoutV = new QVBoxLayout;

    QLabel *labelMode = new QLabel(this);
    QPalette pal2 = palette();

    // set black background
    // pal2.setColor(QPalette::WindowText, Qt::white);

    labelMode->setAutoFillBackground(true);
    labelMode->setPalette(pal2);
    labelMode->show();
    labelMode->setMargin(5);
    labelMode->setText("Strategy :");
    labelMode->setStyleSheet("font-weight: bold");


    labelSizeWm = new QLabel(this);
    labelSizeWm->setAutoFillBackground(true);
    labelSizeWm->setPalette(pal2);
    labelSizeWm->show();
    labelSizeWm->setMargin(5);
    labelSizeWm->setText("Size of low resolution images :");
    labelSizeWm->setStyleSheet("font-weight: bold");

    labelSizeWM = new QLabel(this);
    labelSizeWM->setAutoFillBackground(true);
    labelSizeWM->setPalette(pal2);
    labelSizeWM->show();
    labelSizeWM->setMargin(5);
    labelSizeWM->setText("Size of high resolution images :");
    labelSizeWM->setStyleSheet("font-weight: bold");

    labelOptions = new QLabel(this);
    labelOptions->setAutoFillBackground(true);
    labelOptions->setPalette(pal2);
    labelOptions->show();
    labelOptions->setMargin(5);
    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");


    //      ExpTxt INT
        QLabel *ExpTxt = new QLabel(this);
        ExpTxt->setText("ExpTxt");
        ExpTxtQt = new QCheckBox();
        QHBoxLayout *boxlayouth3 = new QHBoxLayout;
        boxlayouth3->addWidget(ExpTxt);
        boxlayouth3->addWidget(ExpTxtQt);
        ExpTxtQt->setFixedWidth(300);
        ExpTxtQt->setFixedHeight(27);
    //      DoC] INT :: {Do Compensation}
        QLabel *ByP = new QLabel(this);
        ByP->setText("ByP");
        ByPQt = new QTextEdit();
        QHBoxLayout *boxlayouth4 = new QHBoxLayout;
        boxlayouth4->addWidget(ByP);
        boxlayouth4->addWidget(ByPQt);
        ByPQt->setFixedWidth(300);
        ByPQt->setFixedHeight(27);
    //      ForCalib] INT :: {Is for calibration (Change def value of LMV and prop diag)?}
        QLabel *PostFix = new QLabel(this);
        PostFix->setText("PostFix");
        PostFixQt = new QTextEdit();
        QHBoxLayout *boxlayouth5 = new QHBoxLayout;
        boxlayouth5->addWidget(PostFix);
        boxlayouth5->addWidget(PostFixQt);
        PostFixQt->setFixedWidth(300);
        PostFixQt->setFixedHeight(27);
    //      Focs] Pt2dr :: {Keep images with focal length inside range [A,B] (A,B in mm) (Def=keep all)}
        QLabel *NbMinPt = new QLabel(this);
        NbMinPt->setText("NbMinPt");
        NbMinPtQt = new QTextEdit();
        QHBoxLayout *boxlayouth6 = new QHBoxLayout;
        boxlayouth6->addWidget(NbMinPt);
        boxlayouth6->addWidget(NbMinPtQt);
        NbMinPtQt->setFixedWidth(300);
        NbMinPtQt->setFixedHeight(27);
    //      VitesseInit] INT
        QLabel *DLR = new QLabel(this);
        DLR->setText("DLR");
        DLRQt = new QTextEdit();
        QHBoxLayout *boxlayouth7 = new QHBoxLayout;
        boxlayouth7->addWidget(DLR);
        boxlayouth7->addWidget(DLRQt);
        DLRQt->setFixedWidth(300);
        DLRQt->setFixedHeight(27);

        QLabel *Detect = new QLabel(this);
        Detect->setText("Detect");
        DetectQt = new QTextEdit();
        QHBoxLayout *boxlayouth9 = new QHBoxLayout;
        boxlayouth9->addWidget(Detect);
        boxlayouth9->addWidget(DetectQt);
        DetectQt->setFixedWidth(300);
        DetectQt->setFixedHeight(27);
    //      PropDiag] REAL :: {Hemi-spherik fisheye diameter to diagonal ratio}
        QLabel *Match = new QLabel(this);
        Match->setText("Match");
        MatchQt = new QTextEdit();
        QHBoxLayout *boxlayouth10 = new QHBoxLayout;
        boxlayouth10->addWidget(Match);
        boxlayouth10->addWidget(MatchQt);
        MatchQt->setFixedWidth(300);
        MatchQt->setFixedHeight(27);
    //      SauvAutom] string :: {Save intermediary results to, Set NONE if dont want any}
        QLabel *NoMax = new QLabel(this);
        NoMax->setText("NoMaxx");
        NoMaxQt = new QCheckBox();
        QHBoxLayout *boxlayouth11= new QHBoxLayout;
        boxlayouth11->addWidget(NoMax);
        boxlayouth11->addWidget(NoMaxQt);
        NoMaxQt->setFixedWidth(300);
        NoMaxQt->setFixedHeight(27);
    //      ImInit] string :: {Force first image}
        QLabel *NoMin = new QLabel(this);
        NoMin->setText("NoMin");
        NoMinQt = new QCheckBox();
        QHBoxLayout *boxlayouth12= new QHBoxLayout;
        boxlayouth12->addWidget(NoMin);
        boxlayouth12->addWidget(NoMinQt);
        NoMinQt->setFixedWidth(300);
        NoMinQt->setFixedHeight(27);
    //      MOI] bool :: {MOI}
        QLabel *NoUnknown = new QLabel(this);
        NoUnknown->setText("NoUnknown");
        NoUnknownQt = new QCheckBox();
        QHBoxLayout *boxlayouth13= new QHBoxLayout;
        boxlayouth13->addWidget(NoUnknown);
        boxlayouth13->addWidget(NoUnknownQt);
        NoUnknownQt->setFixedWidth(300);
        NoUnknownQt->setFixedHeight(27);
        QLabel *Ratio = new QLabel(this);
        Ratio->setText("Ratio");
        RatioQt = new QTextEdit();
        QHBoxLayout *boxlayouth14= new QHBoxLayout;
        boxlayouth14->addWidget(Ratio);
        boxlayouth14->addWidget(RatioQt);
        RatioQt->setFixedWidth(300);
        RatioQt->setFixedHeight(27);


//    boxLayoutV->addWidget(labelImages);

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



    mulScale = new QRadioButton(tr("MulScale"));
    QObject::connect(mulScale, SIGNAL(clicked(bool)), this, SLOT(mulScaleC()));

    all = new QRadioButton(tr("All"));
    QObject::connect(all, SIGNAL(clicked(bool)), this, SLOT(allC()));

    line = new QRadioButton(tr("Line"));
    QObject::connect(line, SIGNAL(clicked(bool)), this, SLOT(lineC()));

    file = new QRadioButton(tr("File"));
    QObject::connect(file, SIGNAL(clicked(bool)), this, SLOT(fileC()));

    graph = new QRadioButton(tr("Graph"));
    QObject::connect(graph, SIGNAL(clicked(bool)), this, SLOT(graphC()));

    georeph = new QRadioButton(tr("Georeph"));
    QObject::connect(georeph, SIGNAL(clicked(bool)), this, SLOT(georephC()));

    mulScale->setChecked(true);

    boxLayoutV->addWidget(labelMode);
    boxLayoutV->addWidget(mulScale);
    boxLayoutV->addWidget(all);
    boxLayoutV->addWidget(line);
    boxLayoutV->addWidget(file);
    boxLayoutV->addWidget(graph);
    boxLayoutV->addWidget(georeph);

    resom = new QTextEdit();

    resoM = new QTextEdit();
    resom->setFixedHeight(27);
    resoM->setFixedHeight(27);


    boxLayoutV->addWidget(labelSizeWm);
    boxLayoutV->addWidget(resom);
    boxLayoutV->addWidget(labelSizeWM);
    boxLayoutV->addWidget(resoM);
    boxLayoutV->addWidget(labelOptions);


    QScrollArea *scrollArea = new QScrollArea;
    scrollArea->setFixedSize(430, 300);
    QWidget *layoutWidget = new QWidget(this);
    QVBoxLayout *boxlayoutscroll = new QVBoxLayout;
    layoutWidget->setLayout(boxlayoutscroll);


     boxlayoutscroll->addLayout(boxlayouth3);
     boxlayoutscroll->addLayout(boxlayouth4);
     boxlayoutscroll->addLayout(boxlayouth5);
     boxlayoutscroll->addLayout(boxlayouth6);
     boxlayoutscroll->addLayout(boxlayouth7);
     boxlayoutscroll->addLayout(boxlayouth9);
     boxlayoutscroll->addLayout(boxlayouth10);
     boxlayoutscroll->addLayout(boxlayouth11);
     boxlayoutscroll->addLayout(boxlayouth12);
     boxlayoutscroll->addLayout(boxlayouth13);
     boxlayoutscroll->addLayout(boxlayouth14);



     boxLayoutV->addWidget(scrollArea);


     scrollArea->setWidget(layoutWidget);
     scrollArea->show();


    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
this->setLayout(boxLayoutV);

}


void Tapioca::mm3d(){

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


if(mulScale->isChecked())
{
    mode="MulScale";}
if(all->isChecked())
{
    mode="All";
}
if(line->isChecked())
{
    mode="Line";}
if(file->isChecked())
{
    mode="File";}
if(graph->isChecked())
{
   mode="Graph";}
if(georeph->isChecked())
{
    mode="Georeph";}

std::cout <<  "mode = " + mode << std::endl;

resom_str = resom->toPlainText().toStdString();
std::cout <<  "Size m = " + resom_str << std::endl;

resoM_str = resoM->toPlainText().toStdString();
std::cout <<  "Size M = " + resoM_str << std::endl;





if(ExpTxtQt->isChecked()){
    ExpTxtVar="ExpTxt=1";
}
if(ByPQt->toPlainText()!=""){
    ByPVar="ByP="+ByPQt->toPlainText().toStdString();
}
if(PostFixQt->toPlainText()!=""){
    PostFixVar="PostFix="+PostFixQt->toPlainText().toStdString();
}
if(NbMinPtQt->toPlainText()!=""){
    NbMinPtVar="NbMinPt="+NbMinPtQt->toPlainText().toStdString();
}
if(DLRQt->toPlainText()!=""){
    DLRVar="DLR="+DLRQt->toPlainText().toStdString();
}
if(DetectQt->toPlainText()!=""){
    DetectVar="Detect="+DetectQt->toPlainText().toStdString();
}
if(MatchQt->toPlainText()!=""){
    MatchVar="Match="+MatchQt->toPlainText().toStdString();
}
if(NoMaxQt->isChecked()){
    NoMaxVar="NoMax=1";
}
if(NoMinQt->isChecked()){
    NoMinVar="NoMin=1";
}
if(NoUnknownQt->isChecked()){
    NoUnknownVar="NoUnknown=1";
}
if(RatioQt->toPlainText()!=""){
    RatioVar="Ratio="+RatioQt->toPlainText().toStdString();
}



cmd = " mm3d Tapioca " + mode +" \""+ images +"\" "+ resom_str +" "+ resoM_str +" "
         + ExpTxtVar.c_str()+" "+
         + ByPVar.c_str()+" "+
         + PostFixVar.c_str()+" "+
         + NbMinPtVar.c_str()+" "+
         + DLRVar.c_str()+" "+
         + DetectVar.c_str()+" "+
         + MatchVar.c_str()+" "+
         + NoMaxVar.c_str()+" "+
         + NoMinVar.c_str()+" "+
         + NoUnknownVar.c_str()+" "+
         + RatioVar.c_str()+" "+
         +" @ExitOnBrkp";

    msgBox2.setWindowTitle("mm3d command");

    const QString cmd_str = QString::fromStdString(cmd);
    msgBox2.setText("Here is the commande you are about to launch : \n " + cmd_str);
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
        connect(&p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));  //Add this line will cause error at runtime
        msg();



      }else{images.clear();
        ExpTxtVar=" ";
        ByPVar=" ";
        PostFixVar=" ";
        NbMinPtVar=" ";
        DLRVar=" ";
        DetectVar=" ";
        MatchVar=" ";
        NoMaxVar=" ";
        NoMinVar=" ";
        NoUnknownVar=" ";
        RatioVar=" ";
        }
    }

}

QString Tapioca::getTxt(){
    return sendCons;
}

void Tapioca::msg(){
         qDi.setWindowTitle("Tapioca state");
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
        qDi.show();
      }

void Tapioca::mulScaleC(){
        labelSizeWm->setVisible(true);
        resom->setVisible(true);
        labelSizeWM->setVisible(true);
        labelSizeWm->setText("Size of low resolution images :");
        labelSizeWm->setStyleSheet("font-weight: bold");
        labelSizeWM->setText("Size of high resolution images :");
        labelSizeWM->setStyleSheet("font-weight: bold");
        labelSizeWM->setText("Maximum size resolution :");
        resom->setText("");
        resoM->setText("");
        resom->setFixedHeight(27);
        resom->setFixedWidth(500);
        resoM->setFixedHeight(27);
        resoM->setFixedWidth(500);
        resoM->setVisible(true);

      }

void Tapioca::allC(){
        labelSizeWm->setVisible(true);
        resom->setVisible(true);
        labelSizeWm->setText("Size of image :");
        labelSizeWM->setVisible(true);
        labelSizeWM->setText("");
        resoM->setFixedWidth(0);
        resom->setText("");
        resoM->setText("");
        resoM->setVisible(true);
      }

void Tapioca::lineC(){
    labelSizeWm->setVisible(true);
    resom->setVisible(true);
    labelSizeWM->setVisible(true);
    labelSizeWm->setText("Image Size :");
    labelSizeWm->setStyleSheet("font-weight: bold");
    labelSizeWM->setText("Number of adjacent images to look for :");
    labelSizeWM->setStyleSheet("font-weight: bold");
    resom->setText("");
    resoM->setText("");
    resom->setFixedHeight(27);
    resom->setFixedWidth(500);
    resoM->setFixedHeight(27);
    resoM->setFixedWidth(500);
    resoM->setVisible(true);
      }

void Tapioca::fileC(){
    labelSizeWm->setVisible(true);
    resom->setVisible(true);
    labelSizeWM->setVisible(true);
    labelSizeWm->setText("XML-File of pair :");
    labelSizeWm->setStyleSheet("font-weight: bold");
    labelSizeWM->setText("Resolution :");
    labelSizeWM->setStyleSheet("font-weight: bold");
    resom->setText("");
    resoM->setText("");
    resom->setFixedHeight(27);
    resom->setFixedWidth(500);
    resoM->setFixedHeight(27);
    resoM->setFixedWidth(500);
    resoM->setVisible(true);
      }

void Tapioca::graphC(){
    labelSizeWm->setVisible(true);
    resom->setVisible(true);
    labelSizeWm->setText("Processing size of image (for the greater dimension) :");
    labelSizeWM->setVisible(true);
    labelSizeWM->setText("");
    resoM->setFixedWidth(0);
    resom->setText("");
    resoM->setText("");
    resoM->setVisible(true);
      }

void Tapioca::georephC(){
    labelSizeWm->setVisible(true);
    resom->setVisible(true);
    labelSizeWm->setText("Orientation directory :");
    labelSizeWM->setVisible(true);
    labelSizeWM->setText("");
    resoM->setFixedWidth(0);
    resom->setText("");
    resoM->setText("");
    resoM->setVisible(true);
      }

void Tapioca::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();
      }


QString Tapioca::msg1(){
        return sendCons;
    }


void Tapioca::readyReadStandardOutput(){
  Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));

}

