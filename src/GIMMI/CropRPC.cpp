
#include "MainWindow.h"
#include "NewProject.h"
#include "CropRPC.h"
#include "Console.h"
//#include "XML_GEN/ParamChantierPhotogram.h"
//#include "general/MessageHandler.h"
//#include "StdAfx.h"


CropRPC::CropRPC(QList <QString> list,QList <QString> listCheckedItems, QString path) : QWidget()
{

    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    sendCons="";
    aPath = path;
    setWindowTitle("CropRPC");
    path_s = path.toStdString();
    boxLayoutV = new QVBoxLayout;
    QLabel *labelImages = new QLabel(this);
    labelImages->setText("Pattern of orientation files to be cropped accordingly (in cXml_CamGenPolBundle format)");
    labelImages->setStyleSheet("font-weight: bold");

    oriFileCrop = new QLabel(this);
    oriFileCrop->setText("Orientation file of the image defining the crop extent (in cXml_CamGenPolBundle format):");
    oriFileCrop->setStyleSheet("font-weight: bold");


    QLabel *directoryOutput = new QLabel(this);
    directoryOutput->setText("Directory of output orientation files:");
    directoryOutput->setStyleSheet("font-weight: bold");


    QLabel *Sz = new QLabel(this);
    Sz->setText("Sz");
    SzQt = new QTextEdit();
    SzQt2 = new QTextEdit();
    QHBoxLayout *boxlayouth6 = new QHBoxLayout;
    QHBoxLayout *boxlayouth66 = new QHBoxLayout;
    boxlayouth6->addWidget(Sz);
    boxlayouth66->addWidget(SzQt);
    boxlayouth66->addWidget(SzQt2);
    SzQt->setFixedWidth(147);
    SzQt->setFixedHeight(27);
    SzQt2->setFixedWidth(147);
    SzQt2->setFixedHeight(27);
    boxlayouth6->addLayout(boxlayouth66);


    QLabel *Org = new QLabel(this);
    Org->setText("Org");
    OrgQt = new QTextEdit();
    OrgQt2 = new QTextEdit();
    QHBoxLayout *boxlayouth7 = new QHBoxLayout;
    QHBoxLayout *boxlayouth77 = new QHBoxLayout;
    boxlayouth7->addWidget(Org);
    boxlayouth77->addWidget(OrgQt);
    boxlayouth77->addWidget(OrgQt2);
    OrgQt->setFixedWidth(147);
    OrgQt->setFixedHeight(27);
    OrgQt2->setFixedWidth(147);
    OrgQt2->setFixedHeight(27);
    boxlayouth7->addLayout(boxlayouth77);


    patternTxt = new QTextEdit();
    patternTxt->setFixedHeight(27);
    oriFileCropTxt = new QTextEdit();
    oriFileCropTxt->setFixedHeight(27);
    directoryOutputTxt = new QTextEdit();
    directoryOutputTxt->setFixedHeight(27);

    boxLayoutV->addWidget(labelImages);

    QHBoxLayout *boxlayouth1 = new QHBoxLayout;
    QPushButton *buttonOri1 = new QPushButton();
    buttonOri1->setText("...");
    buttonOri1->setFixedWidth(30);
    buttonOri1->setFixedHeight(30);
    QObject::connect(buttonOri1, SIGNAL(clicked()), this, SLOT(oriSearch1()));
    boxlayouth1->addWidget(patternTxt);
    boxLayoutV->addLayout(boxlayouth1);
    boxLayoutV->addWidget(oriFileCrop);

    QHBoxLayout *boxlayouth = new QHBoxLayout;
    QPushButton *buttonOri = new QPushButton();
    QObject::connect(buttonOri, SIGNAL(clicked()), this, SLOT(oriSearch()));

    buttonOri->setText("...");
    buttonOri->setFixedWidth(30);
    buttonOri->setFixedHeight(30);
    QObject::connect(buttonOri, SIGNAL(clicked()), this, SLOT(rpcFile()));

    boxlayouth->addWidget(oriFileCropTxt);
    boxlayouth->addWidget(buttonOri);

    boxLayoutV->addLayout(boxlayouth);

    boxLayoutV->addWidget(directoryOutput);
    boxLayoutV->addWidget(directoryOutputTxt);



    QScrollArea *scrollArea = new QScrollArea;
    scrollArea->setFixedHeight(90);
    QWidget *layoutWidget = new QWidget(this);
    QVBoxLayout *boxlayoutscroll = new QVBoxLayout;
    layoutWidget->setLayout(boxlayoutscroll);


    boxlayoutscroll->addLayout(boxlayouth6);
    boxlayoutscroll->addLayout(boxlayouth7);




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


void CropRPC::mm3d(){

    patternTxt_str = "\""+patternTxt->toPlainText().toStdString()+"\"";
    std::cout <<  "Size m = " + patternTxt_str << std::endl;

    oriFileCropTxt_str = oriFileCropTxt->toPlainText().toStdString();
    std::cout <<  "Size m = " + oriFileCropTxt_str << std::endl;

    directoryOutputTxt_str = directoryOutputTxt->toPlainText().toStdString();
    std::cout <<  "Size M = " + directoryOutputTxt_str << std::endl;

    if(OrgQt->toPlainText()!="" && OrgQt2->toPlainText()!=""){
        OrgVar="Org=["+OrgQt->toPlainText().toStdString()+","+OrgQt2->toPlainText().toStdString()+"]";
    }

    if(SzQt->toPlainText()!="" && SzQt2->toPlainText()!=""){
        SzVar="Sz=["+SzQt->toPlainText().toStdString()+","+SzQt2->toPlainText().toStdString()+"]";
    }

    cmd = "mm3d SateLib CropRPC " +oriFileCropTxt_str+" "+patternTxt_str+" "+ directoryOutputTxt_str +" "+ SzVar +" "+ OrgVar +" @ExitOnBrkp";

    msgBox2.setWindowTitle("mm3d command");

    const QString cmd_str = QString::fromStdString(cmd);
    msgBox2.setText("Here is the commande you are about to launch : \n " + cmd_str);
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



    }else{
        images.clear();
        SzVar=" ";
        OrgVar=" ";
        oriFileCropTxt_str= " ";
        patternTxt_str= " ";
        directoryOutputTxt_str= " ";


    }
}



QString CropRPC::getTxt(){
    return sendCons;

}


void CropRPC::msg(){
    qDi.setWindowTitle("CropRPC state");
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




QString CropRPC::msg1(){
    return sendCons;
}




void CropRPC::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}


void CropRPC::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();


}




void CropRPC::oriSearch(){
    fichier = QFileDialog::getOpenFileName(0, ("Select Output Folder"), aPath);
    QString name = fichier.path()+"/"+fichier.fileName();
    if(!fichier.fileName().isEmpty()){
        QMessageBox msgBox;
        msgBox.setWindowTitle("title");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + name);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);
        if(msgBox.exec() == QMessageBox::Yes){
            oriFileCropTxt->setText(name);
        }else{
        }}
}



void CropRPC::oriSearch1(){
    fichier = QFileDialog::getOpenFileName(0, ("Select Output Folder"), aPath);
    QString name = aPath+"/"+fichier.fileName();
    if(!fichier.fileName().isEmpty()){
        QMessageBox msgBox;
        msgBox.setWindowTitle("title");
        msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + name);
        msgBox.setStandardButtons(QMessageBox::Yes);
        msgBox.addButton(QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::No);
        if(msgBox.exec() == QMessageBox::Yes){
            patternTxt->setText(name);
        }else{
        }}
}


