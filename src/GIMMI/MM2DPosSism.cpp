
#include "MainWindow.h"
#include "NewProject.h"
#include "MM2DPosSism.h"



MM2DPosSism::MM2DPosSism(QList <QString> list,QList <QString> listCheckedItems, QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    aPath = path;

    setWindowTitle("MM2DPosSism");
    path_s = path.toStdString();
    QVBoxLayout *boxLayoutV = new QVBoxLayout;

    QLabel *labelImage1 = new QLabel(this);
    labelImage1->setText("Select your first ortho-image:");
    labelImage1->setStyleSheet("font-weight: bold");

    QLabel *labelImage2 = new QLabel(this);
    labelImage2->setText("Select your second ortho-image:");
    labelImage2->setStyleSheet("font-weight: bold");


    labelImageTxt1 = new QTextEdit();
    labelImageTxt1->setFixedWidth(600);
    labelImageTxt1->setFixedHeight(27);

    labelImageTxt2 = new QTextEdit();
    labelImageTxt2->setFixedWidth(600);
    labelImageTxt2->setFixedHeight(27);


    QHBoxLayout *boxlayouth1 = new QHBoxLayout;
    QPushButton *buttonImage1 = new QPushButton();
    buttonImage1->setText("...");
    buttonImage1->setFixedWidth(30);
    buttonImage1->setFixedHeight(30);
    QObject::connect(buttonImage1, SIGNAL(clicked()), this, SLOT(oriSearch1()));

    QHBoxLayout *boxlayouth2 = new QHBoxLayout;
    QPushButton *buttonImage2 = new QPushButton();
    buttonImage2->setText("...");
    buttonImage2->setFixedWidth(30);
    buttonImage2->setFixedHeight(30);
    QObject::connect(buttonImage2, SIGNAL(clicked()), this, SLOT(oriSearch2()));


    QLabel *xML = new QLabel(this);
    xML->setText("Options :");
    xML->setStyleSheet("font-weight: bold");

    QLabel *DirMECLabel = new QLabel(this);
    DirMECLabel->setText("DirMEC :");
    DirMECLabel->setStyleSheet("font-weight: bold");

    dirMec = new QTextEdit();
    dirMec->setFixedHeight(25);

    boxLayoutV->addWidget(labelImage1);
    boxlayouth1->addWidget(labelImageTxt1);
    boxlayouth1->addWidget(buttonImage1);
    boxLayoutV->addLayout(boxlayouth1);

    boxLayoutV->addWidget(labelImage2);
    boxlayouth2->addWidget(labelImageTxt2);
    boxlayouth2->addWidget(buttonImage2);
    boxLayoutV->addLayout(boxlayouth2);;



    boxLayoutV->addWidget(xML);
    boxXml = new QCheckBox("Dequant=false");
    boxLayoutV->addWidget(boxXml);

    QHBoxLayout *boxLayouth = new QHBoxLayout;
    boxLayouth->addWidget(DirMECLabel);
    boxLayouth->addWidget(dirMec);
    boxLayoutV->addLayout(boxLayouth);

    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
    this->setLayout(boxLayoutV);
}





QString MM2DPosSism::msg1(){
    return sendCons;
}





void MM2DPosSism::mm3d(){

    if(labelImageTxt1->toPlainText().isEmpty() || labelImageTxt2->toPlainText().isEmpty()){
        msgBox2.setWindowTitle("MM2DPosSism Error");
        msgBox2.setText("Please, Select 2 images");
        msgBox2.setStandardButtons(QMessageBox::Ok);
        msgBox2.setDefaultButton(QMessageBox::Ok);
        if(msgBox2.exec() == QMessageBox::Ok){
        }

    }else{

        if(boxXml->isChecked())
        {    txt="Dequant=false";
        }



        if(!dirMec->toPlainText().isEmpty()){
            dirMecToStdStrg="DirMEC="+dirMec->toPlainText().toStdString();
        }

        cmd = "mm3d MM2DPosSism "+ labelImageTxt1->toPlainText().toStdString() +" "+ labelImageTxt2->toPlainText().toStdString() +" "+ txt +" "+ dirMecToStdStrg +" @ExitOnBrkp";
        msgBox2.setWindowTitle("title");
        const QString cmd_str = QString::fromStdString(cmd);
        msgBox2.setText("Here is the commande you're about to launch : \n " + cmd_str);
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

        }
    }}





void MM2DPosSism::msg(){
    qDi.setWindowTitle("MM2DPosSism state");

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

void MM2DPosSism::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));

}


void MM2DPosSism::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();
}


void MM2DPosSism::oriSearch1(){
    fichier = QFileDialog::getOpenFileName(0, ("Select Output Folder"), aPath);
    QString name = aPath+"/"+fichier.fileName();
    if(!fichier.fileName().isEmpty()){

    QMessageBox msgBox;
    msgBox.setWindowTitle("title");
    msgBox.setText("Here is the image you selected. Do you confirm your choice? \n" + name);
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if(msgBox.exec() == QMessageBox::Yes){
        labelImageTxt1->setText(name);
    }else{
    }
}}

void MM2DPosSism::oriSearch2(){
    fichier = QFileDialog::getOpenFileName(0, ("Select Output Folder"), aPath);
    QString name = aPath+"/"+fichier.fileName();
    if(!fichier.fileName().isEmpty()){

    QMessageBox msgBox;
    msgBox.setWindowTitle("title");
    msgBox.setText("Here is the image you selected. Do you confirm your choice? \n" + name);
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if(msgBox.exec() == QMessageBox::Yes){
        labelImageTxt2->setText(name);
    }else{
    }
}}


