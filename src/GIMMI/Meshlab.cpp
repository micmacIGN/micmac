

#include "MainWindow.h"
#include "NewProject.h"
#include "Meshlab.h"



Meshlab::Meshlab() : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    setWindowTitle("Meshlab");
    QVBoxLayout *boxLayoutV = new QVBoxLayout;


    QLabel *rPCFolder = new QLabel(this);
    rPCFolder->setText("Select the file you would like to open :");
    rPCFolder->setStyleSheet("font-weight: bold");

    boxLayoutV->addWidget(rPCFolder);
    rPC = new QTextEdit();
    rPC->setFixedHeight(25);
    QHBoxLayout *boxlayouth = new QHBoxLayout;
    QPushButton *rpc = new QPushButton();
    QObject::connect(rpc, SIGNAL(clicked()), this, SLOT(rpcFile()));

    rpc->setText("...");
    rpc->setFixedWidth(30);
    rpc->setFixedHeight(30);

    boxlayouth->addWidget(rPC);
    boxlayouth->addWidget(rpc);
    boxLayoutV->addLayout(boxlayouth);



    this->setLayout(boxLayoutV);


    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
    this->setLayout(boxLayoutV);


}





QString Meshlab::msg1(){
return sendCons= cons.lab->toPlainText();
        }





void Meshlab::mm3d(){

    if(rPC->toPlainText().isEmpty()){
        msgBox2.setWindowTitle("title");

        msgBox2.setText("Please choose a valid file \n ");
        msgBox2.setStandardButtons(QMessageBox::Ok);
        msgBox2.setDefaultButton(QMessageBox::Ok);
        if(msgBox2.exec() == QMessageBox::Ok){



            }




    }else{

//std::cout <<  pyFile << std::endl;


std::string  pyFile ;

pyFile = rPC->toPlainText().toStdString();


 cmd = "meshlab "+ pyFile  ;

    msgBox2.setWindowTitle("Meshlab Visualisation ");

    const QString cmd_str = QString::fromStdString(cmd);
    msgBox2.setText("Here is the commande you're about to launch : \n " + cmd_str);
    msgBox2.setStandardButtons(QMessageBox::Yes);
    msgBox2.addButton(QMessageBox::No);
    msgBox2.setDefaultButton(QMessageBox::No);
    if(msgBox2.exec() == QMessageBox::Yes){

        p.setWorkingDirectory(path_s.c_str());
        p.waitForFinished(-1);
        qDebug() <<  QDir::currentPath();
        qDebug() <<  cmd.c_str() ;
        p.start(cmd.c_str());
        p.setReadChannel(QProcess::StandardOutput);
        this->close();



      }
}}



void Meshlab::rpcFile(){
  fichier = QFileDialog::getOpenFileName(0, ("Select File"));

QString name = fichier.absoluteFilePath();
    QMessageBox msgBox;
    msgBox.setWindowTitle("title");
    msgBox.setText("Do you confirm that the selected folder where your images have to be loaded is the following one? \n" + name);
    msgBox.setStandardButtons(QMessageBox::Yes);
    msgBox.addButton(QMessageBox::No);
    msgBox.setDefaultButton(QMessageBox::No);
    if(msgBox.exec() == QMessageBox::Yes){
rPC->setText(name);
    }else {        }
}





void Meshlab::msg(){
    QMessageBox msgBox;
         msgBox.setWindowTitle("Tapioca state");
         msgBox.setText("Push Ok to Stop  \n");

         msgBox.setStandardButtons(QMessageBox::Ok);

         msgBox.setDefaultButton(QMessageBox::Ok);
         if(msgBox.exec() == QMessageBox::Ok){


             p.kill();
             this->deleteLater();


         }else {


    }}
