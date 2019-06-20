


#include "MainWindow.h"
#include "NewProject.h"
#include "Cmd.h"



Cmd::Cmd(QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);

    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    path_s = path.toStdString();
    setWindowTitle("Command");
    QVBoxLayout *boxLayoutV = new QVBoxLayout;
    QLabel *rPCFolder = new QLabel(this);
    rPCFolder->setText("Enter a mm3d commmand :");
    rPCFolder->setStyleSheet("font-weight: bold");
    boxLayoutV->addWidget(rPCFolder);
    rPC = new QTextEdit();
    rPC->setFixedHeight(25);
    boxLayoutV->addWidget(rPC);

    this->setLayout(boxLayoutV);
    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));

    this->setLayout(boxLayoutV);


}






void Cmd::mm3d(){

    if(rPC->toPlainText().isEmpty()){
        msgBox2.setWindowTitle("Command Error");
        msgBox2.setText("Please, enter a valid command \n ");
        msgBox2.setStandardButtons(QMessageBox::Ok);
        msgBox2.setDefaultButton(QMessageBox::Ok);
        if(msgBox2.exec() == QMessageBox::Ok){



        }




    }else{

        //std::cout <<  pyFile << std::endl;


        std::string  pyFile ;
        pyFile = rPC->toPlainText().toStdString();
        cmd = pyFile + " ";
        msgBox2.setWindowTitle("Command ");
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



void Cmd::rpcFile(){
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


void Cmd::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();


}




void Cmd::msg(){
    qDi.setWindowTitle("Cmd state");

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



QString Cmd::msg1(){
    return sendCons;
}



void Cmd::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}
