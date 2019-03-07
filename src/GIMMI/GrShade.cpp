
#include "MainWindow.h"
#include "NewProject.h"
#include "GrShade.h"



GrShade::GrShade(QString pathImg) : QWidget()
{

    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active


    cmd = "mm3d GrShade " + pathImg.toStdString() + " @ExitOnBrkp";

    msgBox2.setWindowTitle("GrShade");

    const QString cmd_str = QString::fromStdString(cmd);
    msgBox2.setText("Here is the commande you're about to launch : \n " + cmd_str);
    msgBox2.setStandardButtons(QMessageBox::Yes);
    msgBox2.addButton(QMessageBox::No);
    msgBox2.setDefaultButton(QMessageBox::No);
    if(msgBox2.exec() == QMessageBox::Yes){
        this->deleteLater();
        p.start(cmd.c_str());
        p.setReadChannel(QProcess::StandardOutput);
        connect(&p,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
        connect(&p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));  //Add this line will cause error at runtime
        msg();




    }}

void GrShade::msg(){
    qDi.setWindowTitle("GrShade state");

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

void GrShade::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
    }


void GrShade::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();

}


QString GrShade::msg1(){
    return sendCons;
    }


