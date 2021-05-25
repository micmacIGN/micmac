
#include "MainWindow.h"
#include "NewProject.h"
#include "Aperi.h"



Aperi::Aperi(QList <QString> list_all, QList <QString> list,QList <QString> listCheckedItems, QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    this->setFixedWidth(200);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    setWindowTitle("AperiCloud");
    path_s = path.toStdString();
    QVBoxLayout *boxLayoutV = new QVBoxLayout;

    QLabel *labelMode = new QLabel(this);
    labelMode->setText("Orientation :");
    labelMode->setStyleSheet("font-weight: bold");


    for (int i = 0; i < list.count(); i++)
    {
        std::string mot = list[i].toStdString();
        if(     (mot.find(".JPG")<10000) ||
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

        for (int y = 0; y < listImages.count(); y++)
         if(listImages[i]==listCheckedItems[y]){
             qDebug() << listImages[i] + "      listImages";
             qDebug() << listCheckedItems[i] + "listCheckedItems";
           imagesCheckBox->setChecked(true);
           y=listImages.count();
         }
     }

    boxLayoutV->addWidget(labelMode);

    groupBox = new QComboBox();

    for (int i = 0; i < list2.count(); i++)
  {
        groupBox->addItem(list2[i]);
    }



    boxLayoutV->addWidget(groupBox);
    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
    this->setLayout(boxLayoutV);
}



void Aperi::mm3d(){


    images.erase();
    for( int i = 0; i< listImages.size(); i++){
            images.append(listImages.at(i).toStdString()+"|");
    }

    images = images.erase(images.size()-1);
    mode = groupBox->currentText().toStdString();

   std::cout <<  "mode = " + mode << std::endl;

    cmd = "mm3d AperiCloud \""+ images +"\" "+mode+" @ExitOnBrkp";
        msgBox2.setWindowTitle("Aperi");
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
            connect(&p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));
            msg();
            this->close();
          }

}


QString Aperi::msg1(){
return sendCons;
        }

void Aperi::msg(){

           qDi.setWindowTitle("Aperi state");
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


void Aperi::rpcFile(){
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


void Aperi::readyReadStandardOutput(){
  Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}



void Aperi::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();
      }
