
#include "MainWindow.h"
#include "NewProject.h"
#include "Sel.h"



Sel::Sel(QList <QString> list,QList <QString> listCheckedItems, QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    setWindowTitle("SEL");
    path_s = path.toStdString();
    QVBoxLayout *boxLayoutV = new QVBoxLayout;

    QLabel *labelImages = new QLabel(this);
    labelImages->setText("Select 2 images for the tie points visualisation :");
    labelImages->setStyleSheet("font-weight: bold");

    QLabel *xML = new QLabel(this);
    xML->setText("Check this box if you want a few set of tie points and save in XML format :");
    xML->setStyleSheet("font-weight: bold");

    boxLayoutV->addWidget(labelImages);

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
        boxLayoutV->addWidget(imagesCheckBox);
        imagesCheckBox->setChecked(false);
    }

    boxLayoutV->addWidget(xML);
    boxXml = new QCheckBox("KH=S");
    boxLayoutV->addWidget(boxXml);
    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
    this->setLayout(boxLayoutV);
}





QString Sel::msg1(){
    return sendCons= cons.lab->toPlainText();
}





void Sel::mm3d(){

    int a = 0;
    images.erase();
    std::cout <<  "images = "  << std::endl;
    for( int i = 0; i< listeCasesImages.size(); i++){
        if(listeCasesImages.at(i)->isChecked())
        {a+=1;
            images.append(listImages.at(i).toStdString()+" ");
        }
    }


    if(a!=2){
        msgBox2.setWindowTitle("SEL Error");

        msgBox2.setText("Please, select 2 images \n ");
        msgBox2.setStandardButtons(QMessageBox::Ok);
        msgBox2.setDefaultButton(QMessageBox::Ok);
        if(msgBox2.exec() == QMessageBox::Ok){
       }

    }else{

        std::cout <<  images << std::endl;
        if(boxXml->isChecked())
        {    txt="KH=S";
        }else
        {
            txt="KH=NB";
        }
        cmd = "mm3d SEL ./ "+ images + txt ;
        msgBox2.setWindowTitle("SEL");
        const QString cmd_str = QString::fromStdString(cmd);
        msgBox2.setText("Here is the commande you're about to launch : \n " + cmd_str);
        msgBox2.setStandardButtons(QMessageBox::Yes);
        msgBox2.addButton(QMessageBox::No);
        msgBox2.setDefaultButton(QMessageBox::No);
        if(msgBox2.exec() == QMessageBox::Yes){
            p.setWorkingDirectory(path_s.c_str());
            p.waitForFinished(-1);
            p.start(cmd.c_str());
            p.setReadChannel(QProcess::StandardOutput);
            this->close();
        }
    }}



void Sel::msg(){
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
