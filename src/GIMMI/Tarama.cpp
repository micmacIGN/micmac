
#include "Tarama.h"

Tarama::Tarama(QList <QString> list_all ,QList <QString> list, QList <QString> listCheckedItems, QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    sendCons="";
    path_s = path.toStdString();
    setWindowTitle("Tarama");
    qDebug()<< path;
    QVBoxLayout *boxlayoutv = new QVBoxLayout;
//    QLabel *labelImages = new QLabel(this);
//    labelImages->setText("Choose the images that you want to use :");
//    labelImages->setStyleSheet("font-weight: bold");



    QLabel *labelMode1 = new QLabel(this);
    labelMode1->setText("Orientation :");
    labelMode1->setStyleSheet("font-weight: bold");

    QLabel *outPutCamp = new QLabel(this);
    outPutCamp->setText("Zoom:");
    outPutCamp->setStyleSheet("font-weight: bold");

    QLabel *labelOptions = new QLabel(this);
    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");


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

    for (int i = 0; i < list2.count(); i++)
  {
        groupBox1->addItem(list2[i]);
    }

    boxlayoutv->addWidget(labelMode1);
    boxlayoutv->addWidget(groupBox1);


    boxlayoutv->addWidget(outPutCamp);
    outPutCampT = new QTextEdit();
    outPutCampT->setFixedHeight(30);
    outPutCampT->setFixedWidth(300);


    boxlayoutv->addWidget(outPutCampT);



    outPutCamp->setFixedHeight(25);



    boxlayoutv->addWidget(labelOptions);
    boxTxt = new QCheckBox("Txt");
    boxlayoutv->addWidget(boxTxt);

    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);



    boxlayoutv->addWidget(ok);




    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));

    this->setLayout(boxlayoutv);





}


void Tarama::mm3d(){


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

outPutCampStd = outPutCampT->toPlainText().toStdString();
std::cout <<  "Size m = " + outPutCampStd << std::endl;

if(!outPutCampT->toPlainText().isEmpty())
{

    outPutCampStd= "Zoom="+ outPutCampStd;

}

modo = groupBox1->currentText().toStdString();

std::cout <<  "mode = " + modo << std::endl;


if(boxTxt->isChecked())
{

    txt="ExpTxt=1";
    std::cout <<  "Options = " + txt << std::endl;

}

//cmd = "mm3d Tarama " + mode +" \""+ images +"\" "+ outPutCampStd +" "+ modo +" "+ txt;
cmd = "mm3d Tarama \""+ images +"\" "+ modo +" "+ outPutCampStd +" "+ txt +" @ExitOnBrkp";

std::cout <<  cmd << std::endl;




QMessageBox msgBox2;
    msgBox2.setWindowTitle("Tarama");
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


            }else{ images.clear();}

        // do something else



}}




QString Tarama::getTxt(){

    return p_stdout;

}

void Tarama::msg(){
    qDi.setWindowTitle("Tarama state");

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


void Tarama::rpcFile(){
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





QString Tarama::msg1(){
return sendCons;
        }


void Tarama::stopped(){
    sendCons= Qtr->toPlainText();
qDi.accept();
this->close();


      }

void Tarama::readyReadStandardOutput(){
  Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}

