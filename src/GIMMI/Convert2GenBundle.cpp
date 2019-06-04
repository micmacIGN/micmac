
#include "Convert2GenBundle.h"

Convert2GenBundle::Convert2GenBundle(QList <QString> list_all ,QList <QString> list, QList <QString> listCheckedItems, QString path)  : QWidget()
{

    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    sendCons="";
    path_s = path.toStdString();
    setWindowTitle("Convert2GenBundle");
    qDebug()<< path;
    QVBoxLayout *boxlayoutv = new QVBoxLayout;


    QLabel *labelImages = new QLabel(this);
    labelImages->setText("Choose the images that you want to convert :");
    labelImages->setStyleSheet("font-weight: bold");

    QLabel *labelMode1 = new QLabel(this);
    labelMode1->setText("RPC file :");
    labelMode1->setStyleSheet("font-weight: bold");

    QLabel *labelMode3 = new QLabel(this);
    labelMode3->setText("Directory of output Orientation :");
    labelMode3->setStyleSheet("font-weight: bold");

    QLabel *labelOptions = new QLabel(this);
    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");

    QLabel *ChSys = new QLabel(this);
    ChSys->setText("ChSys");
    ChSysQt = new QTextEdit();
    QHBoxLayout *boxlayouth1 = new QHBoxLayout;
    boxlayouth1->addWidget(ChSys);

    QLabel *Degre = new QLabel(this);
    Degre->setText("Degre");
    DegreQt = new QTextEdit();
    QHBoxLayout *boxlayouth2 = new QHBoxLayout;
    boxlayouth2->addWidget(Degre);
    boxlayouth2->addWidget(DegreQt);
    DegreQt->setFixedWidth(300);
    DegreQt->setFixedHeight(27);

    QLabel *Type = new QLabel(this);
    Type->setText("Type");
    TypeQt = new QTextEdit();
    QHBoxLayout *boxlayouth3 = new QHBoxLayout;
    boxlayouth3->addWidget(Type);
    boxlayouth3->addWidget(TypeQt);
    TypeQt->setFixedWidth(300);
    TypeQt->setFixedHeight(27);

    QLabel *PertubAng = new QLabel(this);
    PertubAng->setText("PertubAng");
    PertubAngQt = new QTextEdit();
    QHBoxLayout *boxlayouth4 = new QHBoxLayout;
    boxlayouth4->addWidget(PertubAng);
    boxlayouth4->addWidget(PertubAngQt);
    PertubAngQt->setFixedWidth(300);
    PertubAngQt->setFixedHeight(27);


    //   boxlayoutv->addWidget(labelImages);

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
        if(mot.find(".xml")<10000 || mot.find(".XML")<10000 ){
            list2.append(list_all[i]);
            qDebug()<<list2;
        }}


    for (int i = 0; i < list_all.count(); i++)
    {
        std::string mot = list_all[i].toStdString();
        if(mot.find("Ori-")<10000){
            list3.append(list_all[i]);
            qDebug()<<list3;
        }}



    groupBox1 = new QComboBox();
    groupBox2 = new QComboBox();
    groupBox3 = new QComboBox();
    groupBox4 = new QComboBox();
    groupBox5 = new QComboBox();
    groupBox5->setFixedWidth(300);

    outPut=new QTextEdit;
    outPut->setFixedHeight(27);

    for (int i = 0; i < list2.count(); i++)
    {
        groupBox1->addItem(list2[i]);
    }


    for (int i = 0; i < list3.count(); i++)
    {
        groupBox2->addItem(list3[i]);
    }


    for (int i = 0; i < list2.count(); i++)
    {
        groupBox3->addItem(list2[i]);
    }


    for (int i = 0; i < listImages.count(); i++)
    {
        groupBox4->addItem(listImages[i]);
    }


    groupBox5->addItem("");

    for (int i = 0; i < list2.count(); i++)
    {
        groupBox5->addItem(list2[i]);
    }
    boxlayouth1->addWidget(groupBox5);

    boxlayoutv->addWidget(labelImages);
    boxlayoutv->addWidget(groupBox4);

    boxlayoutv->addWidget(labelMode1);
    boxlayoutv->addWidget(groupBox1);

    boxlayoutv->addWidget(labelMode3);
    boxlayoutv->addWidget(outPut);


    boxlayoutv->addWidget(labelOptions);

    QScrollArea *scrollArea = new QScrollArea;

    scrollArea->setFixedSize(430, 300);
    QWidget *layoutWidget = new QWidget(this);
    QVBoxLayout *boxlayoutscroll = new QVBoxLayout;

    layoutWidget->setLayout(boxlayoutscroll);

    boxlayoutscroll->addLayout(boxlayouth1);
    boxlayoutscroll->addLayout(boxlayouth2);
    boxlayoutscroll->addLayout(boxlayouth3);
    boxlayoutscroll->addLayout(boxlayouth4);

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


void Convert2GenBundle::mm3d(){





    //        images = images.erase(images.size()-1);
    //        std::cout <<  images << std::endl;

    modo1 = groupBox1->currentText().toStdString();
    modo2 = groupBox2->currentText().toStdString();
    modo3 = groupBox3->currentText().toStdString();
    images = groupBox4->currentText().toStdString();

    if(groupBox5->currentText().toStdString()!=""){
        ChSysVar = "ChSys="+ groupBox5->currentText().toStdString();
    }


    if(outPut->toPlainText()!=""){

        outPutCampStd = outPut->toPlainText().toStdString();
    }
    if(TypeQt->toPlainText()!=""){

        TypeVar = "Type="+TypeQt->toPlainText().toStdString();
    }

    if(DegreQt->toPlainText()!=""){

        DegreVar = "Degre="+DegreQt->toPlainText().toStdString();

    }




    if(PertubAngQt->toPlainText()!=""){

        PertubAngVar = "PertubAng="+PertubAngQt->toPlainText().toStdString();

    }



    cmd = "mm3d Convert2GenBundle "+ images +" "+ modo1 +" "+labelAutreVar.c_str()+ " "+ outPutCampStd+ " "
            + TypeVar.c_str()+ " " +
            + DegreVar.c_str()+ " " +
            + ChSysVar.c_str()+ " " +
            + PertubAngVar.c_str()+ " " +

            +"@ExitOnBrkp";

    std::cout <<  cmd << std::endl;




    QMessageBox msgBox2;
    msgBox2.setWindowTitle("title");
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
        outPutCampStd="";
        TypeVar="";
        DegreVar="";
        ChSysVar="";
        PertubAngVar="";}

    // do something else


}


QString Convert2GenBundle::getTxt(){

    return p_stdout;

}


void Convert2GenBundle::msg(){
    qDi.setWindowTitle("Convert2GenBundle state");

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



void Convert2GenBundle::rpcFile(){
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





QString Convert2GenBundle::msg1(){
    return sendCons;
}


void Convert2GenBundle::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();


}

void Convert2GenBundle::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));
}
