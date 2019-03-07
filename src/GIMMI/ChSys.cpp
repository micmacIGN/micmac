


#include "MainWindow.h"
#include "NewProject.h"
#include "ChSys.h"



ChSys::ChSys(QString path) : QWidget()
{
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    setWindowTitle("ChSys");
    QVBoxLayout *boxLayoutV = new QVBoxLayout;
    chemin = path.toStdString().c_str();
    QLabel *rPCFolder = new QLabel(this);
    rPCFolder->setText("Enter the informations concerning your photos:");
    rPCFolder->setStyleSheet("font-weight: bold");
    boxLayoutV->addWidget(rPCFolder);

    QLabel *utm = new QLabel(this);
    utm->setText("UTM:");
    utmQt = new QTextEdit();
    QHBoxLayout *boxlayouth0 = new QHBoxLayout;
    boxlayouth0->addWidget(utm);
    boxlayouth0->addWidget(utmQt);
    utmQt->setFixedWidth(300);
    utmQt->setFixedHeight(25);
    boxLayoutV->addLayout(boxlayouth0);

    boxSouth = new QCheckBox("South");
    boxLayoutV->addLayout(boxlayouth0);
    boxLayoutV->addWidget(boxSouth);

    this->setLayout(boxLayoutV);

    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);
    boxLayoutV->addWidget(ok);
    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));
    this->setLayout(boxLayoutV);


}







void ChSys::mm3d(){

    if(utmQt->toPlainText().toStdString()!="")
    {
        if(boxSouth->isChecked())
        {

            south=utmQt->toPlainText().toStdString() + "+south";
            std::cout <<  south << std::endl;
        }
        else{
            south=utmQt->toPlainText().toStdString();
            std::cout <<  south << std::endl;

        }

    }


    string const utm(chemin+"/WGS84toUTM.XML");

    ofstream utmFlux(utm.c_str());


    if(utmFlux)

    {

        utmFlux << "<?xml version=\"1.0\" ?>\n"
                   "<SystemeCoord>\n"
                   "     <BSC>\n"
                   "          <TypeCoord>eTC_Proj4</TypeCoord>\n"
                   "          <AuxStr>+proj=utm +zone="+south+" +ellps=WGS84 +datum=WGS84 +units=m +no_defs</AuxStr>\n"
                                                              "     </BSC>\n"
                                                              "</SystemeCoord>";
    }
    else
    {
        cout << "ERREUR: Impossible d'ouvrir le fichier." << endl;
    }


    std::cout << utm << std::endl;

    this->close();
}




