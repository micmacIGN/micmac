
#include "Console.h"



Console::Console() : QWidget()
{


    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active
    setWindowTitle("Console");
    QVBoxLayout *boxLayoutV = new QVBoxLayout;
    QLabel *labelCmd = new QLabel(this);
    labelCmd->setText("Commande");
    boxLayoutV->addWidget(labelCmd);
    this->setGeometry(100,100,800,600);
    lab = new QTextEdit;
    boxLayoutV->addWidget(lab);
    this->setLayout(boxLayoutV);


}
void Console::setLabCons(){


}

