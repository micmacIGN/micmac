#if(ELISE_QT5)

#include "general/visual_mainwindow.h"
#include "general/mes_boutons.h"
#include "StdAfx.h"
#include <iostream>
#include <QFileDialog>


visual_MainWindow::visual_MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    resize(400, 300);

    gridLayoutWidget = new QWidget(this);
    this->setCentralWidget(gridLayoutWidget);

    gridLayout = new QGridLayout(gridLayoutWidget);
    gridLayoutWidget->setLayout(gridLayout);

    //label = new QLabel(gridLayoutWidget);
    //gridLayout->addWidget(label, 0, 0, 1, 1);
    //label->setText("Saisie de la commande Tapas...");


    //gridLayout->setRowStretch(0, 1);
    //gridLayout->setRowStretch(4, 1);

    //vecteur_Commentaires.push_back(new QLabel(gridLayoutWidget));
    //gridLayout->addWidget(vecteur_Commentaires.back(), 1, 0, 1, 1);
    //vecteur_Commentaires.back()->setText("Modèle de Calibration");

    //Combo = new QComboBox(gridLayoutWidget);
    //vecteur_val_enumerees.push_back(Combo);
    //gridLayout->addWidget(Combo, 2,0,1,1);

//    vecteur_Commentaires.push_back(new QLabel(gridLayoutWidget));
//    gridLayout->addWidget(vecteur_Commentaires.back(), 3, 0, 1, 1);
//    vecteur_Commentaires.back()->setText("Chemin des images");

//    Sel_Fichiers = new QLineEdit(gridLayoutWidget);
//    gridLayout->addWidget(Sel_Fichiers,4,0,1,1);

//    Parcourir = new QPushButton(gridLayoutWidget);
//    gridLayout->addWidget(Parcourir,4,1,1,1);
//    Parcourir->setText("Parcourir");
//    connect(Parcourir,SIGNAL(clicked()),this,SLOT(press_parcours()));


    Valider = new QPushButton(gridLayoutWidget);
    Valider->setText("Valider");
    gridLayout->addWidget(Valider,5,1,1,1);
    connect(Valider,SIGNAL(clicked()),this,SLOT(press_valider()));
}


visual_MainWindow::~visual_MainWindow()
{
    delete gridLayoutWidget;
    delete gridLayout;
    delete label;
    delete Combo;
    delete Sel_Fichiers;
    delete Parcourir;
    delete Valider;
}

void visual_MainWindow::press_valider()
{
    std::cout<<"-----------------"<<std::endl;
    QString commande="mm3d "+QString(argv_recup.c_str())+" ";
    for (unsigned int i=0;i<inputs.size();i++)
    {
        std::cout<<types_inputs[i]<<" "<<inputs[i]<<std::endl;

        switch(types_inputs[i])
        {
            case lineedit:
            {
                commande += ((QLineEdit*)inputs[i])->text();
                break;
            }
            case combobox:
            {
                commande += ((QComboBox*)inputs[i])->currentText();
                break;
            }
            case integer:
            {
            commande += QString("%1").arg( ((QSpinBox*)inputs[i])->value());
                break;
            }

        }
        commande += " ";

    }

    std::cout<<commande.toStdString()<<std::endl;

    std::cout<<"-----------------"<<std::endl;


}

void visual_MainWindow::press_parcours(int aK)
{
    QString full_pattern=0;
    QStringList files = QFileDialog::getOpenFileNames(
                            gridLayoutWidget,
                            "Sélectionnez vos images",
                            "/home",
                            "Images (*.png *.xpm *.jpg)");
    std::string Dossier=files[0].toStdString();
    int fin_directory = Dossier.find_last_of("/")+1;
    std::string Dossier_parent = Dossier.substr(0,fin_directory);
    //std::cout<<Dossier_parent<<std::endl;

    QString fichiers="("+QString(files[0].toStdString().substr(fin_directory,std::string::npos).c_str());
    for (int i=1;i<files.length();i++){
        fichiers+="|";
        fichiers.append(QString(files[i].toStdString().substr(fin_directory,std::string::npos).c_str()));
    }
    fichiers+=")";
    full_pattern = QString(Dossier_parent.c_str())+fichiers;

    vecteur_Fichiers_images[aK]->setText(full_pattern);
    //std::cout<<full_pattern.toStdString()<<std::endl;
    commande+=" "+full_pattern;
}

void visual_MainWindow::ajoute_ligne_combo(QString str)
{
    QComboBox* combo = vecteur_val_enumerees.back();
    combo->addItem(str);
}

void visual_MainWindow::create_combo(int aK, std::list<std::string> liste_valeur_enum )
{
    QComboBox* nom_combo = new QComboBox(gridLayoutWidget);
    vecteur_val_enumerees.push_back(nom_combo);
    gridLayout->addWidget(nom_combo,aK,1,1,1);
    types_inputs.push_back(combobox);
    inputs.push_back(nom_combo);

    for (
         std::list<std::string>::const_iterator val_enum= liste_valeur_enum.begin();
         val_enum != liste_valeur_enum.end();
         val_enum++)
    {
        std::string nom_enum = *val_enum;
        QString qnom=QString(nom_enum.c_str());
        ajoute_ligne_combo(qnom);
    }
}

void visual_MainWindow::create_select_images(int aK)
{
    QLineEdit* nom_sel_fichier = new QLineEdit(gridLayoutWidget);
    vecteur_Fichiers_images.push_back(nom_sel_fichier);
    gridLayout->addWidget(nom_sel_fichier,aK,1,1,1);
    create_bouton_parcourir(aK);
    types_inputs.push_back(lineedit);
    inputs.push_back(nom_sel_fichier);
}

void visual_MainWindow::create_select_orientation(int aK)
{
    QLineEdit* nom_sel_fichier = new QLineEdit(gridLayoutWidget);
    vecteur_Fichiers_images.push_back(nom_sel_fichier);
    gridLayout->addWidget(nom_sel_fichier,aK,1,1,1);
    create_bouton_parcourir(aK);
    types_inputs.push_back(lineedit);
    inputs.push_back(nom_sel_fichier);
}





void visual_MainWindow::create_comment(std::string str_com, int aK)
{
    QLabel * com = new QLabel(gridLayoutWidget);
    vecteur_Commentaires.push_back(com);
    gridLayout->addWidget(com,aK,0,1,1);
    com->setText(QString(str_com.c_str()));
}

void visual_MainWindow::create_bouton_parcourir(int aK)
{
    Parcourir = new mon_bouton_parcours(gridLayoutWidget);
    gridLayout->addWidget(Parcourir,aK,3,1,1);
    Parcourir->setText("Parcourir");
    connect(Parcourir,SIGNAL(mon_click(int)),this,SLOT(press_parcours(int)));

}

void visual_MainWindow::create_champ_int(int aK)
{
    QSpinBox *nom_champ = new QSpinBox(gridLayoutWidget);
    //vecteur_Fichiers_images.push_back(nom_champ);
    gridLayout->addWidget(nom_champ,aK,1,1,1);
    types_inputs.push_back(integer);
    inputs.push_back(nom_champ);
}




void visual_MainWindow::set_argv_recup(std::string argv)
{
    argv_recup = argv;
}

#endif //ELISE_QT5


