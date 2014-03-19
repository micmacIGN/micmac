#if(ELISE_QT5)

#include "general/visual_mainwindow.h"
#include "general/visual_buttons.h"

#include "StdAfx.h"

visual_MainWindow::visual_MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    gridLayoutWidget = new QWidget(this);
    setCentralWidget(gridLayoutWidget);

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


    runCommandButton = new QPushButton("Run command", gridLayoutWidget);
    gridLayout->addWidget(runCommandButton,5,1,1,1);
    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));
}


visual_MainWindow::~visual_MainWindow()
{
    delete gridLayoutWidget;
    delete gridLayout;
    delete label;
    delete Combo;
    delete selectFile_LineEdit;
    delete selectFile_Button;
    delete runCommandButton;
}

void visual_MainWindow::onRunCommandPressed()
{
    cout<<"-----------------"<<endl;
    QString commande="mm3d "+QString(argv_recup.c_str())+" ";
    for (unsigned int i=0;i<inputs.size();i++)
    {
        cout<<inputTypes[i]<<" "<<inputs[i]<<endl;

        switch(inputTypes[i])
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

    cout<<commande.toStdString()<<endl;

    cout<<"-----------------"<<endl;
}

void visual_MainWindow::onSelectFilePressed(int aK)
{
    QString full_pattern=0;
    QStringList files = QFileDialog::getOpenFileNames(
                            gridLayoutWidget,
                            tr("Select images"),
                            "/home",
                            "Images (*.png *.xpm *.jpg *.tif)");

    string aDir, aNameFile;
    SplitDirAndFile(aDir,aNameFile,files[0].toStdString());

    QString fileList="("+QString(aNameFile.c_str());
    for (int i=1;i<files.length();i++)
    {
        SplitDirAndFile(aDir,aNameFile,files[i].toStdString());

        fileList.append("|" + QString(aNameFile.c_str()));
    }
    fileList+=")";
    full_pattern = QString(aDir.c_str())+fileList;

    vImageFiles[aK]->setText(full_pattern);
    //cout<<full_pattern.toStdString()<<endl;
    commande += " " + full_pattern;
}

void visual_MainWindow::add_combo_line(QString str)
{
    QComboBox* combo = vEnumValues.back();
    combo->addItem(str);
}

void visual_MainWindow::create_combo(int aK, list<string> liste_valeur_enum )
{
    QComboBox* aCombo = new QComboBox(gridLayoutWidget);
    vEnumValues.push_back(aCombo);
    gridLayout->addWidget(aCombo,aK,1,1,1);
    inputTypes.push_back(combobox);
    inputs.push_back(aCombo);

    for (
         list<string>::const_iterator it= liste_valeur_enum.begin();
         it != liste_valeur_enum.end();
         it++)
    {
        add_combo_line(QString((*it).c_str()));
    }
}

void visual_MainWindow::create_select_images(int aK)
{
    QLineEdit* aLineEdit = new QLineEdit(gridLayoutWidget);
    vImageFiles.push_back(aLineEdit);
    gridLayout->addWidget(aLineEdit,aK,1,1,1);
    create_selectFile_button(aK);
    inputTypes.push_back(lineedit);
    inputs.push_back(aLineEdit);
}

void visual_MainWindow::create_select_orientation(int aK)
{
    QLineEdit* aLineEdit = new QLineEdit(gridLayoutWidget);
    vImageFiles.push_back(aLineEdit);
    gridLayout->addWidget(aLineEdit,aK,1,1,1);
    create_selectFile_button(aK);
    inputTypes.push_back(lineedit);
    inputs.push_back(aLineEdit);
}

void visual_MainWindow::create_comment(string str_com, int aK)
{
    QLabel * com = new QLabel(gridLayoutWidget);
    vCommentaries.push_back(com);
    gridLayout->addWidget(com,aK,0,1,1);
    com->setText(QString(str_com.c_str()));
}

void visual_MainWindow::create_selectFile_button(int aK)
{
    selectFile_Button = new imgListButton("Select images", gridLayoutWidget);
    gridLayout->addWidget(selectFile_Button,aK,3,1,1);
    connect(selectFile_Button,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
}

void visual_MainWindow::create_champ_int(int aK)
{
    QSpinBox *aSpinBox = new QSpinBox(gridLayoutWidget);
    //vImageFiles.push_back(aSpinBox);
    gridLayout->addWidget(aSpinBox,aK,1,1,1);
    inputTypes.push_back(integer);
    inputs.push_back(aSpinBox);
}

void visual_MainWindow::set_argv_recup(string argv)
{
    argv_recup = argv;
}

void visual_MainWindow::resizeEvent(QResizeEvent *)
{
    const QPoint global = qApp->desktop()->availableGeometry().center();
    move(global.x() - width() / 2, global.y() - height() / 2);
}

#endif //ELISE_QT5


