#if(ELISE_QT5)

#include "general/visual_mainwindow.h"
#include "general/visual_buttons.h"

#include "StdAfx.h"

visual_MainWindow::visual_MainWindow(vector<cMMSpecArg> & aVAM, vector<cMMSpecArg> & aVAO, QWidget *parent) :
    QMainWindow(parent),
    mlastDir(QDir::homePath())
{
    gridLayoutWidget = new QWidget(this);
    setCentralWidget(gridLayoutWidget);

    gridLayout = new QGridLayout(gridLayoutWidget);
    gridLayoutWidget->setLayout(gridLayout);

    runCommandButton = new QPushButton("Run command", gridLayoutWidget);
    gridLayout->addWidget(new QLabel(" "),5,1,1,1);
    gridLayout->addWidget(runCommandButton,6,1,1,1);
    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));

    //list of all arguments
    // aVAM: Mandatory args
    // aVAO: Optional args

    //std::cout << "---------- All arguments ----------" << std::endl;

    for (int aK=0 ; aK<int(aVAM.size()) ; aK++)
    {
        cMMSpecArg aArg = aVAM[aK];
        //cout << "Mandatory arg " << aK << " ; Type is " << aArg.NameType();
        std::string aComment = aArg.Comment();
        create_comment(aComment,aK);
        /*if (aComment != "")
        {
            std::cout << "   Comment=" << aCom << "\n";
        }
        else
        {
            std::cout<<"\n";
        }*/

        //Si le type est une string
        if (aArg.NameType() == "string")
        {
            //On recupere les valeurs enumerees dans une liste
            std::list<std::string> liste_valeur_enum = listPossibleValues(aArg);

            if (!liste_valeur_enum.empty())
            {
                create_combo(aK,liste_valeur_enum);
            }
            else //Si c'est une chaine de caracteres normale
            {
                if (aArg.IsPatFile())
                {
                    create_select_images(aK);
                }
                else if (aArg.IsExistDirOri())
                {
                    create_select_orientation(aK);
                }
                else if (aArg.IsExistFile())
                {
                    //TODO
                }
            }
        }
        //Si le type est int
        if (aArg.NameType() =="INT"||aArg.NameType() =="int")
        {
            create_champ_int(aK);
        }
    }

    for (int aK=0 ; aK<int(aVAO.size()) ; aK++)
    {
        //std::cout <<  "Optional arg type is " <<aVAO[aK].NameArg()  << " ; "  << aVAO[aK].NameType();

        /*std::string aComment = aVAO[aK].Comment();
        if (aComment != "") std::cout << "Commentaire ["<< aK << "] = " << aComment  << "\n";*/

        bool apat = aVAO[aK].IsExistFile();
        if (apat) std::cout<< "opt arg is a file: " << apat << std::endl;

        //ShowEnum(aVAO[aK]);
    }

    //std::cout<<"---------- End all arguments ----------"<<std::endl;
}


visual_MainWindow::~visual_MainWindow()
{
    delete gridLayoutWidget;
    delete gridLayout;
    delete label;
    delete Combo;
    delete select_LineEdit;
    delete select_Button;
    delete runCommandButton;
}

void visual_MainWindow::onRunCommandPressed()
{
    cout<<"-----------------" << endl;

    string aCom = MM3dBinFile(argv_recup) +" ";
    for (unsigned int i=0;i<vInputs.size();i++)
    {
        //cout<<"inputTypes: " << inputTypes[i] <<" "<<inputs[i]<<endl;

        switch(vInputTypes[i])
        {
            case eLineEdit:
            {
                aCom += ((QLineEdit*)vInputs[i])->text().toStdString();
                break;
            }
            case eComboBox:
            {
                aCom += ((QComboBox*)vInputs[i])->currentText().toStdString();
                break;
            }
            case eInteger:
            {
                int val = ((QSpinBox*)vInputs[i])->value();
                stringstream ss;
                ss << val;
                aCom +=  ss.str();
                break;
            }
        }

        aCom += " ";
    }

    cout << aCom << endl;

    int aRes = ::System(aCom);

    cout << "----------------- " << aRes << endl;
}

void visual_MainWindow::onSelectFilePressed(int aK)
{
    string full_pattern;
    QStringList files = QFileDialog::getOpenFileNames(
                            gridLayoutWidget,
                            tr("Select images"),
                            mlastDir,
                            tr("Images (*.png *.xpm *.jpg *.tif)"));

    string aDir, aNameFile;
    SplitDirAndFile(aDir,aNameFile,files[0].toStdString());
    mlastDir = QString(aDir.c_str());

    string fileList="("+ aNameFile;
    for (int i=1;i<files.length();i++)
    {
        SplitDirAndFile(aDir,aNameFile,files[i].toStdString());

        fileList.append("|" + aNameFile);
    }

    fileList+=")";
    full_pattern = QUOTE(aDir+fileList);

    vLineEdit[aK]->setText(QString(full_pattern.c_str()));
    //cout<<full_pattern.toStdString()<<endl;

   /* int maxW = 0;
    for (int bK=0; bK < (int) vLineEdit.size();++bK)
    {
        QString text = vLineEdit[bK]->text();
        QFontMetrics fm = vLineEdit[bK]->fontMetrics();
        int w = fm.boundingRect(text).width();
        if (w > maxW) maxW = w;
    }
    for (int bK=0; bK < (int) vLineEdit.size();++bK)
    {
        vLineEdit[bK]->resize(maxW, vLineEdit[bK]->height());
    }*/
}

void visual_MainWindow::onSelectDirPressed(int aK)
{
    QString aDir = QFileDialog::getExistingDirectory(
                            gridLayoutWidget,
                            tr("Select directory"),
                            mlastDir);

    mlastDir = aDir;

    vLineEdit[aK]->setText(QDir(aDir).dirName());
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
    vInputTypes.push_back(eComboBox);
    vInputs.push_back(aCombo);

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
    vLineEdit.push_back(aLineEdit);
    gridLayout->addWidget(aLineEdit,aK,1,1,1);
    create_select_button(aK);
    vInputTypes.push_back(eLineEdit);
    vInputs.push_back(aLineEdit);
}

void visual_MainWindow::create_select_orientation(int aK)
{
    QLineEdit* aLineEdit = new QLineEdit(gridLayoutWidget);
    vLineEdit.push_back(aLineEdit);
    gridLayout->addWidget(aLineEdit,aK,1,1,1);
    create_select_button(aK, true);
    vInputTypes.push_back(eLineEdit);
    vInputs.push_back(aLineEdit);
}

void visual_MainWindow::create_comment(string str_com, int aK)
{
    QLabel * com = new QLabel(gridLayoutWidget);
    vComments.push_back(com);
    gridLayout->addWidget(com,aK,0,1,1);
    com->setText(QString(str_com.c_str()));
}

void visual_MainWindow::create_select_button(int aK, bool isDir)
{
    QString buttonName = isDir ? tr("Select directory") : tr("Select images");

    select_Button = new imgListButton(buttonName, gridLayoutWidget);
    gridLayout->addWidget(select_Button,aK,3,1,1);

    if (isDir)

        connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));

    else

        connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
}

void visual_MainWindow::create_champ_int(int aK)
{
    QSpinBox *aSpinBox = new QSpinBox(gridLayoutWidget);
    gridLayout->addWidget(aSpinBox,aK,1,1,1);
    vInputTypes.push_back(eInteger);
    vInputs.push_back(aSpinBox);
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


