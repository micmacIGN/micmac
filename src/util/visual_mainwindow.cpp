#if(ELISE_QT5)

#include "general/visual_mainwindow.h"
#include "general/visual_buttons.h"

#include "StdAfx.h"

// aVAM: Mandatory args
// aVAO: Optional args

visual_MainWindow::visual_MainWindow(vector<cMMSpecArg> & aVAM, vector<cMMSpecArg> & aVAO, QWidget *parent) :
    QMainWindow(parent),
    mlastDir(QDir::homePath())
{
    bList.push_back("True");
    bList.push_back("False");

    mainWidget = new QWidget(this);
    setCentralWidget(mainWidget);

    QVBoxLayout *verticalLayout = new QVBoxLayout(mainWidget);

    mainWidget->setLayout(verticalLayout);

    toolBox = new QToolBox();

    verticalLayout->addWidget(toolBox);

    QWidget* pageMandatoryArgs = new QWidget();
    pageMandatoryArgs->setGeometry(QRect(0, 0, 300, 400));
    pageMandatoryArgs->setLayoutDirection(Qt::LeftToRight);

    //Grid Layout
    gridLayout = new QGridLayout(pageMandatoryArgs);

    toolBox->addItem(pageMandatoryArgs, tr("Mandatory arguments"));

    QWidget* pageOptionalArgs = new QWidget();
    pageOptionalArgs->setGeometry(QRect(0, 0, 300, 400));
    pageOptionalArgs->setLayoutDirection(Qt::LeftToRight);

    toolBox->addItem(pageOptionalArgs, tr("Optional arguments"));

    QGridLayout* gridLayout_2 = new QGridLayout(pageOptionalArgs);

    buildUI(aVAM, gridLayout, pageMandatoryArgs);

    buildUI(aVAO, gridLayout_2, pageOptionalArgs, true);

    runCommandButton = new QPushButton(" Run command ", mainWidget);

    verticalLayout->addWidget(runCommandButton, 1, Qt::AlignRight);

    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));

    connect(toolBox, SIGNAL(currentChanged(int)), this, SLOT(_adjustSize(int)));
}

visual_MainWindow::~visual_MainWindow()
{
    delete mainWidget;
    delete gridLayout;
    delete label;
    delete Combo;
    delete select_LineEdit;
    delete select_Button;
    delete runCommandButton;
}

void visual_MainWindow::buildUI(vector<cMMSpecArg>& aVA, QGridLayout *layout, QWidget *parent, bool isOpt)
{
    for (int aK=0 ; aK<int(aVA.size()) ; aK++)
    {
        cMMSpecArg aArg = aVA[aK];
        //cout << "arg " << aK << " ; Type is " << aArg.NameType() <<"\n";

        create_comment(layout, parent, aArg.Comment(), aK);

        if (aArg.NameType() == "string")
        {
            //On recupere les valeurs enumerees dans une liste
            std::list<std::string> liste_valeur_enum = listPossibleValues(aArg);

            if (!liste_valeur_enum.empty())
            {
                create_combo(layout, parent, aK, liste_valeur_enum);
            }
            else //chaine de caracteres normale
            {
                create_select(layout, parent, aK, aArg);
            }
        }
        else if (aArg.NameType()== "INT")
        {
            create_champ_int(layout, parent, aK);
        }
        else if (aArg.NameType()== "bool")
        {
            create_combo(layout, parent, aK, bList);
        }

        //ShowEnum(aVA[aK]);
    }

    QSpacerItem *spacerV = new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    layout->addItem(spacerV, aVA.size(), 0);

    if (isOpt)
    {
        QSpacerItem *spacerH = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        layout->addItem(spacerH, 0, layout->columnCount());
    }
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

void visual_MainWindow::onSelectImgsPressed(int aK)
{
    string full_pattern;
    QStringList files = QFileDialog::getOpenFileNames(
                            mainWidget,
                            tr("Select images"),
                            mlastDir,
                            tr("Images (*.png *.xpm *.jpg *.tif)"));

    if (files.size())
    {
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

        adjustSize();
    }
}

void visual_MainWindow::onSelectFilePressed(int aK)
{
    QString filename = QFileDialog::getOpenFileName(mainWidget,
                                                    tr("Select file"),
                                                    mlastDir);

    if (filename != NULL)
    {
        string aDir, aNameFile;
        SplitDirAndFile(aDir,aNameFile,filename.toStdString());
        mlastDir = QString(aDir.c_str());

        vLineEdit[aK]->setText(filename);

        adjustSize();
    }
}

void visual_MainWindow::onSelectDirPressed(int aK)
{
    QString aDir = QFileDialog::getExistingDirectory(
                            mainWidget,
                            tr("Select directory"),
                            mlastDir);

    if (aDir != NULL)
    {
        mlastDir = aDir;

        vLineEdit[aK]->setText(QDir(aDir).dirName());

        adjustSize();
    }
}

void visual_MainWindow::_adjustSize(int)
{
    adjustSize();
}

void visual_MainWindow::create_combo(QGridLayout* layout, QWidget* parent, int aK, list<string> liste_valeur_enum )
{
    QComboBox* aCombo = new QComboBox(parent);
    vEnumValues.push_back(aCombo);
    layout->addWidget(aCombo,aK,1,1,1);
    vInputTypes.push_back(eComboBox);
    vInputs.push_back(aCombo);

    for (list<string>::const_iterator it= liste_valeur_enum.begin();
         it != liste_valeur_enum.end();
         it++)
    {
        aCombo->addItem(QString((*it).c_str()));
    }
}

void visual_MainWindow::create_comment(QGridLayout* layout, QWidget* parent, string str_com, int ak)
{
    QLabel * com = new QLabel(QString(str_com.c_str()), parent);
    //vComments.push_back(com);
    layout->addWidget(com,ak,0,1,1);
}

void visual_MainWindow::create_select(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg eSAM)
{
    QLineEdit* aLineEdit = new QLineEdit(parent);
    vLineEdit.push_back(aLineEdit);
    layout->addWidget(aLineEdit,aK,1,1,1);

    if (!eSAM.IsOutputFile())
    {
        select_Button = new selectionButton(parent);
        layout->addWidget(select_Button,aK,3,1,1);

        if (eSAM.IsExistDirOri())
        {
            select_Button->setText(tr("Select directory"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));
        }
        else if (eSAM.IsPatFile())
        {
            select_Button->setText(tr("Select images"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectImgsPressed(int)));
        }
        else if (eSAM.IsExistFile())
        {
            select_Button->setText(tr("Select file"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
        }
    }

    vInputTypes.push_back(eLineEdit);
    vInputs.push_back(aLineEdit);
}

void visual_MainWindow::create_champ_int(QGridLayout* layout, QWidget* parent, int aK)
{
    QSpinBox *aSpinBox = new QSpinBox(parent);
    layout->addWidget(aSpinBox,aK,1,1,1);
    vInputTypes.push_back(eInteger);
    vInputs.push_back(aSpinBox);
}

void visual_MainWindow::set_argv_recup(string argv)
{
    argv_recup = argv;
}

QSize getLayoutCellSize(QGridLayout *layout, int row, int column)
{
    QLayoutItem *item = layout->itemAtPosition(row, column);
    if (item)
        return (item->sizeHint());
    return (QSize());
}

template <class T>
int  getWidgetVectorWidth(vector < T* > vWid)
{
    int max = -1;

    for (int aK=0; aK < (int)  vWid.size();++aK)
    {
        QString text =  vWid[aK]->text();
        QFontMetrics fm =  vWid[aK]->fontMetrics();
        int w = fm.boundingRect(text).width();
        if (w > max) max = w;
    }
    return max;
}

void visual_MainWindow::resizeEvent(QResizeEvent *)
{
    QRect screenSz = qApp->desktop()->availableGeometry();

    int maxLineEdit = getWidgetVectorWidth(vLineEdit);
    if (maxLineEdit <= 0) maxLineEdit = 100;

    int maxComment = 0;
    int maxButton = 0;

    //calcul de la taille des colonnes 1 et 3... (Commentaires et boutons)
    for (int aK=0; aK < gridLayout->rowCount(); ++aK)
    {
        QSize cellSize1 = getLayoutCellSize(gridLayout, aK, 1);
        QSize cellSize3 = getLayoutCellSize(gridLayout, aK, 3);

        if (cellSize1.isValid() && (cellSize1.width() > maxComment)) maxComment = cellSize1.width();
        if (cellSize3.isValid() && (cellSize3.width() > maxButton))  maxButton  = cellSize3.width();
    }

    int finalSize = maxLineEdit + maxComment + maxButton + 60;
    if (finalSize > screenSz.width()) finalSize = screenSz.width() - 50;

    if (toolBox->currentIndex() == 0)
        resize(finalSize, 200);
    else
        resize(finalSize, 500);

    //deplacement au centre de l'ecran
    const QPoint global = screenSz.center();
    move(global.x() - width() / 2, global.y() - height() / 2);
}


#endif //ELISE_QT5


