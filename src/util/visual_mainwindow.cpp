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
    mainWidget = new QWidget(this);
    setCentralWidget(mainWidget);

    QVBoxLayout *verticalLayout = new QVBoxLayout(mainWidget);

    mainWidget->setLayout(verticalLayout);

    toolBox = new QToolBox();

    verticalLayout->addWidget(toolBox);

    addGridLayout(aVAM, tr("Mandatory arguments"));

    if (aVAO.size())
    {
        addGridLayout(aVAO, tr("Optional arguments"));

        connect(toolBox, SIGNAL(currentChanged(int)), this, SLOT(_adjustSize(int)));
    }

    runCommandButton = new QPushButton(tr(" Run command "), mainWidget);

    verticalLayout->addWidget(runCommandButton, 1, Qt::AlignRight);

    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));
}

visual_MainWindow::~visual_MainWindow()
{
    delete mainWidget;
    delete label;
    delete Combo;
    delete select_LineEdit;
    delete select_Button;
    delete runCommandButton;
}

void visual_MainWindow::addGridLayout(vector<cMMSpecArg>& aVA, QString pageName)
{
    QWidget* mPage = new QWidget();

    toolBox->addItem(mPage, pageName);

    QGridLayout* gridLayout = new QGridLayout(mPage);

    buildUI(aVA, gridLayout, mPage);
}

void visual_MainWindow::buildUI(vector<cMMSpecArg>& aVA, QGridLayout *layout, QWidget *parent)
{
    for (int aK=0 ; aK<int(aVA.size()) ; aK++)
    {
        cMMSpecArg aArg = aVA[aK];
        cout << "arg " << aK << " ; Type is " << aArg.NameType() << " ; Name is " << aArg.NameArg() <<"\n";

        create_comment(layout, parent, aK, aArg.Comment());

        if (aArg.NameType() == "string")
        {
            if (!aArg.EnumeratedValues().empty()) //valeurs enumerees dans une liste
            {
                create_combo(layout, parent, aK, aArg);
            }
            else //chaine de caracteres normale
            {
                create_select(layout, parent, aK, aArg);
            }
        }
        else if (aArg.NameType()== "INT")
        {
            create_champ_int(layout, parent, aK, aArg);
        }
        else if (aArg.IsBool())
        {
            create_combo(layout, parent, aK, aArg);
        }

        //ShowEnum(aVA[aK]);
    }

    QSpacerItem *spacerV = new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    layout->addItem(spacerV, aVA.size(), 0);
}

void visual_MainWindow::onRunCommandPressed()
{
    cout << "-----------------" << endl;

    string aCom = MM3dBinFile(argv_recup) +" ";
    for (unsigned int i=0;i<vInputs.size();i++)
    {
        string aAdd;

        cInputs* aIn = vInputs[i];

        switch(aIn->Type())
        {
            case eLineEdit:
            {
                string aStr = ((QLineEdit*) aIn->Widget())->text().toStdString();
                if (!aStr.empty()) aAdd += aStr;
                break;
            }
            case eComboBox:
            {
                QString aStr = ((QComboBox*) aIn->Widget())->currentText();

                if (aStr == tr("True")) aStr = "1";
                else if (aStr == tr("False")) aStr = "0";

                aAdd += aStr.toStdString();
                break;
            }
            case eInteger:
            {
                int val = ((QSpinBox*) aIn->Widget())->value();
                stringstream ss;
                ss << val;
                aAdd +=  ss.str();
                break;
            }
        }

        if (!aAdd.empty())
        {
            if (aIn->Arg().IsOpt()) aCom += aIn->Arg().NameArg()+ "=" + aAdd + " ";
            else aCom += aAdd + " ";
        }
    }

    cout << "Com = " << aCom << endl;

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
    QString filename = QFileDialog::getOpenFileName(mainWidget, tr("Select file"), mlastDir);

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
    QString aDir = QFileDialog::getExistingDirectory( mainWidget, tr("Select directory"), mlastDir);

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

void visual_MainWindow::create_combo(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    std::list<std::string> liste_valeur_enum = listPossibleValues(aArg);

    QComboBox* aCombo = new QComboBox(parent);
    vEnumValues.push_back(aCombo);
    layout->addWidget(aCombo,aK,1,1,1);
    //vInputTypes.push_back(eComboBox);
    // vInputs.push_back(new cInput(aCombo);
    vInputs.push_back(new cInputs(aArg, eComboBox, aCombo));

    for (list<string>::const_iterator it= liste_valeur_enum.begin();
         it != liste_valeur_enum.end();
         it++)
    {
        aCombo->addItem(QString((*it).c_str()));
    }
}

void visual_MainWindow::create_comment(QGridLayout* layout, QWidget* parent, int ak, string str_com)
{
    QLabel * com = new QLabel(QString(str_com.c_str()), parent);
    layout->addWidget(com,ak,0,1,1);
}

void visual_MainWindow::create_select(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aSAM)
{
    QLineEdit* aLineEdit = new QLineEdit(parent);
    vLineEdit.push_back(aLineEdit);
    layout->addWidget(aLineEdit,aK,1,1,1);

    if (!aSAM.IsOutputFile() && !aSAM.IsOutputDirOri())
    {
        select_Button = new selectionButton(parent);
        layout->addWidget(select_Button,aK,3,1,1);

        if (aSAM.IsExistDirOri())
        {
            select_Button->setText(tr("Select directory"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));
        }
        else if (aSAM.IsPatFile())
        {
            select_Button->setText(tr("Select images"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectImgsPressed(int)));
        }
        else if (aSAM.IsExistFile())
        {
            select_Button->setText(tr("Select file"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
        }
    }

    //vInputTypes.push_back(eLineEdit);
    //vInputs.push_back(aLineEdit);

    vInputs.push_back(new cInputs(aSAM, eLineEdit, aLineEdit));
}

void visual_MainWindow::create_champ_int(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *aSpinBox = new QSpinBox(parent);
    layout->addWidget(aSpinBox,aK,1,1,1);
    //vInputTypes.push_back(eInteger);
    //vInputs.push_back(aSpinBox);
    vInputs.push_back(new cInputs(aArg, eInteger, aSpinBox));
}

void visual_MainWindow::set_argv_recup(string argv)
{
    argv_recup = argv;
}

void visual_MainWindow::resizeEvent(QResizeEvent *)
{
    //deplacement au centre de l'ecran
    const QPoint global = qApp->desktop()->availableGeometry().center();
    move(global.x() - width() / 2, global.y() - height() / 2);
}

cInputs::cInputs(cMMSpecArg aArg, int aType, QWidget *aWid):
    mArg(aArg),
    mType(aType),
    mWidget(aWid)
{

}

#endif //ELISE_QT5





