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

        create_comment(layout, parent, aK, aArg);

        switch (aArg.Type())
        {
        case AMBT_Box2di:

            break;
        case AMBT_Box2dr:
            break;
        case AMBT_bool:
            create_combo(layout, parent, aK, aArg);
            break;
        case AMBT_INT:
        case AMBT_U_INT1:
            create_spinBox(layout, parent, aK, aArg);
            break;
        case AMBT_REAL:
            create_dSpinBox(layout, parent, aK, aArg);
        break;
        case AMBT_Pt2di:
        break;
        case AMBT_Pt2dr:
        break;
        case AMBT_Pt3dr:
        break;
        case AMBT_Pt3di:
        break;
        case AMBT_string:
        {
            if (!aArg.EnumeratedValues().empty()) //valeurs enumerees dans une liste
            {
                create_combo(layout, parent, aK, aArg);
            }
            else //chaine de caracteres normale
            {
                create_select(layout, parent, aK, aArg);
            }
            break;
        }
        case AMBT_INT1:
        case AMBT_char:
        break;
        case AMBT_vector_Pt2dr:
        break;
        case AMBT_vector_int:
        break;
        case AMBT_vector_double:
        break;
        case AMBT_vvector_int:
        break;
        case AMBT_vector_string:
        break;
        case AMBT_unknown:
            break;
        }

        //ShowEnum(aVA[aK]);
    }

    QSpacerItem *spacerV = new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    layout->addItem(spacerV, aVA.size(), 0);
}

void visual_MainWindow::onRunCommandPressed()
{
    bool runCom = true;

    string aCom = MM3dBinFile(argv_recup) +" ";
    for (unsigned int aK=0;aK<vInputs.size();aK++)
    {
        string aAdd;

        cInputs* aIn = vInputs[aK];

        switch(aIn->Type())
        {
            case eLineEdit:
            {
                string aStr = ((QLineEdit*) aIn->Widget())->text().toStdString();
                if (!aStr.empty()) aAdd += aStr;
                else if (!aIn->Arg().IsOpt())
                {
                    QMessageBox::critical(this, tr("Error"), tr("Mandatory argument not filled!!!"));
                    runCom = false;
                }
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
                aAdd += ss.str();
                break;
            }
            case eDouble:
            {
                double val = ((QDoubleSpinBox*) aIn->Widget())->value();
                stringstream ss;
                ss << val;
                aAdd += ss.str();
                break;
            }
        }

        if (!aAdd.empty())
        {
            if (aIn->Arg().IsOpt()) aCom += aIn->Arg().NameArg()+ "=" + aAdd + " ";
            else aCom += aAdd + " ";
        }
    }

    if (runCom)
    {
        cout << "Com = " << aCom << endl;
        int aRes = ::System(aCom);

        cout << "----------------- " << aRes << endl;
    }
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
    list<string> liste_valeur_enum = listPossibleValues(aArg);

    QComboBox* aCombo = new QComboBox(parent);
    vEnumValues.push_back(aCombo);
    layout->addWidget(aCombo,aK,1,1,1);

    vInputs.push_back(new cInputs(aArg, eComboBox, aCombo));

    list<string>::const_iterator it = liste_valeur_enum.begin();
    for (; it != liste_valeur_enum.end(); it++)
    {
        aCombo->addItem(QString((*it).c_str()));
    }

    if (aArg.IsBool())
    {
        bool aBool = aArg.DefaultValue<bool>();

        if (aBool) aCombo->setCurrentIndex(0);
        else       aCombo->setCurrentIndex(1);
    }
    else
    {
       //string aStr = aArg.DefaultValue<string>(); =>seg fault car pas de valeur par default (ex: vTapas)
       /*  int idx = -1;
        int cpt = 0;
        list<string>::const_iterator it = liste_valeur_enum.begin();
        for (; it != liste_valeur_enum.end(); it++, cpt++)
        {
            if (aStr == *it) idx = cpt;
        }
        if (idx >= 0) aCombo->setCurrentIndex(idx);*/
    }
}

void visual_MainWindow::create_comment(QGridLayout* layout, QWidget* parent, int ak, cMMSpecArg aArg)
{
    QLabel * com = new QLabel(QString(aArg.Comment().c_str()), parent);
    layout->addWidget(com,ak,0,1,1);
}

void visual_MainWindow::create_select(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    QLineEdit* aLineEdit = new QLineEdit(parent);
    vLineEdit.push_back(aLineEdit);
    layout->addWidget(aLineEdit,aK,1,1,1);

    if (!aArg.IsOutputFile() && !aArg.IsOutputDirOri())
    {
        select_Button = new selectionButton(parent);
        layout->addWidget(select_Button,aK,3,1,1);

        if (aArg.IsExistDirOri())
        {
            select_Button->setText(tr("Select directory"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));
        }
        else if (aArg.IsPatFile())
        {
            select_Button->setText(tr("Select images"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectImgsPressed(int)));
        }
        else if (aArg.IsExistFile())
        {
            select_Button->setText(tr("Select file"));
            connect(select_Button,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
        }
    }

    vInputs.push_back(new cInputs(aArg, eLineEdit, aLineEdit));
}

void visual_MainWindow::create_dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *aSpinBox = new QDoubleSpinBox(parent);
    layout->addWidget(aSpinBox,aK,1,1,1);

    aSpinBox->setValue( aArg.DefaultValue<double>() );

    vInputs.push_back(new cInputs(aArg, eDouble, aSpinBox));
}

void visual_MainWindow::create_spinBox(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *aSpinBox = new QSpinBox(parent);
    layout->addWidget(aSpinBox,aK,1,1,1);

    aSpinBox->setValue( aArg.DefaultValue<int>() );

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





